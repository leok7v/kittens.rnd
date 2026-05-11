#!/usr/bin/env python3
"""
End-to-end TTS through the ggml backend: text -> audio.

Orchestrates four kittens-tts modes (textstage, genfront, decoder, fullgen)
plus host-side phonemization and length regulation, then writes a WAV file.

Usage:
    python scripts/tts_e2e.py --text "Hello world" --voice expr-voice-5-m \
        --out tmp/tts_e2e.wav
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from real_calibration import build_vocab, espeak_ipa, phonemize  # noqa: E402

GGUF = REPO_ROOT / "tmp/kitten_full.gguf"
TMP  = REPO_ROOT / "tmp"
BIN_CPU   = TMP / "kittens-tts-cpu"
BIN_METAL = TMP / "kittens-tts-metal"
SR = 24000


def call_kt(mode: str, in_path: Path, out_path: Path, backend: str) -> None:
    bin_path = BIN_METAL if backend == "metal" else BIN_CPU
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", mode,
           "--input", str(in_path), "--output", str(out_path),
           "--backend", backend]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, end="")
        print(r.stderr, end="", file=sys.stderr)
        raise SystemExit(f"{mode} exited {r.returncode}")
    # Surface timing
    for line in r.stderr.splitlines():
        if "ms" in line or "graph nodes" in line:
            print("  ", line.strip())


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def fade_in(samples: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or n > len(samples): return samples
    t = np.arange(n, dtype=np.float32) / max(1, n - 1)
    samples[:n] *= 0.5 - 0.5 * np.cos(np.pi * t)
    return samples


def fade_out(samples: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or n > len(samples): return samples
    t = np.arange(n, dtype=np.float32) / max(1, n - 1)
    samples[-n:] *= 0.5 + 0.5 * np.cos(np.pi * t)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello world.")
    ap.add_argument("--voice", default="expr-voice-5-m")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "metal"])
    ap.add_argument("--out", default=str(TMP / "tts_e2e.wav"))
    args = ap.parse_args()

    vocab = build_vocab()
    print(f"[1/5] phonemize: '{args.text}'")
    print(f"      espeak ipa: {espeak_ipa(args.text)}")
    ids = phonemize(args.text, vocab)
    L = len(ids)
    print(f"      ids[:8]={ids[:8]} (L={L})")

    # Load voice from voices.npz. Swift uses min(chunk.count, 399) with
    # chunk.count = TEXT character count (not phoneme count) — see
    # TTS.CoreML.swift `let refId = min(chunk.count, 399)`.
    voices = np.load(REPO_ROOT / "scripts/models/voices.npz")
    if args.voice not in voices.files:
        raise SystemExit(f"voice '{args.voice}' not in {list(voices.files)}")
    vrows = voices[args.voice].astype(np.float32)   # (400, 256) typically
    ref = min(len(args.text), 399)
    style256 = vrows[ref].copy()                   # (256,)
    print(f"      style256.shape={style256.shape}  ref={ref} (text chars)")

    # ---- Stage 1: TextStage ----
    print(f"[2/5] textstage L={L}")
    in_path = TMP / "e2e_ts_in.bin"
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", L))
        f.write(np.array(ids, dtype=np.int32).tobytes())
        f.write(style256.astype(np.float32).tobytes())
    out_path = TMP / "e2e_ts_out.bin"
    call_kt("textstage", in_path, out_path, args.backend)

    raw = np.fromfile(out_path, dtype=np.float32)
    p_n = 256 * L; t_n = 128 * L; d_n = 50 * L
    prosody_nlc = raw[:p_n].reshape(L, 256).T          # (256, L)
    text_nlc    = raw[p_n:p_n+t_n].reshape(L, 128).T   # (128, L)
    dur_sig     = raw[p_n+t_n:].reshape(L, 50)         # (L, 50)

    # ---- Compute durations and length-regulate ----
    durs = np.maximum(1, np.round(dur_sig.sum(axis=1) / args.speed)).astype(np.int32)
    F = int(durs.sum())
    print(f"      durations[:8]={durs[:8].tolist()}  total F={F}")

    prosody_lr = np.zeros((256, F), dtype=np.float32)
    text_lr    = np.zeros((128, F), dtype=np.float32)
    j = 0
    for i, d in enumerate(durs):
        prosody_lr[:, j:j+d] = prosody_nlc[:, i:i+1]   # repeat
        text_lr[:,    j:j+d] = text_nlc[:,    i:i+1]
        j += int(d)

    # ---- Stage 2: GenFront -> f0_proj, n_proj ----
    print(f"[3/5] genfront F={F}")
    in_path = TMP / "e2e_gf_in.bin"
    style_pr = style256[128:256]   # prosodic half
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", F))
        f.write(prosody_lr.T.astype(np.float32).tobytes())   # NLC: data[t*256+c]
        f.write(style_pr.astype(np.float32).tobytes())
    out_path = TMP / "e2e_gf_out.bin"
    call_kt("genfront", in_path, out_path, args.backend)

    raw = np.fromfile(out_path, dtype=np.float32)
    f0_proj = raw[:2*F]              # (1, 2F) flattened
    # n_proj is also written but unused by us — fullgen recomputes from f0
    # via compute_noise_contribs.

    # ---- Stage 3: Decoder ----
    print(f"[4/5] decoder")
    in_path = TMP / "e2e_dec_in.bin"
    style_aco = style256[:128]
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", F))
        f.write(text_lr.T.astype(np.float32).tobytes())
        f.write(f0_proj.astype(np.float32).tobytes())
        # n_proj
        f.write(raw[2*F:].astype(np.float32).tobytes())
        f.write(style_aco.astype(np.float32).tobytes())
    out_path = TMP / "e2e_dec_out.bin"
    call_kt("decoder", in_path, out_path, args.backend)

    with out_path.open("rb") as f:
        c, l = struct.unpack("<ii", f.read(8))
        dec_out = np.frombuffer(f.read(), dtype=np.float32).reshape(l, c).T  # (256, 2F)
    print(f"      dec_out shape=({c}, {l})")

    # ---- Stage 4: full generator (with noise) -> audio ----
    print(f"[5/5] fullgen")
    in_path = TMP / "e2e_fg_in.bin"
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", F))
        f.write(dec_out.T.astype(np.float32).tobytes())
        f.write(f0_proj.reshape(1, 2*F).T.astype(np.float32).tobytes())
        f.write(style_aco.astype(np.float32).tobytes())
    out_path = TMP / "e2e_fg_out.bin"
    call_kt("fullgen", in_path, out_path, args.backend)

    with out_path.open("rb") as f:
        T_audio, = struct.unpack("<i", f.read(4))
        audio = np.frombuffer(f.read(), dtype=np.float32).copy()
    print(f"      raw audio T={T_audio} samples ({T_audio/SR:.2f} s)")

    # Trim 3 frames (per LLAMA.BACKEND.md design choice — drop decoder edge)
    tail_drop_samples = 3 * 600   # 3 frames × 600 samples/frame
    audio = audio[: max(0, len(audio) - tail_drop_samples)]
    fade_in (audio, 72)    #  3 ms
    fade_out(audio, 960)   # 40 ms

    write_wav(Path(args.out), audio)
    print(f"\nwrote {args.out}  ({len(audio)/SR:.2f} s, {SR} Hz)")
    print(f"play: afplay {args.out}")


if __name__ == "__main__":
    main()

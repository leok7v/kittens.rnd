#!/usr/bin/env python3
"""Compare WAV outputs sample-by-sample and by spectrogram.

Usage: compare_wavs.py mlx.wav coreml_a.wav coreml_b.wav
Writes a PNG with waveform overlays + spectrograms side-by-side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 24000


def load(path: Path) -> np.ndarray:
    sr, data = wavfile.read(path)
    assert sr == SR, f"{path} sr={sr}"
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    return data.astype(np.float32)


def align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align by cross-correlation lag, then trim to common length."""
    n = min(len(a), len(b))
    win = min(n, SR)  # use first 1 second for lag search
    corr = signal.correlate(b[:win], a[:win], mode="full")
    lag = np.argmax(corr) - (win - 1)
    if lag > 0:
        b_al = b[lag:]
        a_al = a
    elif lag < 0:
        a_al = a[-lag:]
        b_al = b
    else:
        a_al = a
        b_al = b
    m = min(len(a_al), len(b_al))
    return a_al[:m], b_al[:m]


def compare_pair(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    a_al, b_al = align(a, b)
    err = b_al - a_al
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Pick top spectral peak of error signal — indicates tonal artifacts.
    spec = np.abs(np.fft.rfft(err * np.hanning(len(err))))
    freqs = np.fft.rfftfreq(len(err), 1.0 / SR)
    peak_idx = int(np.argmax(spec[1:]) + 1)  # skip DC
    return {
        "name_a": name_a, "name_b": name_b,
        "len": len(err),
        "rmse": rmse,
        "peak_hz": float(freqs[peak_idx]),
        "peak_mag": float(spec[peak_idx]),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: compare_wavs.py REF.wav TEST.wav [TEST2.wav ...]")
        return 1
    ref_path = Path(sys.argv[1])
    test_paths = [Path(p) for p in sys.argv[2:]]
    ref = load(ref_path)
    tests = [(p.stem, load(p)) for p in test_paths]

    print(f"reference: {ref_path.name}  len={len(ref)}  dur={len(ref)/SR:.3f}s")
    print()
    print(f"{'pair':<50} {'len':>7} {'rmse':>7}  {'err-peak-Hz':>12} {'peak-mag':>10}")
    print("-" * 100)
    for name, t in tests:
        r = compare_pair(ref, t, ref_path.stem, name)
        print(f"{r['name_a']:>22} vs {r['name_b']:<22}  {r['len']:>7d}  "
              f"{r['rmse']:.4f}  {r['peak_hz']:>10.1f}  {r['peak_mag']:>10.1f}")

    # Spectrogram plot
    n_panels = 1 + len(tests)
    fig, axes = plt.subplots(n_panels, 2, figsize=(14, 3 * n_panels))
    if n_panels == 1:
        axes = axes[None, :]

    def plot_row(row, name, wav):
        tax = np.arange(len(wav)) / SR
        axes[row][0].plot(tax, wav, linewidth=0.4)
        axes[row][0].set_title(f"{name}  waveform")
        axes[row][0].set_xlim(0, len(wav) / SR)
        axes[row][0].set_ylim(-1.0, 1.0)
        axes[row][0].grid(alpha=0.2)

        f, t, Sxx = signal.spectrogram(wav, fs=SR, nperseg=512,
                                        noverlap=384, scaling="spectrum")
        db = 10 * np.log10(Sxx + 1e-10)
        axes[row][1].imshow(
            db, aspect="auto", origin="lower",
            extent=[t[0], t[-1], f[0], f[-1]],
            cmap="magma", vmin=-100, vmax=-20,
        )
        axes[row][1].set_title(f"{name}  spectrogram")
        axes[row][1].set_ylabel("Hz")
        axes[row][1].set_xlabel("s")

    plot_row(0, ref_path.stem, ref)
    for i, (name, t) in enumerate(tests):
        plot_row(i + 1, name, t)

    plt.tight_layout()
    out = Path("tmp/compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Trace TextStage / GeneratorStage with sample inputs and run coremltools.

coremltools 9 flex shapes (RangeDim) convert successfully but fail at
runtime for any L != the trace-time L (the BERT attention shapes get
baked into MIL constants). So we produce one fixed-shape .mlpackage per
bucket and let the Swift loader pick the smallest bucket >= real length.

Usage:
    python convert_to_coreml.py text      -L 64,128,256
    python convert_to_coreml.py generator -N 128,256,512
    python convert_to_coreml.py both      -L 128 -N 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import TextStage, GeneratorStage, WeightBag  # noqa: E402

import coremltools as ct  # noqa: E402


def _patch_coremltools_int_cast():
    """coremltools 9 _int handler crashes when torch's jit emits aten::Int on
    a single-element array instead of a scalar (seen inside nn.LSTM traces).
    Wrap dtype(x.val) to flatten single-element arrays to scalars first."""
    from coremltools.converters.mil.frontend.torch import ops as _tops
    from coremltools.converters.mil.mil import Builder as _mb
    import numpy as _np

    orig_cast = _tops._cast

    def patched_cast(context, node, dtype, dtype_str):
        x = context[node.inputs[0]]
        if hasattr(x, "val") and x.val is not None and hasattr(x.val, "flatten"):
            arr = _np.asarray(x.val)
            if arr.ndim > 0 and arr.size == 1:
                res = _mb.const(val=dtype(arr.flatten()[0]), name=node.name)
                context.add(res)
                return
        return orig_cast(context, node, dtype, dtype_str)

    _tops._cast = patched_cast


_patch_coremltools_int_cast()


OUT_DIR = Path("scripts/models")


def _sample_style() -> torch.Tensor:
    v = np.load("scripts/models/voices.npz")
    return torch.from_numpy(v["expr-voice-5-m"][32:33].astype(np.float32))


def _precision(variant: str) -> ct.precision:
    return {
        "fp32": ct.precision.FLOAT32,
        "fp16": ct.precision.FLOAT16,
    }[variant]


def convert_text_stage(L: int, save: bool = True, out_dir: Path = OUT_DIR,
                       variant: str = "fp32") -> None:
    print(f"=== TextStage L={L} variant={variant} ===")
    w = WeightBag.load("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    stage = TextStage(w).eval()

    input_ids = torch.randint(1, 170, (1, L), dtype=torch.long)
    # Example mask: first half real, second half pad (exercises the mask path
    # at trace time — the runtime mask will differ per call).
    mask = torch.zeros(1, L, dtype=torch.float32)
    mask[:, : max(1, L // 2)] = 1.0
    style = _sample_style()

    with torch.no_grad():
        p, t, d = stage(input_ids, style, mask)
    print(f"  torch outputs: prosody_ncl={tuple(p.shape)} "
          f"text_features={tuple(t.shape)} dur_sig={tuple(d.shape)}")

    print("  tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(stage, (input_ids, style, mask),
                                 check_trace=False, strict=False)

    print("  converting to Core ML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="style", shape=style.shape, dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=mask.shape, dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=_precision(variant),
    )
    mlmodel.short_description = f"KittenTTS text stage (L={L}, {variant})"
    if save:
        out = out_dir / f"kitten_text_L{L}.mlpackage"
        mlmodel.save(str(out))
        print(f"  saved {out}")
    print()


def convert_generator_stage(n_frames: int, save: bool = True,
                            out_dir: Path = OUT_DIR,
                            variant: str = "fp32") -> None:
    print(f"=== GeneratorStage N={n_frames} variant={variant} ===")
    w = WeightBag.load("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    stage = GeneratorStage(w).eval()

    prosody_lr = torch.randn(1, 256, n_frames)
    text_lr = torch.randn(1, 128, n_frames)
    style = _sample_style()

    with torch.no_grad():
        wav = stage(prosody_lr, text_lr, style)
    print(f"  torch output: waveform={tuple(wav.shape)}")

    print("  tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(stage, (prosody_lr, text_lr, style),
                                 check_trace=False, strict=False)

    print("  converting to Core ML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="prosody_lr", shape=prosody_lr.shape, dtype=np.float32),
            ct.TensorType(name="text_lr",    shape=text_lr.shape,    dtype=np.float32),
            ct.TensorType(name="style",      shape=style.shape,      dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=_precision(variant),
    )
    mlmodel.short_description = f"KittenTTS generator stage (N={n_frames}, {variant})"
    if save:
        out = out_dir / f"kitten_generator_N{n_frames}.mlpackage"
        mlmodel.save(str(out))
        print(f"  saved {out}")
    print()


def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["text", "generator", "both"],
                    default="both", nargs="?")
    ap.add_argument("-L", default="128",
                    help="TextStage bucket(s), comma-separated (e.g. '64,128,256')")
    ap.add_argument("-N", default="256",
                    help="GeneratorStage bucket(s), comma-separated")
    ap.add_argument("--out", default=None,
                    help="output directory (default: scripts/models/<variant>/)")
    ap.add_argument("--variant", default="fp32", choices=["fp32", "fp16"],
                    help="weight + activation precision")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.which in ("text", "both"):
        for L in _parse_int_list(args.L):
            convert_text_stage(L, save=not args.no_save, out_dir=out_dir,
                               variant=args.variant)
    if args.which in ("generator", "both"):
        for N in _parse_int_list(args.N):
            convert_generator_stage(N, save=not args.no_save, out_dir=out_dir,
                                    variant=args.variant)


if __name__ == "__main__":
    main()

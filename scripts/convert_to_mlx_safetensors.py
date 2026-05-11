#!/usr/bin/env python3
"""Plan A: convert TextStage / GeneratorStage to a single dynamic-shape
.mlpackage via torch.export + coremltools.

torch.jit.trace bakes tensor shapes into the MIL graph as constants; that's
why our earlier RangeDim-based flex attempt converted but failed at runtime
for any L ≠ trace-time L. torch.export (PyTorch 2.2+) tracks symbolic
dimensions through the whole graph via torch.fx, and coremltools 7+ can
consume ExportedProgram directly.

Goal: one .mlpackage per stage per quantization variant that accepts any
shape in a documented range. Matches ORT's "one graph, any input length".

Variants:
    fp32   — weights fp32, activations fp16.
    fp16   — weights fp16, activations fp16.
    (int8w / int8wa come later once dynamic export is proven.)

Usage:
    python scripts/convert_dynamic.py text      --variant fp32
    python scripts/convert_dynamic.py generator --variant fp16
    python scripts/convert_dynamic.py both      --variant fp32
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.export import Dim, export

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import TextStage, GeneratorStage, WeightBag  # noqa: E402

import coremltools as ct  # noqa: E402

SAFE = "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
OUT_DIR = Path("scripts/models/dynamic")

# Range of shapes we want the dynamic model to accept.
L_MIN, L_MAX = 16, 400
N_MIN, N_MAX = 32, 1024


def precision_of(variant: str) -> ct.precision:
    return {
        "fp32": ct.precision.FLOAT32,
        "fp16": ct.precision.FLOAT16,
    }[variant]


def sample_style() -> torch.Tensor:
    v = np.load("scripts/models/voices.npz")
    return torch.from_numpy(v["expr-voice-5-m"][64:65].astype(np.float32))


def convert_text_stage(variant: str) -> Path:
    print(f"=== TextStage  variant={variant} ===")
    w = WeightBag.load(SAFE)
    stage = TextStage(w).eval()

    # Use a middle-of-range trace shape so torch.export picks a reasonable
    # concrete shape for forward execution; the Dim object tells torch
    # which axis is symbolic.
    trace_L = 64
    input_ids = torch.randint(1, 170, (1, trace_L), dtype=torch.long)
    style = sample_style()
    mask = torch.ones(1, trace_L, dtype=torch.float32)

    # Sanity-run the forward pass to catch obvious errors before export.
    with torch.no_grad():
        p, t, d = stage(input_ids, style, mask)
    print(f"  forward OK  prosody={tuple(p.shape)}  text={tuple(t.shape)}  dur_sig={tuple(d.shape)}")

    L_dim = Dim("L", min=L_MIN, max=L_MAX)
    dynamic_shapes = {
        "input_ids":      {1: L_dim},
        "style256":       None,             # always (1, 256)
        "attention_mask": {1: L_dim},
    }

    print("  torch.export ...")
    with torch.no_grad():
        try:
            ep = export(stage, (input_ids, style, mask),
                        dynamic_shapes=dynamic_shapes)
        except Exception as e:
            print(f"  ❌ torch.export failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

    print(f"  export OK  ({len(list(ep.graph_module.graph.nodes))} fx nodes)")

    print("  ct.convert ...")
    mlmodel = ct.convert(
        ep,
        inputs=[
            ct.TensorType(name="input_ids",
                          shape=ct.Shape(shape=(1, ct.RangeDim(L_MIN, L_MAX, default=trace_L))),
                          dtype=np.int32),
            ct.TensorType(name="style", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="attention_mask",
                          shape=ct.Shape(shape=(1, ct.RangeDim(L_MIN, L_MAX, default=trace_L))),
                          dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=precision_of(variant),
    )
    mlmodel.short_description = f"KittenTTS text stage  dynamic L∈[{L_MIN},{L_MAX}]  {variant}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"kitten_text_dyn_{variant}.mlpackage"
    mlmodel.save(str(out))
    print(f"  saved {out}")
    return out


def convert_generator_stage(variant: str) -> Path:
    print(f"=== GeneratorStage  variant={variant} ===")
    w = WeightBag.load(SAFE)
    stage = GeneratorStage(w).eval()

    trace_N = 128
    prosody_lr = torch.randn(1, 256, trace_N)
    text_lr = torch.randn(1, 128, trace_N)
    style = sample_style()

    with torch.no_grad():
        wav = stage(prosody_lr, text_lr, style)
    print(f"  forward OK  wav={tuple(wav.shape)}")

    N_dim = Dim("N", min=N_MIN, max=N_MAX)
    dynamic_shapes = {
        "prosody_lr_ncl": {2: N_dim},
        "text_lr_ncl":    {2: N_dim},
        "style256":       None,
    }

    print("  torch.export ...")
    with torch.no_grad():
        try:
            ep = export(stage, (prosody_lr, text_lr, style),
                        dynamic_shapes=dynamic_shapes)
        except Exception as e:
            print(f"  ❌ torch.export failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

    print(f"  export OK  ({len(list(ep.graph_module.graph.nodes))} fx nodes)")

    print("  ct.convert ...")
    mlmodel = ct.convert(
        ep,
        inputs=[
            ct.TensorType(name="prosody_lr",
                          shape=ct.Shape(shape=(1, 256, ct.RangeDim(N_MIN, N_MAX, default=trace_N))),
                          dtype=np.float32),
            ct.TensorType(name="text_lr",
                          shape=ct.Shape(shape=(1, 128, ct.RangeDim(N_MIN, N_MAX, default=trace_N))),
                          dtype=np.float32),
            ct.TensorType(name="style", shape=(1, 256), dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=precision_of(variant),
    )
    mlmodel.short_description = f"KittenTTS generator stage  dynamic N∈[{N_MIN},{N_MAX}]  {variant}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"kitten_generator_dyn_{variant}.mlpackage"
    mlmodel.save(str(out))
    print(f"  saved {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["text", "generator", "both"],
                    default="both", nargs="?")
    ap.add_argument("--variant", default="fp32",
                    choices=["fp32", "fp16"],
                    help="compute precision (int8 variants are a separate pass)")
    args = ap.parse_args()

    if args.which in ("text", "both"):
        convert_text_stage(args.variant)
    if args.which in ("generator", "both"):
        convert_generator_stage(args.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())

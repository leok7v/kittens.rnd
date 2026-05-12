# cpu/ backend — next steps (morning of 2026-05-11)

## What's verified working as of the last commit

**End-to-end Xcode build passes:**
```sh
xcodebuild -project KittensRnD.xcodeproj -scheme KittensRnD \
    -configuration Debug -destination 'platform=macOS' build
# => ** BUILD SUCCEEDED **
```

**End-to-end app run passes:**
```sh
./Build/Products/Debug/KittensRnD.app/Contents/MacOS/KittensRnD \
    --bench --text "Hello world." --runs 1 --voice Kiki \
    --out tmp/bench --configs "ggml,cpu"
# => both backends run, both write valid 24 kHz mono WAVs
# => ggml WAV is real speech (energy 5608)
# => cpu  WAV is silence  (energy 0, as expected — stub)
```

**Standalone C tests pass:**
```sh
cd cpu && make           # build everything
make test_basic          # 15 hand-verified tests on tiny inputs
make test_ggml           # 22 ops vs ggml on random inputs, RMSE < 5e-3
make test_gguf           # reader vs kitten_full.gguf, 59 tensors found
make test_cpu_link       # smoke test: ctx open, weights bind, stub synth
```

All four targets pass. The C side compiles with one `#ifdef` block
that picks Accelerate on Apple, `<cblas.h>` everywhere else.

## What does NOT work yet

`kt_cpu_synthesize` returns silence (see TODO at the top of
`cpu/kittens-tts-cpu.c`). The four stages still need to be ported from
the existing `ggml/kittens-tts.c`.

## How the Xcode wiring works (already done)

`KittensRnD.xcodeproj/project.pbxproj` was edited tonight to:

1. Add `cpu` to `PBXFileSystemSynchronizedRootGroup` — same mechanism
   that brings `ggml/` and `mlx/` into the target automatically.
2. Add `cpu` to the target's `fileSystemSynchronizedGroups`.
3. Add `cpu` and `cpu/include` to `HEADER_SEARCH_PATHS` (both Debug
   and Release configs).
4. Add a `PBXFileSystemSynchronizedBuildFileExceptionSet` for cpu/
   that excludes `tests/*.c`, `Makefile`, `build/`, and
   `NEXT_STEPS.md` from the target — those have their own `main()`s
   and would conflict with the app's entry point.

The Accelerate deprecation noise (`cblas_sgemm` etc deprecated in
macOS 13.3+) is silenced by `#define ACCELERATE_NEW_LAPACK` inside
`cpu/kt_tensor.c` before the framework include. No project-wide flag
needed.

## Picking up Phase 3 — porting the four stages

Mechanical translation map from ggml to kt:

| ggml | kt |
|---|---|
| `ggml_get_rows(ctx, w, ids)` | `kt_get_rows(w, ids_arr, n_ids)` |
| `ggml_add(ctx, x, y)` | `kt_add(x, y)` |
| `ggml_mul(ctx, x, y)` | `kt_mul(x, y)` |
| `ggml_mul_mat(ctx, w, x)` | `kt_mul_mat(w, x)` |
| `ggml_scale(ctx, x, s)` | `kt_scale(x, s)` |
| `ggml_norm(ctx, x, eps)` | `kt_norm(x, 0, eps)` |
| `ggml_soft_max_ext(ctx, x, NULL, s, 0)` | `kt_softmax(x, 0, s)` |
| `ggml_sigmoid(ctx, x)` etc | `kt_sigmoid(x)` etc |
| `ggml_concat(ctx, a, b, axis)` | `kt_concat(a, b, axis)` |
| `ggml_reshape_2d(ctx, t, n0, n1)` | `kt_reshape_2d(t, n0, n1)` |
| `ggml_transpose(ctx, t)` | `kt_transpose(t)` |
| `ggml_cont(ctx, t)` | `kt_cont(t)` |
| `ggml_view_1d(ctx, t, n, off)` | `kt_view_1d(t, n, off)` |
| `ggml_view_2d(ctx, t, n0, n1, nb1, off)` | `kt_view_2d(t, n0, n1, nb1, off)` |
| `ggml_permute(ctx, t, p0, p1, p2, p3)` | `kt_permute(t, p0, p1, p2, p3)` |
| `ggml_conv_1d(ctx, w, x, s, p, d)` | `kt_conv_1d(w, x, s, p, d)` |
| `ggml_conv_1d_dw(ctx, w, x, s, p, d)` | `kt_conv_1d_dw(w, x, s, p, d)` |
| `ggml_conv_transpose_1d(ctx, w, x, s, 0, 0)` | `kt_conv_transpose_1d(w, x, s, 0)` |
| `ggml_repeat(ctx, t, target)` | `kt_repeat(t, target)` |
| `ggml_cumsum(ctx, t)` | `kt_cumsum(t, 0)` |
| `ggml_sin(ctx, t)` etc | `kt_sin(t)` etc |
| `ggml_map_custom2(... kt_atan2_op ...)` | `kt_atan2(y, x)` |
| `ggml_build_forward_expand(gf, t)` | — (eager, just drop) |
| `ggml_set_input(t)` / `set_output(t)` | — (drop) |
| `ggml_backend_tensor_set/get` | direct `memcpy` to/from `t->data` |
| `ggml_new_tensor_*d(ctx, F32, ...)` | `kt_new_*d(arena, ...)` |

### Recommended port order

1. **Build helpers** (`kt_layer_norm`, `kt_ada_layer_norm`,
   `kt_ada_in_1d`, `kt_snake_1d`, `kt_conv1d_nlc`, `kt_lstm_dir`,
   `kt_bidir_lstm`). All are pure functions of input tensors → output
   tensors. ~200 LOC total.

2. **`build_albert`** — BERT/Albert encoder. Self-contained: inputs
   are `(L,)` ids, output is `(hidden=768, L)`. One caveat: multi-head
   attention uses 4D tensors for the per-head Q/K/V matmul. My
   `kt_mul_mat` currently asserts `ne[2]==1 && ne[3]==1`. **Workaround:**
   loop over heads, doing one (head_dim, L) × (head_dim, L)^T matmul
   per head, ~12 iterations per layer × 12 layers. Or extend
   `kt_mul_mat` to handle a batch dim along ne[2].

3. **`build_pred_text`** — predictor text encoder. Uses concat +
   LSTM + AdaLayerNorm only. Should be straightforward once the
   helpers exist.

4. **`build_acoustic`** — acoustic text encoder. Two conv1d +
   layer-norm + LSTM. All ops verified vs ggml.

5. **`build_textstage`** — orchestrator that wires the above three
   plus the duration head and dur_proj matmul. Returns the three
   outputs (prosody_ncl, text_ncl, dur_sig) that `kt_cpu_synthesize`
   feeds forward.

6. **`build_genfront`** — shared LSTM + 6 AdaINResBlock1Ds + F0_proj /
   N_proj convs.

7. **`build_decoder`** — decoder.encode + 4 decoder.decode blocks +
   asr_res / F0_conv / N_conv reductions.

8. **`build_generator`** — HiFi-GAN-style upsamplers + ResBlocks +
   conv_post + iSTFT head. The noise path (SineGen + STFT) is the
   most intricate piece; tested kt_atan2 / kt_sin / kt_cos already
   match ggml bit-for-bit, so the math should hold once the wiring
   is right.

9. **Wire into `kt_cpu_synthesize`** — replace the stub with calls to
   the four stage functions, doing length regulation in between
   exactly as `ggml/kittens-tts.c::kt_synthesize` does.

### Verifying each stage

Use the existing `scripts/tts_e2e.py` harness. It already calls the
ggml backend with four CLI modes (`textstage` / `genfront` /
`decoder` / `fullgen`), reading inputs from `.bin` files and writing
outputs to `.bin` files. Add a `--backend cpu` flag (or build a
mirror `kittens-tts-cpu` CLI binary with the same interface) and the
existing parity comparison drops in unchanged.

A worked example for stage 1:

```sh
# 1. Run the ggml side to get ground truth.
./tmp/kittens-tts-cpu --mode textstage --gguf tmp/kitten_full.gguf \
    --input tmp/e2e_ts_in.bin --output tmp/baseline_ts.bin

# 2. Run kt_cpu side on the same input.
./tmp/kittens-tts-cpu --mode textstage --gguf tmp/kitten_full.gguf \
    --input tmp/e2e_ts_in.bin --output tmp/kt_ts.bin

# 3. Diff numerically (RMSE per output stream).
python3 - <<'EOF'
import numpy as np
a = np.fromfile("tmp/baseline_ts.bin", dtype=np.float32)
b = np.fromfile("tmp/kt_ts.bin",       dtype=np.float32)
print(f"RMSE = {np.sqrt(((a-b)**2).mean()):.3e}")
EOF
```

### SESE notes for the port

The original `kittens-tts.c` uses single-exit functions throughout
but mixes ggml graph-building patterns inline. In the kt port,
because every op call already computes immediately, each builder
naturally reads as a sequence of named tensor assignments — exactly
the style `coding-philosophy.md` advocates. The temptation will be
to bury 20-line stretches inside a single function; resist and
extract small helpers (each returning one tensor) so the post-
condition is visible at every closing brace.

### proofdiff usage

For each ported stage, the proof obligation is: "for any valid
input, the kt port's tensor-at-line-L1 has the same bytes (within
RMSE 1e-5) as the ggml original's tensor-at-line-L2". The skill's
state-tree / pre-post matching applies at the tensor-name granularity
— track each named tensor in both versions, verify per-tensor RMSE
at each function-exit boundary. The op-level parity tests already
done in `tests/test_kt_vs_ggml.c` are the per-edge correctness gate;
the stage-level proof is the per-node correctness gate.

## Open questions

1. **`kt_mul_mat` for 4D inputs** — needed for multi-head attention
   in BERT. Decide: extend the impl to handle a batch dim along
   ne[2] (the cleaner answer), or loop over heads in build_albert
   (the cheaper-to-write answer). The extension is ~30 lines of
   `cblas_sgemm_batch` or a per-batch loop in C.

2. **Threading** — kt_tensor is single-threaded today (per Leo's
   v1 decision). Once parity is verified, the natural next step is
   `dispatch_apply` on Apple and OpenMP elsewhere for the inner-loop
   conv / sgemm. ~1 day of work, optional.

3. **F16 weights** — currently dequantized to F32 at load time. ~40
   MB of F32 weights resident vs ~20 MB of F16. If footprint matters,
   keep some weights as F16 and do F16 matmul via `cblas_sgemm` with
   `vDSP_vfp32` conversion at call sites. Probably premature.

## Filesystem layout (after this push)

```
cpu/
   include/
      KittensCPU.h          public C API, mirrors KittensGGML.h
      kt_gguf.h             minimal GGUF v3 reader
      kt_tensor.h           ~30 ops, fp32 NCL, eager evaluation
   kt_tensor.c             ~900 LOC; cblas + scalar fallbacks
   gguf_read.c             ~400 LOC; mmap + parse + lookup
   kittens-tts-cpu.c       ~330 LOC; loads weights, STUB synth
   CPUBackend.swift        KittenTTSCpu class for the app
   Makefile                build everything outside Xcode
   tests/
      test_kt_basic.c       hand-verified op tests
      test_kt_vs_ggml.c     22-op parity against ggml on random
      test_gguf.c           reader vs ggml's gguf_init_from_file
      test_cpu_link.c       end-to-end C surface smoke test
   NEXT_STEPS.md           this file
```

App-side touches:
- `app/Bridging-Header.h` — added `#include "KittensCPU.h"`
- `app/KittensRnDApp.swift` — Backend.cpu case + switch arms
- `app/Bench.swift` — new "cpu" bench row + switch arm

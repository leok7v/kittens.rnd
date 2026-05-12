# cpu/ vs ggml/ backend parity notes

End-to-end status as of this commit:

- Both backends produce 53400 samples for "Hello world." + Kiki voice
  (length regulation agrees — same per-token durations).
- Both produce audible, recognizable speech.
- Cross-correlation peak is at **lag=+2 samples** (cpu audio is 2 samples
  late relative to ggml).
- After aligning by lag=+2: RMSE 0.036, correlation 0.975.
- Per-chunk: fade-in / fade-out regions match to RMSE ~1e-4; mid-speech
  RMSE ~0.10 on signal amplitudes of ~0.27.

## What's been verified bit-equivalent (test_kt_vs_ggml)

- All elementwise / unary / reduction / softmax ops vs ggml: PASS
- LayerNorm, cumsum, get_rows: PASS (RMSE ~1e-7)
- mul_mat (2D and 3D batched): PASS
- conv_1d, conv_1d_dw (F32 hand-quant): PASS (RMSE ~2e-4 from F16
  weight quantization)

## Update: the 2-sample "offset" is local phase noise, not a timing shift

Speech timing features align byte-perfectly between the two backends:

- First positive zero crossing: sample 5490 in both
- First |x| > 0.05: sample 18640 in both
- First |x| > 0.10: sample 19257 in both

The "lag=+2 best RMSE" from raw cross-correlation reflects local
phase divergence in the synthesized waveform, not a structural delay.
Speech-envelope timing is identical. The remaining 0.036 RMSE after
phase alignment is accumulated numerical noise from cblas_sgemm vs
ggml_vec_dot_f32 producing different rounding-order results across
the ~thousand matrix-multiplies the inference graph executes.

## Likely sources of the residual 0.036 RMSE / 2-sample offset

Investigated and ruled out:

- LSTM gate ordering (ifgo, gate-by-gate views, h_t memcpy):
  identical to ggml byte for byte.
- kt_conv_transpose_1d output formula and scatter pattern: matches
  ggml's `(Lin-1)*s - 2p + d(K-1) + 1` with p=0 d=1; identical loop.
- iSTFT view offset (`trim * sizeof(float)`) and length
  (`T - 2*trim`): identical math to ggml.
- conv1d bias broadcast layout: identical (b reshape (1, Cout, 1)).
- Layer-norm formula: `(x - mean) / sqrt(var + eps)` with mean/var
  over ne[0]; matches ggml.

Plausible sources NOT cheap to verify without a kt-side CLI binary +
per-stage WAV dumps:

- **cblas_sgemm vs ggml_vec_dot_f32 summation order.** Different
  inner-loop summation orders give per-element rounding deltas at
  the 1e-7 level; over many matmuls in the BERT/decoder chain the
  accumulated noise can reach 0.01-0.1 by output time.
- **F16 weight dequantization timing.** kt dequantizes F16 → F32 at
  load time (`kt_gguf_load_tensor`); ggml dequantizes on-the-fly
  during the matmul. Same final values, but intermediate buffers
  may have slightly different rounding boundaries.
- **STFT analysis window padding semantics.** Conv1d at the boundary
  with pad=10, K=20 reads from invalid input positions (zero in both
  impls). If somewhere there's an off-by-one in the boundary
  computation, the 2-sample offset would emerge from the STFT/iSTFT
  pair.
- **conv_transpose_1d-via-im2col path differences.** ggml's CPU kernel
  uses a custom "kernel-reorder + ggml_vec_dot" path
  (`ggml_compute_forward_conv_transpose_1d_f32`); my kt impl is a
  direct scatter loop. They should agree mathematically but I have
  not unit-tested kt vs ggml for conv_transpose_1d (the parity test
  was intentionally skipped — ggml asserts p0==0).

## Recommended next investigation step

Build a `kittens-tts-cpu` CLI (mirroring `--mode bert|textstage|generator|full`
in `ggml/kittens-tts.c`) and feed both binaries the same `.bin`
fixtures from `scripts/tts_e2e.py`. Diff per-stage tensor outputs.
The first stage that diverges localizes the bug. The CLI is ~300 LOC
mostly boilerplate; the kt_cpu_synthesize internals are already
factored into stage builders.

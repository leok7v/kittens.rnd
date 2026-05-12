// KittensCPU.h -- public C API for the pure-C / cblas KittenTTS backend.
//
// Mirrors KittensGGML.h 1:1 so the Swift CPUBackend.swift is a near-copy
// of GGMLBackend.swift with the function names rewritten.
//
// Lifecycle:
//   kt_cpu_ctx * ctx = kt_cpu_create("path/to/kitten_full.gguf");
//   kt_cpu_audio a   = kt_cpu_synthesize(ctx, ids, n_ids, style, 1.0f);
//   // ... do something with a.samples (f32 PCM, 24 kHz mono) ...
//   kt_cpu_audio_free(a);
//   kt_cpu_destroy(ctx);
//
// All inputs are caller-owned; outputs returned by kt_cpu_synthesize
// must be released with kt_cpu_audio_free. Thread-unsafe — use one ctx
// per thread, or add your own locking.
//
// Backend characteristic: pure C99 + a portable cblas. On Apple this is
// Accelerate; on Linux/Windows OpenBLAS or BLIS. No ggml linkage.

#pragma once
#ifndef KITTENS_CPU_H
#define KITTENS_CPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kt_cpu_ctx kt_cpu_ctx;

// Create a context that owns the loaded GGUF model and inference state.
// Returns NULL on failure; kt_cpu_last_error() returns the message.
kt_cpu_ctx * kt_cpu_create(const char * gguf_path);

// Destroy a context and free all associated resources.
void kt_cpu_destroy(kt_cpu_ctx * ctx);

// Audio output buffer. `samples` is owned by the library; release via
// kt_cpu_audio_free.
typedef struct {
    float *   samples;
    uint64_t  n_samples;
} kt_cpu_audio;

// Synthesize a single chunk:
//   phonemes:    n_phonemes int32 phoneme IDs (caller-owned)
//   n_phonemes:  L
//   style256:    256 floats — concatenated [acoustic[0:128], prosodic[128:256]]
//   speed:       1.0 = normal speech rate
//
// Output: 24 kHz f32 PCM mono. Returned `samples` is NULL only on error.
kt_cpu_audio kt_cpu_synthesize(kt_cpu_ctx * ctx,
                               const int32_t * phonemes,
                               int             n_phonemes,
                               const float *   style256,
                               float           speed);

// Free a kt_cpu_audio returned by kt_cpu_synthesize.
void kt_cpu_audio_free(kt_cpu_audio a);

// Last-error message for a context (or for global init failures if
// ctx is NULL). Pointer is valid until the next kt_cpu_* call on the
// same context.
const char * kt_cpu_last_error(const kt_cpu_ctx * ctx);

#ifdef __cplusplus
}
#endif

#endif // KITTENS_CPU_H

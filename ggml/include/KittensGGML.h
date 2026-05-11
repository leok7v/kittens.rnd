// kittens-tts.h — public C API for the ggml-backed KittenTTS backend.
//
// Lifecycle:
//     kt_ctx * ctx = kt_create("path/to/kitten_full.gguf", "cpu");
//     kt_audio a = kt_synthesize(ctx, ids, n_ids, style, 1.0f);
//     // ... do something with a.samples (f32 PCM, 24 kHz mono) ...
//     kt_audio_free(a);
//     kt_destroy(ctx);
//
// All inputs are caller-owned; outputs returned by kt_synthesize must be
// released with kt_audio_free. Thread-unsafe — use one ctx per thread, or
// add your own locking.

#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kt_ctx kt_ctx;

// Create a context that owns the loaded GGUF model and a backend.
// `backend` is "cpu" or "metal" (metal not yet supported in v1; pass "cpu").
// Returns NULL on failure.
kt_ctx * kt_create(const char * gguf_path, const char * backend);

// Destroy a context and free all associated resources.
void kt_destroy(kt_ctx * ctx);

// Audio output buffer.  `samples` is owned by the library; release via
// kt_audio_free.
typedef struct {
    float *   samples;
    uint64_t  n_samples;
} kt_audio;

// Synthesize a single chunk:
//   phonemes:    L int32 phoneme IDs (caller-owned)
//   n_phonemes:  L
//   style256:    256 floats — concatenated [acoustic[0:128], prosodic[128:256]]
//   speed:       1.0 = normal speech rate
//
// Output: 24 kHz f32 PCM mono. Returned `samples` is NULL only on error.
// The audio length depends on phoneme durations; you can expect roughly
// (n_phonemes * 600 * 3) samples on average for English at speed=1.0.
kt_audio kt_synthesize(kt_ctx * ctx,
                       const int32_t * phonemes,
                       int             n_phonemes,
                       const float *   style256,
                       float           speed);

// Free a kt_audio returned by kt_synthesize.
void kt_audio_free(kt_audio a);

// Last-error message for a context (or for global init failures if ctx is NULL).
// Pointer is valid until the next kt_* call on the same context.
const char * kt_last_error(const kt_ctx * ctx);

#ifdef __cplusplus
}
#endif

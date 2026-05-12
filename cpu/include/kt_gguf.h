// kt_gguf.h -- minimal GGUF v3 reader for kittens-tts.
//
// Opens kitten_full.gguf as mmap, parses header + KV table + tensor
// table once, then offers:
//   - keyed scalar lookup (u32 / f32 / bool / string) for the arch KVs
//   - tensor lookup by name, returning a kt_tensor that either wraps
//     the mmap'd data directly (F32 tensors) or holds a freshly-
//     allocated arena buffer with the F16 -> F32 dequant already done.
//
// The reader supports only what kittens-tts's GGUF writer
// (scripts/convert_to_gguf.py) emits: GGUF_TYPE_UINT32 /
// GGUF_TYPE_FLOAT32 / GGUF_TYPE_BOOL / GGUF_TYPE_STRING for KVs;
// GGML_TYPE_F32 / GGML_TYPE_F16 for tensors. Other types trip an
// assert at load time.

#pragma once
#ifndef KT_GGUF_H
#define KT_GGUF_H

#include "kt_tensor.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kt_gguf;
typedef struct kt_gguf kt_gguf;

// Open and parse a GGUF v3 file. NULL on failure; caller can read
// kt_gguf_last_error() to find out why.
kt_gguf * kt_gguf_open(const char * path);
void      kt_gguf_close(kt_gguf * g);
const char * kt_gguf_last_error(void);

// Scalar key lookups. Return 1 on found, 0 on missing.
int kt_gguf_get_u32  (const kt_gguf * g, const char * key, uint32_t * out);
int kt_gguf_get_f32  (const kt_gguf * g, const char * key, float    * out);
int kt_gguf_get_bool (const kt_gguf * g, const char * key, int      * out);

// Number of tensors in the file; mostly for iteration.
int kt_gguf_n_tensors(const kt_gguf * g);

// Name of the i'th tensor, owned by g.
const char * kt_gguf_tensor_name(const kt_gguf * g, int i);

// Look up a tensor by name and materialize it inside `arena` as a
// fully-realized fp32 kt_tensor. F32 tensors are wrapped without copy
// (the underlying bytes stay in the mmap region; the kt_tensor itself
// is allocated from `arena`). F16 tensors are dequantized into a fresh
// `arena`-owned buffer.
//
// Returns NULL if the tensor isn't present. Asserts on unsupported
// dtypes.
kt_tensor * kt_gguf_load_tensor(const kt_gguf * g, kt_arena * arena,
                                const char * name);

#ifdef __cplusplus
}
#endif

#endif // KT_GGUF_H

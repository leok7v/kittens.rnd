// kt_tensor.h -- minimal CPU-only fp32 tensor library for kittens-tts.
//
// Designed as a drop-in replacement for the subset of ggml that
// kittens-tts.c actually uses (73 distinct ggml_* symbols, of which
// ~30 are real ops; the rest is graph/backend plumbing that vanishes
// under eager evaluation).
//
// Conventions
// -----------
//   - Single dtype: float32. The kitten-tts model is fp32 end-to-end;
//     quantization is a separate concern for a v2 of this library.
//   - Shape stored in ne[4], ne[0] is the FASTEST-VARYING axis (matches
//     ggml; reverses PyTorch). For NCL activations: ne[0]=L, ne[1]=C,
//     ne[2]=N (=1), ne[3]=1.
//   - Strides stored in nb[4] in BYTES. nb[0] == sizeof(float) when
//     packed.
//   - Tensors are arena-allocated. The arena owns both the tensor
//     header structs and their data slabs. A weights arena lives for
//     the model's lifetime; a scratch arena is reset at the top of
//     every kt_cpu_synthesize call.
//   - All ops are EAGER: each call computes immediately and returns a
//     fresh tensor owned by the same arena as its first input. No
//     graph builder, no two-phase compute.
//   - Broadcasting is restricted to the three cases kittens-tts uses:
//     scalar (1-element tensor) | channel-vector (length C along
//     ne[1]) | matching shape. Anything else trips an assert.
//
// Build
// -----
//   On Apple: link with -framework Accelerate, define
//     ACCELERATE_NEW_LAPACK to silence the macOS 13.3+ deprecation
//     warning on the legacy ILP32 cblas symbols.
//   On Linux/Windows: link with -lopenblas (or any cblas-providing
//     BLAS) and -lm.

#pragma once
#ifndef KT_TENSOR_H
#define KT_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KT_MAX_DIMS 4

struct kt_arena;
typedef struct kt_arena kt_arena;

typedef struct kt_tensor {
    int64_t          ne[KT_MAX_DIMS];   // logical extents, ne[0] inner
    int64_t          nb[KT_MAX_DIMS];   // byte strides
    int              ndim;              // 1..4
    float *          data;              // 64-byte aligned, arena-owned
    struct kt_arena * arena;            // for output allocation
    char             name[32];          // debug only; "" if unnamed
} kt_tensor;


// ---------------------------------------------------------------------------
// Arena lifecycle
// ---------------------------------------------------------------------------

// Create an arena pre-reserved for `initial_bytes` of payload + headers.
// Grows on demand by doubling. NULL on allocation failure.
kt_arena * kt_arena_new(size_t initial_bytes);

// Release the arena and every tensor in it.
void kt_arena_free(kt_arena * a);

// Wipe contents (set cursor to 0); retain reserved capacity. Use
// between synth calls to drop scratch tensors without re-malloc'ing.
void kt_arena_reset(kt_arena * a);

// Current usage in bytes (header + data).
size_t kt_arena_used(const kt_arena * a);

// Total reserved capacity in bytes (after any growth).
size_t kt_arena_capacity(const kt_arena * a);

// Set the "active arena" used as the output target for allocating ops
// (kt_mul_mat / kt_add / kt_conv_1d / ...). When set, those ops put
// their output here instead of in the first input's arena. Pass NULL
// to revert to the first-input behavior.
//
// Without this, an op like kt_mul_mat(W, x) where W lives in a long-
// lived weights arena would allocate the output IN weights_arena —
// activations would accumulate forever. Set this to your per-inference
// scratch arena before each compute so outputs land where you can
// reset them. Low-level constructors (kt_new_*d, kt_wrap_*d) and
// kt_gguf_load_tensor are NOT affected; they keep using their explicit
// arena argument so weight loading still goes to the weights arena.
//
// Process-global single-threaded slot — caller must serialize.
void kt_arena_set_active(kt_arena * a);
kt_arena * kt_arena_get_active(void);


// ---------------------------------------------------------------------------
// Tensor creation
// ---------------------------------------------------------------------------

kt_tensor * kt_new_1d(kt_arena * a, int64_t n0);
kt_tensor * kt_new_2d(kt_arena * a, int64_t n0, int64_t n1);
kt_tensor * kt_new_3d(kt_arena * a, int64_t n0, int64_t n1, int64_t n2);
kt_tensor * kt_new_4d(kt_arena * a, int64_t n0, int64_t n1, int64_t n2,
                      int64_t n3);

// Wrap a caller-owned packed fp32 buffer with no copy. The buffer must
// remain valid for the lifetime of `a`. Used for model weights loaded
// via mmap.
kt_tensor * kt_wrap_2d(kt_arena * a, float * data,
                       int64_t n0, int64_t n1);
kt_tensor * kt_wrap_3d(kt_arena * a, float * data,
                       int64_t n0, int64_t n1, int64_t n2);

// Set the debug name (cap 31 chars + NUL).
void kt_set_name(kt_tensor * t, const char * name);


// ---------------------------------------------------------------------------
// Layout ops (cheap: no data copy unless _cont)
// ---------------------------------------------------------------------------

kt_tensor * kt_view_1d(kt_tensor * src,
                       int64_t n0, size_t off_bytes);
kt_tensor * kt_view_2d(kt_tensor * src,
                       int64_t n0, int64_t n1,
                       size_t nb1, size_t off_bytes);
kt_tensor * kt_view_3d(kt_tensor * src,
                       int64_t n0, int64_t n1, int64_t n2,
                       size_t nb1, size_t nb2,
                       size_t off_bytes);
kt_tensor * kt_reshape_2d(kt_tensor * src, int64_t n0, int64_t n1);
kt_tensor * kt_reshape_3d(kt_tensor * src,
                          int64_t n0, int64_t n1, int64_t n2);
kt_tensor * kt_reshape_4d(kt_tensor * src,
                          int64_t n0, int64_t n1,
                          int64_t n2, int64_t n3);
kt_tensor * kt_permute(kt_tensor * src,
                       int p0, int p1, int p2, int p3);
kt_tensor * kt_transpose(kt_tensor * src);          // swap ne[0]<->ne[1]
kt_tensor * kt_cont(kt_tensor * src);               // realize as packed
kt_tensor * kt_cont_2d(kt_tensor * src, int64_t n0, int64_t n1);

kt_tensor * kt_concat(kt_tensor * a, kt_tensor * b, int axis);

// Tile `src` to match the shape of `shape_like`. Each axis of `src`
// must either equal the corresponding axis of `shape_like` or be 1.
kt_tensor * kt_repeat(kt_tensor * src, const kt_tensor * shape_like);

// Like kt_repeat but takes the target shape as ints. Avoids allocating
// a dummy template tensor whose data is never read — useful when the
// caller only knows the target shape, not a full tensor of that shape.
kt_tensor * kt_repeat_to(kt_tensor * src, int ndim,
                         int64_t n0, int64_t n1,
                         int64_t n2, int64_t n3);

// dst := src (bytes). Shapes must match; strides may differ. The two
// tensors must come from the same arena (so they share their lifetime).
void kt_cpy(const kt_tensor * src, kt_tensor * dst);

// out[i, :, ...] = data[ids[i], :, ...]. `data` is 2D (vocab, embed),
// ids is a host array of length n_ids. Returns (n_ids, embed) packed.
kt_tensor * kt_get_rows(kt_tensor * data,
                        const int32_t * ids, int n_ids);


// ---------------------------------------------------------------------------
// Elementwise binary (broadcast: scalar | C-vector along ne[1] | full)
// ---------------------------------------------------------------------------

kt_tensor * kt_add(kt_tensor * x, kt_tensor * y);
kt_tensor * kt_sub(kt_tensor * x, kt_tensor * y);
kt_tensor * kt_mul(kt_tensor * x, kt_tensor * y);
kt_tensor * kt_div(kt_tensor * x, kt_tensor * y);


// ---------------------------------------------------------------------------
// Elementwise unary
// ---------------------------------------------------------------------------

kt_tensor * kt_scale(kt_tensor * x, float s);
kt_tensor * kt_sigmoid(kt_tensor * x);
kt_tensor * kt_tanh(kt_tensor * x);
kt_tensor * kt_leaky_relu(kt_tensor * x, float slope);
kt_tensor * kt_gelu_erf(kt_tensor * x);
kt_tensor * kt_step(kt_tensor * x);                 // x > 0 ? 1 : 0
kt_tensor * kt_sin(kt_tensor * x);
kt_tensor * kt_cos(kt_tensor * x);
kt_tensor * kt_exp(kt_tensor * x);
kt_tensor * kt_sqrt(kt_tensor * x);

// out = atan2(y, x). Both inputs must have identical shape.
kt_tensor * kt_atan2(kt_tensor * y, kt_tensor * x);


// ---------------------------------------------------------------------------
// Reductions / normalization
// ---------------------------------------------------------------------------

// LayerNorm-style: normalize along `axis`, subtract mean, divide by
// sqrt(var + eps). No learned scale/bias here; the caller multiplies
// and adds afterward.
kt_tensor * kt_norm(kt_tensor * x, int axis, float eps);

// soft_max_ext semantics: out[i] = exp(scale * x[i]) / sum_j exp(...).
// Stable: subtracts row max before exp.
kt_tensor * kt_softmax(kt_tensor * x, int axis, float scale);

// Cumulative sum along `axis`.
kt_tensor * kt_cumsum(kt_tensor * x, int axis);


// ---------------------------------------------------------------------------
// Linear algebra (BLAS sgemm under the hood)
// ---------------------------------------------------------------------------

// out = W @ x, treating W as (M, K) and x as (K, N), result (M, N).
// W is taken from its first two leading dims (ne[0] = K innermost,
// ne[1] = M); x same convention. This matches the ggml_mul_mat layout
// that kittens-tts.c uses.
kt_tensor * kt_mul_mat(kt_tensor * w, kt_tensor * x);


// ---------------------------------------------------------------------------
// 1D convolution (NCL inputs; weight layout (Cout, Cin, K))
// ---------------------------------------------------------------------------

kt_tensor * kt_conv_1d(kt_tensor * w, kt_tensor * x,
                       int stride, int pad, int dilation);

// Depthwise: weight shape (C, 1, K), groups = C.
kt_tensor * kt_conv_1d_dw(kt_tensor * w, kt_tensor * x,
                          int stride, int pad, int dilation);

// Weight layout (Cin, Cout, K), groups = 1.
kt_tensor * kt_conv_transpose_1d(kt_tensor * w, kt_tensor * x,
                                 int stride, int pad);

// Exposed for callers that want to fuse im2col + sgemm by hand. Input
// (B, Cin, L) -> output (B, Cin*K, Lout) where Lout follows the
// standard conv1d formula.
kt_tensor * kt_im2col(kt_tensor * x,
                      int kernel, int stride, int pad, int dilation);


// ---------------------------------------------------------------------------
// Shape inspection
// ---------------------------------------------------------------------------

int64_t kt_nelements(const kt_tensor * t);
size_t  kt_nbytes(const kt_tensor * t);
bool    kt_is_packed(const kt_tensor * t);
bool    kt_same_shape(const kt_tensor * a, const kt_tensor * b);


#ifdef __cplusplus
}
#endif

#endif // KT_TENSOR_H

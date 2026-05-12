// kt_tensor.c -- implementation. See kt_tensor.h for the API contract.

// Must be defined BEFORE any include that could transitively pull in
// Accelerate's cblas headers — otherwise the legacy LP64 cblas_sgemm
// declarations (deprecated in macOS 13.3) are picked up and the
// compiler issues deprecation warnings on every BLAS call site.
#if defined(__APPLE__) && !defined(ACCELERATE_NEW_LAPACK)
    #define ACCELERATE_NEW_LAPACK
#endif

#include "kt_tensor.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif


// ---------------------------------------------------------------------------
// Platform: aligned alloc (slab pages)
// ---------------------------------------------------------------------------
//
// Slabs are allocated via mmap directly (not posix_memalign / free) so
// that munmap returns the pages to the kernel IMMEDIATELY. macOS's
// libsystem_malloc caches medium/large allocations on a private free
// list, which means `posix_memalign(N) ... free()` looks like a leak
// to the OS RSS counter even after we've correctly freed it. A single
// long-sentence stage-4 peak (hundreds of MB of scratch slabs) would
// show up as multi-GB resident memory until process exit.
//
// mmap is page-aligned (16 KB on Apple Silicon); we round the request
// up to a page boundary and store the rounded size alongside the slab
// so munmap can pass the same length back. Linux/macOS both support
// MAP_ANON for anonymous private pages.

#include <sys/mman.h>
#include <unistd.h>

#define KT_ALIGN 64

static size_t kt_page_size(void) {
    static size_t cached = 0;
    if (cached == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        cached = ps > 0 ? (size_t)ps : 4096;
    }
    return cached;
}

static void * kt_aligned_alloc(size_t bytes, size_t * out_mapped) {
    const size_t pg = kt_page_size();
    size_t rounded = (bytes + pg - 1) & ~(pg - 1);
    void * result = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, -1, 0);
    if (result == MAP_FAILED) { result = NULL; rounded = 0; }
    *out_mapped = rounded;
    return result;
}

static void kt_aligned_free(void * p, size_t mapped) {
    if (p != NULL && mapped > 0) {
        munmap(p, mapped);
    }
}


// ---------------------------------------------------------------------------
// Arena: chained slabs, bump cursor inside each slab.
// ---------------------------------------------------------------------------

typedef struct kt_slab {
    char *           data;       // page-aligned (mmap'd)
    size_t           capacity;   // bytes the user asked for
    size_t           mapped;     // bytes actually mmap'd (>= capacity)
    size_t           used;
    struct kt_slab * next;
} kt_slab;

struct kt_arena {
    kt_slab * head;              // active slab, has free space
    kt_slab * first;             // never freed by reset
    size_t    initial_bytes;
};

static kt_slab * kt_slab_new(size_t bytes) {
    kt_slab * s = (kt_slab *)calloc(1, sizeof(kt_slab));
    assert(s != NULL);
    size_t mapped = 0;
    s->data = (char *)kt_aligned_alloc(bytes, &mapped);
    assert(s->data != NULL);
    s->capacity = bytes;
    s->mapped = mapped;
    s->used = 0;
    s->next = NULL;
    return s;
}

static void kt_slab_free(kt_slab * s) {
    if (s != NULL) {
        kt_aligned_free(s->data, s->mapped);
        free(s);
    }
}

kt_arena * kt_arena_new(size_t initial_bytes) {
    kt_arena * a = (kt_arena *)calloc(1, sizeof(kt_arena));
    assert(a != NULL);
    size_t sz = initial_bytes < 4096 ? 4096 : initial_bytes;
    a->initial_bytes = sz;
    a->first = kt_slab_new(sz);
    a->head = a->first;
    return a;
}

void kt_arena_free(kt_arena * a) {
    if (a != NULL) {
        kt_slab * s = a->first;
        while (s != NULL) {
            kt_slab * next = s->next;
            kt_slab_free(s);
            s = next;
        }
        free(a);
    }
}

void kt_arena_reset(kt_arena * a) {
    assert(a != NULL);
    // Drop every slab past the first; reset cursor of the first.
    kt_slab * s = a->first->next;
    while (s != NULL) {
        kt_slab * next = s->next;
        kt_slab_free(s);
        s = next;
    }
    a->first->next = NULL;
    a->first->used = 0;
    a->head = a->first;
}

size_t kt_arena_used(const kt_arena * a) {
    assert(a != NULL);
    size_t total = 0;
    const kt_slab * s = a->first;
    while (s != NULL) {
        total += s->used;
        s = s->next;
    }
    return total;
}

size_t kt_arena_capacity(const kt_arena * a) {
    assert(a != NULL);
    size_t total = 0;
    const kt_slab * s = a->first;
    while (s != NULL) {
        total += s->capacity;
        s = s->next;
    }
    return total;
}

// "Active arena" target for allocating ops. See kt_tensor.h.
static kt_arena * g_active_arena = NULL;

void kt_arena_set_active(kt_arena * a) { g_active_arena = a; }
kt_arena * kt_arena_get_active(void)   { return g_active_arena; }

// Helper used at every op alloc site: route output to the active arena
// when one is set, otherwise to the input's own arena. Critical for
// avoiding leaks into weights_arena when ops compute on weight inputs.
static inline kt_arena * kt_aout(kt_arena * fallback) {
    return g_active_arena != NULL ? g_active_arena : fallback;
}

// Carve `bytes` from the arena, 64-byte aligned. Grows on demand.
//
// Growth policy: allocate a new slab sized to fit just this request
// (rounded up to a 1 MB minimum so follow-up small allocations
// coalesce into the same slab). The previous "double the capacity"
// policy nearly doubled peak memory — for a 600 MB peak need the slab
// chain was 1+2+4+...+512 = 1023 MB, which OOM-killed the app on iOS
// (jetsam at ~2 GB phys_footprint). Linear growth means peak slab
// total is ~1x actual usage.
static void * kt_arena_alloc(kt_arena * a, size_t bytes) {
    assert(a != NULL);
    size_t rounded = (bytes + (KT_ALIGN - 1)) & ~((size_t)(KT_ALIGN - 1));
    kt_slab * s = a->head;
    size_t aligned_used = (s->used + (KT_ALIGN - 1))
                          & ~((size_t)(KT_ALIGN - 1));
    if (aligned_used + rounded > s->capacity) {
        const size_t MIN_NEW_SLAB = (size_t)1 << 20;   // 1 MB
        size_t new_cap = rounded + KT_ALIGN;
        if (new_cap < MIN_NEW_SLAB) { new_cap = MIN_NEW_SLAB; }
        kt_slab * ns = kt_slab_new(new_cap);
        s->next = ns;
        a->head = ns;
        s = ns;
        aligned_used = 0;
    }
    void * out = s->data + aligned_used;
    s->used = aligned_used + rounded;
    return out;
}


// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

static void kt_set_packed_strides(kt_tensor * t) {
    t->nb[0] = sizeof(float);
    for (int i = 1; i < KT_MAX_DIMS; i++) {
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
}

int64_t kt_nelements(const kt_tensor * t) {
    assert(t != NULL);
    int64_t n = 1;
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        n *= t->ne[i];
    }
    return n;
}

size_t kt_nbytes(const kt_tensor * t) {
    return (size_t)kt_nelements(t) * sizeof(float);
}

bool kt_is_packed(const kt_tensor * t) {
    bool packed = true;
    int64_t expected = sizeof(float);
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        if (t->ne[i] > 1 && t->nb[i] != expected) {
            packed = false;
        }
        expected *= t->ne[i];
    }
    return packed;
}

bool kt_same_shape(const kt_tensor * a, const kt_tensor * b) {
    bool same = (a->ndim == b->ndim);
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) { same = false; }
    }
    return same;
}


// ---------------------------------------------------------------------------
// Tensor creation
// ---------------------------------------------------------------------------

static kt_tensor * kt_alloc_header(kt_arena * a) {
    kt_tensor * t = (kt_tensor *)kt_arena_alloc(a, sizeof(kt_tensor));
    memset(t, 0, sizeof(*t));
    t->arena = a;
    for (int i = 0; i < KT_MAX_DIMS; i++) { t->ne[i] = 1; }
    return t;
}

static kt_tensor * kt_alloc_with_data(kt_arena * a,
                                      int ndim,
                                      int64_t n0, int64_t n1,
                                      int64_t n2, int64_t n3) {
    kt_tensor * t = kt_alloc_header(a);
    t->ndim = ndim;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2; t->ne[3] = n3;
    kt_set_packed_strides(t);
    size_t bytes = (size_t)kt_nelements(t) * sizeof(float);
    t->data = (float *)kt_arena_alloc(a, bytes);
    return t;
}

kt_tensor * kt_new_1d(kt_arena * a, int64_t n0) {
    return kt_alloc_with_data(a, 1, n0, 1, 1, 1);
}

kt_tensor * kt_new_2d(kt_arena * a, int64_t n0, int64_t n1) {
    return kt_alloc_with_data(a, 2, n0, n1, 1, 1);
}

kt_tensor * kt_new_3d(kt_arena * a, int64_t n0, int64_t n1, int64_t n2) {
    return kt_alloc_with_data(a, 3, n0, n1, n2, 1);
}

kt_tensor * kt_new_4d(kt_arena * a, int64_t n0, int64_t n1,
                      int64_t n2, int64_t n3) {
    return kt_alloc_with_data(a, 4, n0, n1, n2, n3);
}

kt_tensor * kt_wrap_2d(kt_arena * a, float * data,
                       int64_t n0, int64_t n1) {
    kt_tensor * t = kt_alloc_header(a);
    t->ndim = 2;
    t->ne[0] = n0; t->ne[1] = n1;
    kt_set_packed_strides(t);
    t->data = data;
    return t;
}

kt_tensor * kt_wrap_3d(kt_arena * a, float * data,
                       int64_t n0, int64_t n1, int64_t n2) {
    kt_tensor * t = kt_alloc_header(a);
    t->ndim = 3;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2;
    kt_set_packed_strides(t);
    t->data = data;
    return t;
}

void kt_set_name(kt_tensor * t, const char * name) {
    assert(t != NULL && name != NULL);
    size_t n = strlen(name);
    if (n > sizeof(t->name) - 1) { n = sizeof(t->name) - 1; }
    memcpy(t->name, name, n);
    t->name[n] = '\0';
}


// ---------------------------------------------------------------------------
// Layout ops
// ---------------------------------------------------------------------------

// Build a header sharing src's data buffer (view).
static kt_tensor * kt_make_view(kt_tensor * src, int ndim,
                                int64_t n0, int64_t n1,
                                int64_t n2, int64_t n3,
                                int64_t nb0, int64_t nb1,
                                int64_t nb2, int64_t nb3,
                                size_t off_bytes) {
    kt_tensor * t = kt_alloc_header(kt_aout(src->arena));
    t->ndim = ndim;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2; t->ne[3] = n3;
    t->nb[0] = nb0; t->nb[1] = nb1; t->nb[2] = nb2; t->nb[3] = nb3;
    t->data = (float *)((char *)src->data + off_bytes);
    return t;
}

kt_tensor * kt_view_1d(kt_tensor * src, int64_t n0, size_t off_bytes) {
    return kt_make_view(src, 1, n0, 1, 1, 1,
                        src->nb[0], src->nb[0] * n0,
                        src->nb[0] * n0, src->nb[0] * n0,
                        off_bytes);
}

kt_tensor * kt_view_2d(kt_tensor * src,
                       int64_t n0, int64_t n1,
                       size_t nb1, size_t off_bytes) {
    return kt_make_view(src, 2, n0, n1, 1, 1,
                        src->nb[0], (int64_t)nb1,
                        (int64_t)nb1 * n1, (int64_t)nb1 * n1,
                        off_bytes);
}

kt_tensor * kt_view_3d(kt_tensor * src,
                       int64_t n0, int64_t n1, int64_t n2,
                       size_t nb1, size_t nb2,
                       size_t off_bytes) {
    return kt_make_view(src, 3, n0, n1, n2, 1,
                        src->nb[0], (int64_t)nb1, (int64_t)nb2,
                        (int64_t)nb2 * n2,
                        off_bytes);
}

kt_tensor * kt_reshape_2d(kt_tensor * src, int64_t n0, int64_t n1) {
    assert(kt_is_packed(src));
    assert(n0 * n1 == kt_nelements(src));
    kt_tensor * t = kt_alloc_header(kt_aout(src->arena));
    t->ndim = 2;
    t->ne[0] = n0; t->ne[1] = n1;
    kt_set_packed_strides(t);
    t->data = src->data;
    return t;
}

kt_tensor * kt_reshape_3d(kt_tensor * src,
                          int64_t n0, int64_t n1, int64_t n2) {
    assert(kt_is_packed(src));
    assert(n0 * n1 * n2 == kt_nelements(src));
    kt_tensor * t = kt_alloc_header(kt_aout(src->arena));
    t->ndim = 3;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2;
    kt_set_packed_strides(t);
    t->data = src->data;
    return t;
}

kt_tensor * kt_reshape_4d(kt_tensor * src,
                          int64_t n0, int64_t n1,
                          int64_t n2, int64_t n3) {
    assert(kt_is_packed(src));
    assert(n0 * n1 * n2 * n3 == kt_nelements(src));
    kt_tensor * t = kt_alloc_header(kt_aout(src->arena));
    t->ndim = 4;
    t->ne[0] = n0; t->ne[1] = n1; t->ne[2] = n2; t->ne[3] = n3;
    kt_set_packed_strides(t);
    t->data = src->data;
    return t;
}

kt_tensor * kt_permute(kt_tensor * src, int p0, int p1, int p2, int p3) {
    assert(p0 >= 0 && p0 < 4);
    assert(p1 >= 0 && p1 < 4);
    assert(p2 >= 0 && p2 < 4);
    assert(p3 >= 0 && p3 < 4);
    kt_tensor * t = kt_alloc_header(kt_aout(src->arena));
    t->ndim = src->ndim;
    int perm[4] = { p0, p1, p2, p3 };
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        t->ne[i] = src->ne[perm[i]];
        t->nb[i] = src->nb[perm[i]];
    }
    t->data = src->data;
    return t;
}

kt_tensor * kt_transpose(kt_tensor * src) {
    return kt_permute(src, 1, 0, 2, 3);
}

// Copy a strided tensor into a fresh packed buffer.
kt_tensor * kt_cont(kt_tensor * src) {
    kt_tensor * t = kt_alloc_with_data(kt_aout(src->arena), src->ndim,
                                       src->ne[0], src->ne[1],
                                       src->ne[2], src->ne[3]);
    const int64_t n0 = src->ne[0];
    const int64_t n1 = src->ne[1];
    const int64_t n2 = src->ne[2];
    const int64_t n3 = src->ne[3];
    const int64_t s0 = src->nb[0];
    const int64_t s1 = src->nb[1];
    const int64_t s2 = src->nb[2];
    const int64_t s3 = src->nb[3];
    const char * sb = (const char *)src->data;
    float * dst = t->data;
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const char * row = sb + i3 * s3 + i2 * s2 + i1 * s1;
                if (s0 == (int64_t)sizeof(float)) {
                    memcpy(dst, row, (size_t)n0 * sizeof(float));
                    dst += n0;
                } else {
                    for (int64_t i0 = 0; i0 < n0; i0++) {
                        *dst++ = *(const float *)(row + i0 * s0);
                    }
                }
            }
        }
    }
    return t;
}

kt_tensor * kt_cont_2d(kt_tensor * src, int64_t n0, int64_t n1) {
    kt_tensor * packed = kt_cont(src);
    return kt_reshape_2d(packed, n0, n1);
}

void kt_cpy(const kt_tensor * src, kt_tensor * dst) {
    assert(kt_same_shape(src, dst));
    const int64_t n0 = src->ne[0];
    const int64_t n1 = src->ne[1];
    const int64_t n2 = src->ne[2];
    const int64_t n3 = src->ne[3];
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const char * srow = (const char *)src->data
                                  + i3 * src->nb[3]
                                  + i2 * src->nb[2]
                                  + i1 * src->nb[1];
                char * drow = (char *)dst->data
                            + i3 * dst->nb[3]
                            + i2 * dst->nb[2]
                            + i1 * dst->nb[1];
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    *(float *)(drow + i0 * dst->nb[0]) =
                        *(const float *)(srow + i0 * src->nb[0]);
                }
            }
        }
    }
}

// Concat along axis. Both inputs must agree on all other axes and be
// packed (for v1; strided concat is harder and not needed yet).
kt_tensor * kt_concat(kt_tensor * a, kt_tensor * b, int axis) {
    assert(axis >= 0 && axis < KT_MAX_DIMS);
    assert(a->ndim == b->ndim);
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        if (i != axis) { assert(a->ne[i] == b->ne[i]); }
    }
    assert(kt_is_packed(a) && kt_is_packed(b));
    int64_t out_ne[KT_MAX_DIMS];
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        out_ne[i] = (i == axis) ? (a->ne[i] + b->ne[i]) : a->ne[i];
    }
    kt_tensor * t = kt_alloc_with_data(kt_aout(a->arena), a->ndim,
                                       out_ne[0], out_ne[1],
                                       out_ne[2], out_ne[3]);
    // Iterate over the "outer" axes (> axis), then within each slab
    // copy A's row then B's row along `axis`.
    int64_t outer = 1;
    for (int i = axis + 1; i < KT_MAX_DIMS; i++) { outer *= a->ne[i]; }
    int64_t inner = 1;
    for (int i = 0; i < axis; i++) { inner *= a->ne[i]; }
    size_t row_a = (size_t)inner * (size_t)a->ne[axis] * sizeof(float);
    size_t row_b = (size_t)inner * (size_t)b->ne[axis] * sizeof(float);
    char * dst = (char *)t->data;
    const char * sa = (const char *)a->data;
    const char * sb = (const char *)b->data;
    for (int64_t k = 0; k < outer; k++) {
        memcpy(dst, sa + k * row_a, row_a);
        dst += row_a;
        memcpy(dst, sb + k * row_b, row_b);
        dst += row_b;
    }
    return t;
}

// Same as kt_repeat but takes the target shape as ints. Avoids the
// caller having to materialize a full template tensor (which would
// allocate the full output buffer just to be ignored).
kt_tensor * kt_repeat_to(kt_tensor * src, int ndim,
                         int64_t n0, int64_t n1,
                         int64_t n2, int64_t n3) {
    kt_tensor template;
    memset(&template, 0, sizeof(template));
    template.ndim = ndim;
    template.ne[0] = n0; template.ne[1] = n1;
    template.ne[2] = n2; template.ne[3] = n3;
    return kt_repeat(src, &template);
}

// repeat: tile src to match shape_like. Each src axis must be 1 or
// equal to shape_like's. (For v1: trivial broadcast tiling.)
kt_tensor * kt_repeat(kt_tensor * src, const kt_tensor * shape_like) {
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        assert(src->ne[i] == 1 || src->ne[i] == shape_like->ne[i]);
    }
    kt_tensor * t = kt_alloc_with_data(kt_aout(src->arena),
                                       shape_like->ndim,
                                       shape_like->ne[0],
                                       shape_like->ne[1],
                                       shape_like->ne[2],
                                       shape_like->ne[3]);
    const int64_t n0 = t->ne[0], n1 = t->ne[1];
    const int64_t n2 = t->ne[2], n3 = t->ne[3];
    const int64_t r0 = src->ne[0] == 1 ? 0 : src->nb[0];
    const int64_t r1 = src->ne[1] == 1 ? 0 : src->nb[1];
    const int64_t r2 = src->ne[2] == 1 ? 0 : src->nb[2];
    const int64_t r3 = src->ne[3] == 1 ? 0 : src->nb[3];
    float * dst = t->data;
    const char * sb = (const char *)src->data;
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const char * row = sb + i3 * r3 + i2 * r2 + i1 * r1;
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    *dst++ = *(const float *)(row + i0 * r0);
                }
            }
        }
    }
    return t;
}

kt_tensor * kt_get_rows(kt_tensor * data,
                        const int32_t * ids, int n_ids) {
    assert(data->ndim == 2);
    assert(kt_is_packed(data));
    const int64_t embed = data->ne[0];
    const int64_t vocab = data->ne[1];
    kt_tensor * t = kt_new_2d(kt_aout(data->arena), embed,
                              (int64_t)n_ids);
    for (int i = 0; i < n_ids; i++) {
        int32_t row = ids[i];
        assert(row >= 0 && (int64_t)row < vocab);
        memcpy(t->data + (size_t)i * (size_t)embed,
               data->data + (size_t)row * (size_t)embed,
               (size_t)embed * sizeof(float));
    }
    return t;
}


// ---------------------------------------------------------------------------
// Broadcasting kernels (ggml-style: y.ne[i] must be 1 or equal to x.ne[i]).
// ---------------------------------------------------------------------------

// `y` is broadcastable against `x` iff for every axis i, y.ne[i] is 1
// (broadcast along that axis) or y.ne[i] equals x.ne[i]. Returns 1 on
// pass, 0 on shape mismatch.
static int kt_broadcastable(const kt_tensor * x, const kt_tensor * y) {
    int ok = 1;
    for (int i = 0; i < KT_MAX_DIMS; i++) {
        if (!(y->ne[i] == 1 || y->ne[i] == x->ne[i])) { ok = 0; }
    }
    return ok;
}

typedef enum { KT_BIN_ADD, KT_BIN_SUB, KT_BIN_MUL, KT_BIN_DIV } kt_bin;

// Vector-vector for two contiguous buffers via vDSP (Apple's hand-tuned
// SIMD kernels — fast even at -O0 because the work is in libBLAS).
static void kt_vec_vv(kt_bin op, const float * x, const float * y,
                      float * out, int64_t n) {
    const vDSP_Length N = (vDSP_Length)n;
    switch (op) {
        case KT_BIN_ADD: vDSP_vadd(x, 1, y, 1, out, 1, N); break;
        // vDSP_vsub computes A - B with arg order (B, IB, A, IA, ...).
        case KT_BIN_SUB: vDSP_vsub(y, 1, x, 1, out, 1, N); break;
        case KT_BIN_MUL: vDSP_vmul(x, 1, y, 1, out, 1, N); break;
        // vDSP_vdiv computes A / B with arg order (B, IB, A, IA, ...).
        case KT_BIN_DIV: vDSP_vdiv(y, 1, x, 1, out, 1, N); break;
    }
}

// Vector-scalar (y is broadcast as a single value) via vDSP.
static void kt_vec_vs(kt_bin op, const float * x, float s,
                      float * out, int64_t n) {
    const vDSP_Length N = (vDSP_Length)n;
    float arg;
    switch (op) {
        case KT_BIN_ADD: vDSP_vsadd(x, 1, &s, out, 1, N); break;
        case KT_BIN_SUB: arg = -s;
                         vDSP_vsadd(x, 1, &arg, out, 1, N); break;
        case KT_BIN_MUL: vDSP_vsmul(x, 1, &s, out, 1, N); break;
        case KT_BIN_DIV: arg = 1.0f / s;
                         vDSP_vsmul(x, 1, &arg, out, 1, N); break;
    }
}

// Scalar fallback for the strided / broadcast path. Switch dispatch
// (no function pointer) so -O0 doesn't pay a call per element.
static inline float kt_scalar_op(kt_bin op, float a, float b) {
    switch (op) {
        case KT_BIN_ADD: return a + b;
        case KT_BIN_SUB: return a - b;
        case KT_BIN_MUL: return a * b;
        case KT_BIN_DIV: return a / b;
    }
    return 0.0f;
}

static kt_tensor * kt_apply_binop(kt_tensor * x, kt_tensor * y,
                                  kt_bin op) {
    assert(kt_broadcastable(x, y));
    kt_tensor * out = kt_alloc_with_data(kt_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0], n1 = x->ne[1];
    const int64_t n2 = x->ne[2], n3 = x->ne[3];
    const float * xb = x->data;
    const float * yb = y->data;
    float * ob = out->data;
    int64_t total = kt_nelements(x);
    // Fast path 1: identical packed shape -> vDSP vector-vector kernel.
    if (kt_same_shape(x, y) && kt_is_packed(x) && kt_is_packed(y)) {
        kt_vec_vv(op, xb, yb, ob, total);
    } else if (kt_nelements(y) == 1) {
        // Fast path 2: y is scalar -> vDSP vector-scalar kernel.
        if (kt_is_packed(x)) {
            kt_vec_vs(op, xb, yb[0], ob, total);
        } else {
            float s = yb[0];
            for (int64_t i3 = 0; i3 < n3; i3++) {
              for (int64_t i2 = 0; i2 < n2; i2++) {
                for (int64_t i1 = 0; i1 < n1; i1++) {
                  for (int64_t i0 = 0; i0 < n0; i0++) {
                    const float * xp = (const float *)
                        ((const char *)xb + i3 * x->nb[3]
                                          + i2 * x->nb[2]
                                          + i1 * x->nb[1]
                                          + i0 * x->nb[0]);
                    int64_t oi = ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
                    ob[oi] = kt_scalar_op(op, *xp, s);
                  }
                }
              }
            }
        }
    } else {
        // Fast path 3: general broadcast where both x and y are packed
        // along ne[0] (the inner axis). For each row, dispatch the
        // inner kernel through vDSP — vector-vector when y.ne[0] equals
        // x.ne[0], vector-scalar when y broadcasts along ne[0].
        const int64_t ys0 = (y->ne[0] == 1) ? 0 : y->nb[0];
        const int64_t ys1 = (y->ne[1] == 1) ? 0 : y->nb[1];
        const int64_t ys2 = (y->ne[2] == 1) ? 0 : y->nb[2];
        const int64_t ys3 = (y->ne[3] == 1) ? 0 : y->nb[3];
        const int x_inner_packed = (x->nb[0] == (int64_t)sizeof(float));
        const int y_inner_packed_or_scalar =
            (ys0 == 0) || (ys0 == (int64_t)sizeof(float));
        for (int64_t i3 = 0; i3 < n3; i3++) {
          for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
              const char * xrow = (const char *)xb
                                + i3 * x->nb[3]
                                + i2 * x->nb[2]
                                + i1 * x->nb[1];
              const char * yrow = (const char *)yb
                                + i3 * ys3 + i2 * ys2 + i1 * ys1;
              float * orow = ob
                           + ((i3 * n2 + i2) * n1 + i1) * n0;
              if (x_inner_packed && y_inner_packed_or_scalar) {
                  if (ys0 == 0) {
                      kt_vec_vs(op, (const float *)xrow,
                                *(const float *)yrow, orow, n0);
                  } else {
                      kt_vec_vv(op, (const float *)xrow,
                                (const float *)yrow, orow, n0);
                  }
              } else if (ys0 == 0) {
                  float yv = *(const float *)yrow;
                  for (int64_t i0 = 0; i0 < n0; i0++) {
                      float xv = *(const float *)(xrow + i0 * x->nb[0]);
                      orow[i0] = kt_scalar_op(op, xv, yv);
                  }
              } else {
                  for (int64_t i0 = 0; i0 < n0; i0++) {
                      float xv = *(const float *)(xrow + i0 * x->nb[0]);
                      float yv = *(const float *)(yrow + i0 * ys0);
                      orow[i0] = kt_scalar_op(op, xv, yv);
                  }
              }
            }
          }
        }
    }
    return out;
}

kt_tensor * kt_add(kt_tensor * x, kt_tensor * y) {
    return kt_apply_binop(x, y, KT_BIN_ADD);
}
kt_tensor * kt_sub(kt_tensor * x, kt_tensor * y) {
    return kt_apply_binop(x, y, KT_BIN_SUB);
}
kt_tensor * kt_mul(kt_tensor * x, kt_tensor * y) {
    return kt_apply_binop(x, y, KT_BIN_MUL);
}
kt_tensor * kt_div(kt_tensor * x, kt_tensor * y) {
    return kt_apply_binop(x, y, KT_BIN_DIV);
}


// ---------------------------------------------------------------------------
// Elementwise unary
// ---------------------------------------------------------------------------

typedef enum {
    KT_U_SCALE, KT_U_SIGMOID, KT_U_TANH, KT_U_LRELU, KT_U_GELU,
    KT_U_STEP, KT_U_SIN, KT_U_COS, KT_U_EXP, KT_U_SQRT
} kt_unop;

// Vector-only fast paths via vForce (transcendentals) and vDSP (linear
// ops). These are SIMD ASM kernels in Accelerate, so they hit memory
// bandwidth even when the calling code is built at -O0.
static void kt_vec_unary(kt_unop op, const float * x, float * out,
                         int64_t n, float param) {
    int len = (int)n;
    const vDSP_Length N = (vDSP_Length)n;
    switch (op) {
    case KT_U_SCALE: {
        float s = param;
        vDSP_vsmul(x, 1, &s, out, 1, N);
        break;
    }
    case KT_U_SIGMOID: {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Implemented as: out = -x; out = exp(out); out = 1+out;
        // out = 1/out — three vForce/vDSP passes; still beats per-elem
        // function calls at -O0.
        vDSP_vneg(x, 1, out, 1, N);
        vvexpf(out, out, &len);
        float one = 1.0f;
        vDSP_vsadd(out, 1, &one, out, 1, N);
        vvrecf(out, out, &len);
        break;
    }
    case KT_U_TANH: vvtanhf(out, x, &len); break;
    case KT_U_LRELU: {
        // leaky_relu(x, s) = max(x, x*s) when s in (0, 1)
        // Implemented as: pos = max(x, 0); neg = min(x, 0)*s; out = pos+neg
        // For typical 0 < s < 1 this matches max(x, x*s).
        float zero = 0.0f;
        // Compute slope*x into out; then take max(x, slope*x).
        vDSP_vsmul(x, 1, &param, out, 1, N);
        vDSP_vmax(x, 1, out, 1, out, 1, N);
        (void)zero;
        break;
    }
    case KT_U_GELU: {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        const float inv_sqrt2 = (float)M_SQRT1_2;
        vDSP_vsmul(x, 1, &inv_sqrt2, out, 1, N);
        // No vForce vverf — fall back to scalar erff per element. Still
        // a single pass.
        for (int64_t i = 0; i < n; i++) {
            float v = out[i];
            out[i] = 0.5f * x[i] * (1.0f + erff(v));
        }
        break;
    }
    case KT_U_STEP:
        for (int64_t i = 0; i < n; i++) out[i] = x[i] > 0.0f ? 1.0f : 0.0f;
        break;
    case KT_U_SIN:  vvsinf (out, x, &len); break;
    case KT_U_COS:  vvcosf (out, x, &len); break;
    case KT_U_EXP:  vvexpf (out, x, &len); break;
    case KT_U_SQRT: vvsqrtf(out, x, &len); break;
    }
}

// Scalar fallback for the strided path. Switch dispatch (no function
// pointer call per element).
static inline float kt_scalar_unary(kt_unop op, float v, float p) {
    switch (op) {
    case KT_U_SCALE:   return v * p;
    case KT_U_SIGMOID: return 1.0f / (1.0f + expf(-v));
    case KT_U_TANH:    return tanhf(v);
    case KT_U_LRELU:   return v > 0.0f ? v : v * p;
    case KT_U_GELU:    return 0.5f * v *
                              (1.0f + erff(v * (float)M_SQRT1_2));
    case KT_U_STEP:    return v > 0.0f ? 1.0f : 0.0f;
    case KT_U_SIN:     return sinf(v);
    case KT_U_COS:     return cosf(v);
    case KT_U_EXP:     return expf(v);
    case KT_U_SQRT:    return sqrtf(v);
    }
    return 0.0f;
}

static kt_tensor * kt_apply_unary(kt_tensor * x, kt_unop op, float param) {
    kt_tensor * out = kt_alloc_with_data(kt_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    int64_t n = kt_nelements(x);
    if (kt_is_packed(x)) {
        kt_vec_unary(op, x->data, out->data, n, param);
    } else {
        // Strided input: indexed walk. Per-row vector path would need
        // packed inner stride; we can still handle that inline if it
        // matters later.
        const int64_t n0 = x->ne[0], n1 = x->ne[1];
        const int64_t n2 = x->ne[2], n3 = x->ne[3];
        for (int64_t i3 = 0; i3 < n3; i3++) {
            for (int64_t i2 = 0; i2 < n2; i2++) {
                for (int64_t i1 = 0; i1 < n1; i1++) {
                    for (int64_t i0 = 0; i0 < n0; i0++) {
                        const float * xp = (const float *)
                            ((const char *)x->data + i3 * x->nb[3]
                                                   + i2 * x->nb[2]
                                                   + i1 * x->nb[1]
                                                   + i0 * x->nb[0]);
                        int64_t oi = ((i3 * n2 + i2) * n1 + i1)
                                         * n0 + i0;
                        out->data[oi] = kt_scalar_unary(op, *xp, param);
                    }
                }
            }
        }
    }
    return out;
}

kt_tensor * kt_scale     (kt_tensor * x, float s) {
    return kt_apply_unary(x, KT_U_SCALE, s);
}
kt_tensor * kt_sigmoid   (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_SIGMOID, 0.0f);
}
kt_tensor * kt_tanh      (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_TANH, 0.0f);
}
kt_tensor * kt_leaky_relu(kt_tensor * x, float slope) {
    return kt_apply_unary(x, KT_U_LRELU, slope);
}
kt_tensor * kt_gelu_erf  (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_GELU, 0.0f);
}
kt_tensor * kt_step      (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_STEP, 0.0f);
}
kt_tensor * kt_sin       (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_SIN, 0.0f);
}
kt_tensor * kt_cos       (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_COS, 0.0f);
}
kt_tensor * kt_exp       (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_EXP, 0.0f);
}
kt_tensor * kt_sqrt      (kt_tensor * x) {
    return kt_apply_unary(x, KT_U_SQRT, 0.0f);
}

kt_tensor * kt_atan2(kt_tensor * y, kt_tensor * x) {
    assert(kt_same_shape(x, y));
    assert(kt_is_packed(x) && kt_is_packed(y));
    kt_tensor * out = kt_alloc_with_data(kt_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    int64_t n = kt_nelements(x);
    for (int64_t i = 0; i < n; i++) {
        out->data[i] = atan2f(y->data[i], x->data[i]);
    }
    return out;
}


// ---------------------------------------------------------------------------
// Reductions / normalization
// ---------------------------------------------------------------------------

// Reduce along ne[axis], keepdim (write per-outer scalar into out).
// LayerNorm: subtract mean, divide by sqrt(var + eps). No scale/bias.
kt_tensor * kt_norm(kt_tensor * x, int axis, float eps) {
    // For kittens-tts the only axis we ever normalize over is the
    // innermost (ne[0]) — matches ggml_norm semantics.
    assert(axis == 0);
    assert(kt_is_packed(x));
    kt_tensor * out = kt_alloc_with_data(kt_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0];
    int64_t outer = kt_nelements(x) / n0;
    const float * xb = x->data;
    float * ob = out->data;
    const vDSP_Length N = (vDSP_Length)n0;
    for (int64_t r = 0; r < outer; r++) {
        const float * row = xb + r * n0;
        float * orow = ob + r * n0;
        // mean(x) and mean(x^2) via vDSP — single SIMD pass each.
        float mean = 0.0f;
        float meanSq = 0.0f;
        vDSP_meanv (row, 1, &mean,   N);
        vDSP_measqv(row, 1, &meanSq, N);
        const float var = meanSq - mean * mean;
        const float invstd = 1.0f / sqrtf(var + eps);
        // out = (x - mean) * invstd  ==  x * invstd + (-mean * invstd)
        // Encoded as a single fused multiply-add via vDSP_vsmsa.
        const float bias = -mean * invstd;
        vDSP_vsmsa((float *)row, 1, (float *)&invstd, (float *)&bias,
                   orow, 1, N);
    }
    return out;
}

kt_tensor * kt_softmax(kt_tensor * x, int axis, float scale) {
    assert(axis == 0);
    assert(kt_is_packed(x));
    kt_tensor * out = kt_alloc_with_data(kt_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0];
    int64_t outer = kt_nelements(x) / n0;
    const float * xb = x->data;
    float * ob = out->data;
    const vDSP_Length N = (vDSP_Length)n0;
    int Nint = (int)n0;
    for (int64_t r = 0; r < outer; r++) {
        const float * row = xb + r * n0;
        float * orow = ob + r * n0;
        // 1) orow = row * scale
        vDSP_vsmul(row, 1, (float *)&scale, orow, 1, N);
        // 2) max(orow)
        float mx;
        vDSP_maxv(orow, 1, &mx, N);
        // 3) orow -= mx
        float negmx = -mx;
        vDSP_vsadd(orow, 1, &negmx, orow, 1, N);
        // 4) orow = exp(orow)
        vvexpf(orow, orow, &Nint);
        // 5) sum and divide
        float sum;
        vDSP_sve(orow, 1, &sum, N);
        float inv = 1.0f / sum;
        vDSP_vsmul(orow, 1, &inv, orow, 1, N);
    }
    return out;
}

kt_tensor * kt_cumsum(kt_tensor * x, int axis) {
    assert(axis == 0);
    assert(kt_is_packed(x));
    kt_tensor * out = kt_alloc_with_data(kt_aout(x->arena), x->ndim,
                                         x->ne[0], x->ne[1],
                                         x->ne[2], x->ne[3]);
    const int64_t n0 = x->ne[0];
    int64_t outer = kt_nelements(x) / n0;
    const float * xb = x->data;
    float * ob = out->data;
    for (int64_t r = 0; r < outer; r++) {
        const float * row = xb + r * n0;
        float * orow = ob + r * n0;
        float acc = 0.0f;
        for (int64_t i = 0; i < n0; i++) {
            acc += row[i];
            orow[i] = acc;
        }
    }
    return out;
}


// ---------------------------------------------------------------------------
// Linear algebra: kt_mul_mat dispatches to cblas_sgemm.
//
// Convention (matches ggml_mul_mat):
//   w: ne[0]=K (contraction, innermost), ne[1]=M (output rows)
//   x: ne[0]=K, ne[1]=N
//   out: ne[0]=M, ne[1]=N
// Math: out[m, n] = sum_k w[k, m] * x[k, n].
// Both inputs must be packed for v1.
// ---------------------------------------------------------------------------

kt_tensor * kt_mul_mat(kt_tensor * w, kt_tensor * x) {
    // Convention exactly matches ggml_mul_mat(a, b):
    //   a (= w) stored ne[0]=K, ne[1]=Nw  (PyTorch (Nw, K) row-major)
    //   b (= x) stored ne[0]=K, ne[1]=Nx  (PyTorch (Nx, K) row-major)
    //   c (= out) stored ne[0]=Nw, ne[1]=Nx
    // Batched: ne[2] and ne[3] are batch dims; w and x must agree on
    // them. Output preserves the batch dims.
    if (!kt_is_packed(w)) { w = kt_cont(w); }
    if (!kt_is_packed(x)) { x = kt_cont(x); }
    assert(w->ne[0] == x->ne[0]);
    assert(w->ne[2] == x->ne[2]);
    assert(w->ne[3] == x->ne[3]);
    const int64_t K  = w->ne[0];
    const int64_t Nw = w->ne[1];
    const int64_t Nx = x->ne[1];
    const int64_t B2 = w->ne[2];
    const int64_t B3 = w->ne[3];
    kt_arena * oa = kt_aout(w->arena);
    kt_tensor * out;
    if (B2 == 1 && B3 == 1) {
        out = kt_new_2d(oa, Nw, Nx);
    } else if (B3 == 1) {
        out = kt_new_3d(oa, Nw, Nx, B2);
    } else {
        out = kt_new_4d(oa, Nw, Nx, B2, B3);
    }
    const size_t w_stride = (size_t)Nw * (size_t)K;
    const size_t x_stride = (size_t)Nx * (size_t)K;
    const size_t o_stride = (size_t)Nw * (size_t)Nx;
    for (int64_t b3 = 0; b3 < B3; b3++) {
        for (int64_t b2 = 0; b2 < B2; b2++) {
            const size_t b = (size_t)b3 * (size_t)B2 + (size_t)b2;
            const float * w_b = w->data + b * w_stride;
            const float * x_b = x->data + b * x_stride;
            float * o_b = out->data + b * o_stride;
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,  // x rm (Nx, K)
                        CblasTrans,    // w rm (Nw, K) -> (K, Nw)
                        (int)Nx, (int)Nw, (int)K,
                        1.0f,
                        x_b, (int)K,
                        w_b, (int)K,
                        0.0f,
                        o_b, (int)Nw);
        }
    }
    return out;
}


// ---------------------------------------------------------------------------
// 1D convolution.
//
// Layout: input  (B, Cin, L) with B in ne[2], Cin in ne[1], L in ne[0].
//         weight (Cout, Cin, K) with K in ne[0], Cin in ne[1],
//                Cout in ne[2].
//         output (B, Cout, Lout).
// ---------------------------------------------------------------------------

static int64_t kt_conv_out_len(int64_t L_in, int K,
                               int stride, int pad, int dilation) {
    return (L_in + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
}

// Produce (B, Cin*K, Lout) im2col matrix. Input (B, Cin, L).
kt_tensor * kt_im2col(kt_tensor * x,
                      int kernel, int stride, int pad, int dilation) {
    assert(kt_is_packed(x));
    const int64_t L_in = x->ne[0];
    const int64_t Cin  = x->ne[1];
    const int64_t B    = x->ne[2];
    const int64_t Lout = kt_conv_out_len(L_in, kernel,
                                         stride, pad, dilation);
    kt_tensor * out = kt_new_3d(kt_aout(x->arena),
                                Lout, Cin * (int64_t)kernel, B);
    const float * xb = x->data;
    float * ob = out->data;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t c = 0; c < Cin; c++) {
            for (int k = 0; k < kernel; k++) {
                int64_t off = b * Cin * (int64_t)kernel * Lout
                            + (c * (int64_t)kernel + k) * Lout;
                for (int64_t lo = 0; lo < Lout; lo++) {
                    int64_t li = lo * stride - pad + k * dilation;
                    float v = 0.0f;
                    if (li >= 0 && li < L_in) {
                        v = xb[b * Cin * L_in + c * L_in + li];
                    }
                    ob[off + lo] = v;
                }
            }
        }
    }
    return out;
}

kt_tensor * kt_conv_1d(kt_tensor * w, kt_tensor * x,
                       int stride, int pad, int dilation) {
    // w stored (K, Cin, Cout) — ne[0]=K, ne[1]=Cin, ne[2]=Cout.
    //   Memory offset for w[co, ci, k] is k + ci*K + co*Cin*K.
    //   As a row-major matrix it's (Cout, Cin*K), with each "row" co
    //   holding Cin*K elements ordered (ci outer, k inner).
    // x stored (L, Cin, B). Memory offset b*Cin*L + ci*L + l.
    // im2col output cols stored (Lout, Cin*K, B); memory base
    //   b*Cin*K*Lout + z*Lout + lo, where z = ci*K + k.
    //   As row-major per batch: cols[b] is (Cin*K, Lout) with lda=Lout.
    // For each batch b, sgemm:
    //   out[b] row-major (Cout, Lout) = W (Cout, Cin*K)
    //                                  @ cols[b] (Cin*K, Lout).
    assert(kt_is_packed(w));
    assert(kt_is_packed(x));
    const int64_t K    = w->ne[0];
    const int64_t Cin  = w->ne[1];
    const int64_t Cout = w->ne[2];
    const int64_t L_in = x->ne[0];
    const int64_t B    = x->ne[2];
    assert(x->ne[1] == Cin);
    const int64_t Lout = kt_conv_out_len(L_in, (int)K,
                                         stride, pad, dilation);
    kt_tensor * cols = kt_im2col(x, (int)K, stride, pad, dilation);
    kt_tensor * out  = kt_new_3d(kt_aout(w->arena), Lout, Cout, B);
    for (int64_t b = 0; b < B; b++) {
        const float * c_b = cols->data
                          + b * (size_t)(Cin * K) * (size_t)Lout;
        float * o_b = out->data + b * (size_t)Cout * (size_t)Lout;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    (int)Cout, (int)Lout, (int)(Cin * K),
                    1.0f,
                    w->data, (int)(Cin * K),
                    c_b,     (int)Lout,
                    0.0f,
                    o_b,     (int)Lout);
    }
    return out;
}

kt_tensor * kt_conv_1d_dw(kt_tensor * w, kt_tensor * x,
                          int stride, int pad, int dilation) {
    // Depthwise: w (K, 1, C), x (L, C, B), out (Lout, C, B).
    assert(kt_is_packed(w));
    assert(kt_is_packed(x));
    const int64_t K = w->ne[0];
    const int64_t C = w->ne[2];
    assert(w->ne[1] == 1);
    assert(x->ne[1] == C);
    const int64_t L_in = x->ne[0];
    const int64_t B    = x->ne[2];
    const int64_t Lout = kt_conv_out_len(L_in, (int)K,
                                         stride, pad, dilation);
    kt_tensor * out = kt_new_3d(kt_aout(w->arena), Lout, C, B);
    const float * xb = x->data;
    const float * wb = w->data;
    float * ob = out->data;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t c = 0; c < C; c++) {
            const float * xrow = xb + b * C * L_in + c * L_in;
            const float * wrow = wb + c * K;
            float * orow = ob + b * C * Lout + c * Lout;
            for (int64_t lo = 0; lo < Lout; lo++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    int64_t li = lo * stride - pad + k * dilation;
                    if (li >= 0 && li < L_in) {
                        acc += wrow[k] * xrow[li];
                    }
                }
                orow[lo] = acc;
            }
        }
    }
    return out;
}

kt_tensor * kt_conv_transpose_1d(kt_tensor * w, kt_tensor * x,
                                 int stride, int pad) {
    // w (K, Cout, Cin) — ne[0]=K, ne[1]=Cout, ne[2]=Cin (PyTorch order).
    // x (Lin, Cin, B), out (Lout, Cout, B).
    assert(kt_is_packed(w));
    assert(kt_is_packed(x));
    const int64_t K    = w->ne[0];
    const int64_t Cout = w->ne[1];
    const int64_t Cin  = w->ne[2];
    const int64_t Lin  = x->ne[0];
    const int64_t B    = x->ne[2];
    assert(x->ne[1] == Cin);
    int64_t Lfull = (Lin - 1) * stride + K;
    int64_t Lout = Lfull - 2 * pad;
    if (Lout < 0) { Lout = 0; }
    kt_tensor * out = kt_new_3d(kt_aout(w->arena), Lout, Cout, B);
    memset(out->data, 0, (size_t)kt_nbytes(out));
    // Scatter form.
    for (int64_t b = 0; b < B; b++) {
        for (int64_t co = 0; co < Cout; co++) {
            float * orow = out->data + b * Cout * Lout + co * Lout;
            for (int64_t ci = 0; ci < Cin; ci++) {
                const float * xrow = x->data
                                   + b * Cin * Lin + ci * Lin;
                const float * wrow = w->data
                                   + ci * Cout * K + co * K;
                for (int64_t li = 0; li < Lin; li++) {
                    float xv = xrow[li];
                    int64_t base = li * stride - pad;
                    for (int64_t k = 0; k < K; k++) {
                        int64_t lo = base + k;
                        if (lo >= 0 && lo < Lout) {
                            orow[lo] += wrow[k] * xv;
                        }
                    }
                }
            }
        }
    }
    return out;
}

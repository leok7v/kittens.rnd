// test_kt_vs_ggml.c -- per-op parity tests against ggml on random inputs.
//
// For every kt_* op, the test:
//   1. Generates a deterministic random input buffer.
//   2. Runs the equivalent ggml graph and captures its output.
//   3. Runs the kt_* op on a copy of the same input.
//   4. Computes element-wise RMSE between the two outputs.
//   5. Reports PASS if RMSE < tol, else dumps the first few mismatches.
//
// This is the gate that catches stride / convention / off-by-one bugs
// that the standalone test_kt_basic.c can't see (small inputs are
// always in some corner of the broadcast / padding space).

#include "kt_tensor.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

// Deterministic PRNG (xorshift32) so tests are reproducible.
static uint32_t g_seed = 0x1234567Bu;
static float rnd_uniform(void) {
    uint32_t x = g_seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_seed = x;
    // Map to (-1, 1).
    return ((float)x / (float)0xFFFFFFFFu) * 2.0f - 1.0f;
}
static void fill_random(float * data, int64_t n) {
    for (int64_t i = 0; i < n; i++) { data[i] = rnd_uniform(); }
}

static float rmse(const float * a, const float * b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)sqrt(s / (double)n);
}

static float max_abs_diff(const float * a, const float * b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) { m = d; }
    }
    return m;
}

static void report(const char * name,
                   const float * got, const float * want,
                   int64_t n, float tol) {
    float r = rmse(got, want, n);
    float m = max_abs_diff(got, want, n);
    bool ok = (r < tol);
    if (ok) {
        printf("PASS %-26s  RMSE %.3e  max %.3e  (n=%lld)\n",
               name, r, m, (long long)n);
    } else {
        printf("FAIL %-26s  RMSE %.3e  max %.3e  tol %.3e\n",
               name, r, m, tol);
        for (int64_t i = 0; i < n && i < 8; i++) {
            printf("    [%lld] kt %.6f  ggml %.6f  diff %.3e\n",
                   (long long)i, got[i], want[i],
                   got[i] - want[i]);
        }
        g_failures++;
    }
}


// ---------------------------------------------------------------------------
// ggml context helper. Each test case gets its own context + backend.
// ---------------------------------------------------------------------------

typedef struct {
    void *                  mem;
    struct ggml_context *   ctx;
    struct ggml_cgraph *    gf;
    ggml_backend_t          backend;
    ggml_gallocr_t          ga;
} gg_env;

static gg_env gg_open(void) {
    gg_env e;
    size_t mem_size = ggml_tensor_overhead() * 1024ull
                    + ggml_graph_overhead_custom(1024, false);
    e.mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, e.mem, true };
    e.ctx = ggml_init(ip);
    e.gf = ggml_new_graph_custom(e.ctx, 1024, false);
    e.backend = ggml_backend_cpu_init();
    e.ga = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(e.backend));
    return e;
}

static void gg_finalize_and_alloc(gg_env * e, struct ggml_tensor * out) {
    ggml_set_output(out);
    ggml_build_forward_expand(e->gf, out);
    ggml_gallocr_alloc_graph(e->ga, e->gf);
}

static void gg_close(gg_env * e) {
    ggml_gallocr_free(e->ga);
    ggml_backend_free(e->backend);
    ggml_free(e->ctx);
    free(e->mem);
}

static struct ggml_tensor * gg_input_f32_1d(gg_env * e, int64_t n) {
    struct ggml_tensor * t = ggml_new_tensor_1d(e->ctx, GGML_TYPE_F32, n);
    ggml_set_input(t);
    return t;
}
static struct ggml_tensor * gg_input_f32_2d(gg_env * e,
                                             int64_t n0, int64_t n1) {
    struct ggml_tensor * t = ggml_new_tensor_2d(e->ctx, GGML_TYPE_F32,
                                                n0, n1);
    ggml_set_input(t);
    return t;
}
static struct ggml_tensor * gg_input_f32_3d(gg_env * e,
                                             int64_t n0, int64_t n1,
                                             int64_t n2) {
    struct ggml_tensor * t = ggml_new_tensor_3d(e->ctx, GGML_TYPE_F32,
                                                n0, n1, n2);
    ggml_set_input(t);
    return t;
}

static void gg_set_data(struct ggml_tensor * t, const void * src,
                        size_t bytes) {
    ggml_backend_tensor_set(t, src, 0, bytes);
}

static void gg_get_data(struct ggml_tensor * t, void * dst,
                        size_t bytes) {
    ggml_backend_tensor_get(t, dst, 0, bytes);
}


// ---------------------------------------------------------------------------
// Per-op tests
// ---------------------------------------------------------------------------

static void test_add_full(void) {
    const int64_t L = 17, C = 5, B = 1;
    const int64_t n = L * C * B;
    float * a = (float *)malloc(n * sizeof(float));
    float * b = (float *)malloc(n * sizeof(float));
    fill_random(a, n); fill_random(b, n);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_3d(&e, L, C, B);
    struct ggml_tensor * gb_t = gg_input_f32_3d(&e, L, C, B);
    struct ggml_tensor * gy = ggml_add(e.ctx, ga_t, gb_t);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    gg_set_data(gb_t, b, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_3d(ka, L, C, B);
    kt_tensor * ky = kt_new_3d(ka, L, C, B);
    memcpy(kx->data, a, n * sizeof(float));
    memcpy(ky->data, b, n * sizeof(float));
    kt_tensor * kout = kt_add(kx, ky);
    report("add (full)", kout->data, want, n, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(b); free(want);
}

static void test_add_channel_bias(void) {
    const int64_t L = 19, C = 7, B = 1;
    const int64_t n = L * C * B;
    float * a = (float *)malloc(n * sizeof(float));
    float * b = (float *)malloc(C * sizeof(float));
    fill_random(a, n); fill_random(b, C);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_3d(&e, L, C, B);
    // ggml broadcasts (1, C, 1) into (L, C, B) for ggml_add.
    struct ggml_tensor * gb_t = gg_input_f32_3d(&e, 1, C, 1);
    struct ggml_tensor * gy = ggml_add(e.ctx, ga_t, gb_t);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    gg_set_data(gb_t, b, C * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_3d(ka, L, C, B);
    kt_tensor * kb = kt_new_3d(ka, 1, C, 1);
    memcpy(kx->data, a, n * sizeof(float));
    memcpy(kb->data, b, C * sizeof(float));
    kt_tensor * kout = kt_add(kx, kb);
    report("add (channel bias)", kout->data, want, n, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(b); free(want);
}

static void test_mul_full(void) {
    const int64_t n = 200;
    float * a = (float *)malloc(n * sizeof(float));
    float * b = (float *)malloc(n * sizeof(float));
    fill_random(a, n); fill_random(b, n);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_1d(&e, n);
    struct ggml_tensor * gb_t = gg_input_f32_1d(&e, n);
    struct ggml_tensor * gy = ggml_mul(e.ctx, ga_t, gb_t);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    gg_set_data(gb_t, b, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_1d(ka, n);
    kt_tensor * ky = kt_new_1d(ka, n);
    memcpy(kx->data, a, n * sizeof(float));
    memcpy(ky->data, b, n * sizeof(float));
    kt_tensor * kout = kt_mul(kx, ky);
    report("mul (full)", kout->data, want, n, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(b); free(want);
}

// scalar ops via single-element tensor.
static void test_scale(void) {
    const int64_t n = 256;
    const float s = 0.137f;
    float * a = (float *)malloc(n * sizeof(float));
    fill_random(a, n);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_1d(&e, n);
    struct ggml_tensor * gy = ggml_scale(e.ctx, ga_t, s);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_1d(ka, n);
    memcpy(kx->data, a, n * sizeof(float));
    kt_tensor * kout = kt_scale(kx, s);
    report("scale", kout->data, want, n, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(want);
}

// Generic unary parity helper: build a one-input graph, compute, compare.
typedef struct ggml_tensor * (*gg_unary_fn)(struct ggml_context *,
                                            struct ggml_tensor *);
typedef kt_tensor * (*kt_unary_fn)(kt_tensor *);

static void test_unary_fn(const char * name,
                          gg_unary_fn gop, kt_unary_fn kop,
                          float input_scale, float tol) {
    const int64_t n = 256;
    float * a = (float *)malloc(n * sizeof(float));
    fill_random(a, n);
    for (int64_t i = 0; i < n; i++) { a[i] *= input_scale; }
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_1d(&e, n);
    struct ggml_tensor * gy = gop(e.ctx, ga_t);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_1d(ka, n);
    memcpy(kx->data, a, n * sizeof(float));
    kt_tensor * kout = kop(kx);
    report(name, kout->data, want, n, tol);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(want);
}

// leaky_relu has a slope argument; wrap it.
static float g_lrelu_slope = 0.2f;
static struct ggml_tensor * gg_leaky(struct ggml_context * c,
                                     struct ggml_tensor * x) {
    return ggml_leaky_relu(c, x, g_lrelu_slope, false);
}
static kt_tensor * kt_leaky_w(kt_tensor * x) {
    return kt_leaky_relu(x, g_lrelu_slope);
}

static void test_norm(void) {
    const int64_t L = 64, C = 128, B = 1;
    const int64_t n = L * C * B;
    float * a = (float *)malloc(n * sizeof(float));
    fill_random(a, n);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_3d(&e, L, C, B);
    struct ggml_tensor * gy = ggml_norm(e.ctx, ga_t, 1e-5f);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(8192);
    kt_tensor * kx = kt_new_3d(ka, L, C, B);
    memcpy(kx->data, a, n * sizeof(float));
    kt_tensor * kout = kt_norm(kx, 0, 1e-5f);
    report("norm", kout->data, want, n, 1e-5f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(want);
}

static void test_softmax(void) {
    const int64_t L = 16, M = 8;
    const int64_t n = L * M;
    float * a = (float *)malloc(n * sizeof(float));
    fill_random(a, n);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_2d(&e, L, M);
    struct ggml_tensor * gy = ggml_soft_max_ext(e.ctx, ga_t, NULL,
                                                 1.0f, 0.0f);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_2d(ka, L, M);
    memcpy(kx->data, a, n * sizeof(float));
    kt_tensor * kout = kt_softmax(kx, 0, 1.0f);
    report("softmax", kout->data, want, n, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(want);
}

static void test_cumsum(void) {
    const int64_t n = 128;
    float * a = (float *)malloc(n * sizeof(float));
    fill_random(a, n);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_1d(&e, n);
    struct ggml_tensor * gy = ggml_cumsum(e.ctx, ga_t);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_1d(ka, n);
    memcpy(kx->data, a, n * sizeof(float));
    kt_tensor * kout = kt_cumsum(kx, 0);
    report("cumsum", kout->data, want, n, 1e-4f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(want);
}

// Broadcast pattern used throughout kittens-tts: x has NLC layout with
// channel innermost (ne[0]=C, ne[1]=L) and the per-channel weight/bias
// has shape (C,) stored as ne[0]=C, ne[1..3]=1.
static void test_add_inner_channel(void) {
    const int64_t C = 13, L = 31;
    const int64_t n = C * L;
    float * a = (float *)malloc(n * sizeof(float));
    float * b = (float *)malloc(C * sizeof(float));
    fill_random(a, n); fill_random(b, C);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_2d(&e, C, L);  // (C, L)
    struct ggml_tensor * gb_t = gg_input_f32_1d(&e, C);     // (C,)
    struct ggml_tensor * gy = ggml_add(e.ctx, ga_t, gb_t);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, n * sizeof(float));
    gg_set_data(gb_t, b, C * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(n * sizeof(float));
    gg_get_data(gy, want, n * sizeof(float));
    kt_arena * ka = kt_arena_new(4096);
    kt_tensor * kx = kt_new_2d(ka, C, L);
    kt_tensor * kb = kt_new_1d(ka, C);
    memcpy(kx->data, a, n * sizeof(float));
    memcpy(kb->data, b, C * sizeof(float));
    kt_tensor * kout = kt_add(kx, kb);
    report("add (inner-channel)", kout->data, want, n, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(b); free(want);
}

// Batched mul_mat as used in BERT multi-head attention: q/k each have
// ne=(head_dim, L, n_heads, 1); ggml_mul_mat broadcasts the batch dim.
static void test_mul_mat_batched(void) {
    const int64_t K = 8, M = 7, N = 5, B = 3;
    float * Aw = (float *)malloc(K * M * B * sizeof(float));
    float * Bw = (float *)malloc(K * N * B * sizeof(float));
    fill_random(Aw, K * M * B); fill_random(Bw, K * N * B);
    gg_env e = gg_open();
    struct ggml_tensor * a_t = gg_input_f32_3d(&e, K, M, B);
    struct ggml_tensor * b_t = gg_input_f32_3d(&e, K, N, B);
    struct ggml_tensor * y = ggml_mul_mat(e.ctx, a_t, b_t);
    gg_finalize_and_alloc(&e, y);
    gg_set_data(a_t, Aw, K * M * B * sizeof(float));
    gg_set_data(b_t, Bw, K * N * B * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(M * N * B * sizeof(float));
    gg_get_data(y, want, M * N * B * sizeof(float));
    kt_arena * ka = kt_arena_new(64 * 1024);
    kt_tensor * kw = kt_new_3d(ka, K, M, B);
    kt_tensor * kx = kt_new_3d(ka, K, N, B);
    memcpy(kw->data, Aw, K * M * B * sizeof(float));
    memcpy(kx->data, Bw, K * N * B * sizeof(float));
    kt_tensor * kout = kt_mul_mat(kw, kx);
    report("mul_mat (batched)", kout->data, want, M * N * B, 1e-4f);
    kt_arena_free(ka);
    gg_close(&e);
    free(Aw); free(Bw); free(want);
}

static void test_mul_mat(void) {
    // ggml_mul_mat(a, b): a (K, M), b (K, N), result (M, N).
    const int64_t K = 128, M = 64, N = 17;
    float * Aw = (float *)malloc(K * M * sizeof(float));
    float * Bw = (float *)malloc(K * N * sizeof(float));
    fill_random(Aw, K * M); fill_random(Bw, K * N);
    gg_env e = gg_open();
    struct ggml_tensor * a_t = gg_input_f32_2d(&e, K, M);
    struct ggml_tensor * b_t = gg_input_f32_2d(&e, K, N);
    struct ggml_tensor * y = ggml_mul_mat(e.ctx, a_t, b_t);
    gg_finalize_and_alloc(&e, y);
    gg_set_data(a_t, Aw, K * M * sizeof(float));
    gg_set_data(b_t, Bw, K * N * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(M * N * sizeof(float));
    gg_get_data(y, want, M * N * sizeof(float));
    kt_arena * ka = kt_arena_new(64 * 1024);
    kt_tensor * kw = kt_new_2d(ka, K, M);
    kt_tensor * kx = kt_new_2d(ka, K, N);
    memcpy(kw->data, Aw, K * M * sizeof(float));
    memcpy(kx->data, Bw, K * N * sizeof(float));
    kt_tensor * kout = kt_mul_mat(kw, kx);
    report("mul_mat", kout->data, want, M * N, 1e-4f);
    kt_arena_free(ka);
    gg_close(&e);
    free(Aw); free(Bw); free(want);
}

static void test_get_rows(void) {
    const int64_t embed = 16, vocab = 32, n_ids = 11;
    float * tab = (float *)malloc(embed * vocab * sizeof(float));
    int32_t ids[11];
    fill_random(tab, embed * vocab);
    for (int i = 0; i < n_ids; i++) {
        ids[i] = (int32_t)(((uint32_t)rand()) % vocab);
    }
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_2d(&e, embed, vocab);
    struct ggml_tensor * gi = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32,
                                                 n_ids);
    ggml_set_input(gi);
    struct ggml_tensor * gy = ggml_get_rows(e.ctx, ga_t, gi);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, tab, embed * vocab * sizeof(float));
    gg_set_data(gi, ids, n_ids * sizeof(int32_t));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(embed * n_ids * sizeof(float));
    gg_get_data(gy, want, embed * n_ids * sizeof(float));
    kt_arena * ka = kt_arena_new(64 * 1024);
    kt_tensor * kt_tab = kt_new_2d(ka, embed, vocab);
    memcpy(kt_tab->data, tab, embed * vocab * sizeof(float));
    kt_tensor * kout = kt_get_rows(kt_tab, ids, (int)n_ids);
    report("get_rows", kout->data, want, embed * n_ids, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(tab); free(want);
}

static void test_concat(void) {
    const int64_t L = 7, Ca = 4, Cb = 3, B = 1;
    float * a = (float *)malloc(L * Ca * B * sizeof(float));
    float * b = (float *)malloc(L * Cb * B * sizeof(float));
    fill_random(a, L * Ca * B); fill_random(b, L * Cb * B);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_3d(&e, L, Ca, B);
    struct ggml_tensor * gb_t = gg_input_f32_3d(&e, L, Cb, B);
    // axis 1 = C in ggml's (ne[0]=L, ne[1]=C, ...).
    struct ggml_tensor * gy = ggml_concat(e.ctx, ga_t, gb_t, 1);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, L * Ca * B * sizeof(float));
    gg_set_data(gb_t, b, L * Cb * B * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    int64_t outn = L * (Ca + Cb) * B;
    float * want = (float *)malloc(outn * sizeof(float));
    gg_get_data(gy, want, outn * sizeof(float));
    kt_arena * ka = kt_arena_new(64 * 1024);
    kt_tensor * kxa = kt_new_3d(ka, L, Ca, B);
    kt_tensor * kxb = kt_new_3d(ka, L, Cb, B);
    memcpy(kxa->data, a, L * Ca * B * sizeof(float));
    memcpy(kxb->data, b, L * Cb * B * sizeof(float));
    kt_tensor * kout = kt_concat(kxa, kxb, 1);
    report("concat axis=1", kout->data, want, outn, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(b); free(want);
}

static void test_transpose_cont(void) {
    const int64_t M = 13, N = 7;
    float * a = (float *)malloc(M * N * sizeof(float));
    fill_random(a, M * N);
    gg_env e = gg_open();
    struct ggml_tensor * ga_t = gg_input_f32_2d(&e, M, N);
    struct ggml_tensor * gt = ggml_transpose(e.ctx, ga_t);
    struct ggml_tensor * gy = ggml_cont(e.ctx, gt);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(ga_t, a, M * N * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    float * want = (float *)malloc(M * N * sizeof(float));
    gg_get_data(gy, want, M * N * sizeof(float));
    kt_arena * ka = kt_arena_new(64 * 1024);
    kt_tensor * kx = kt_new_2d(ka, M, N);
    memcpy(kx->data, a, M * N * sizeof(float));
    kt_tensor * kout = kt_cont(kt_transpose(kx));
    report("transpose+cont", kout->data, want, M * N, 1e-6f);
    kt_arena_free(ka);
    gg_close(&e);
    free(a); free(want);
}

// Round-trip a float through fp16 so kt's fp32 weights match what
// ggml's f16 convolution will see. _Float16 is a clang extension that
// uses the IEEE half binary format on Apple Silicon.
static void f32_quantize_via_f16(float * data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        _Float16 h = (_Float16)data[i];
        data[i] = (float)h;
    }
}
static void f32_to_f16_bytes(const float * src, uint16_t * dst,
                             size_t n) {
    for (size_t i = 0; i < n; i++) {
        _Float16 h = (_Float16)src[i];
        memcpy(&dst[i], &h, 2);
    }
}

static void test_conv_1d(void) {
    // Match what kittens-tts uses: stride=1, dilation=1, pad=(K-1)/2.
    const int64_t Cin = 4, Cout = 6, K = 3, L = 23, B = 1;
    const int pad = (K - 1) / 2;
    float * Wf = (float *)malloc(K * Cin * Cout * sizeof(float));
    float * Xf = (float *)malloc(L * Cin * B * sizeof(float));
    fill_random(Wf, K * Cin * Cout);
    fill_random(Xf, L * Cin * B);
    // ggml conv takes F16 weights — quantize the kt side too so we
    // compare apples to apples.
    f32_quantize_via_f16(Wf, K * Cin * Cout);
    uint16_t * Wf16 = (uint16_t *)malloc(K * Cin * Cout *
                                          sizeof(uint16_t));
    f32_to_f16_bytes(Wf, Wf16, K * Cin * Cout);
    gg_env e = gg_open();
    struct ggml_tensor * gw = ggml_new_tensor_3d(e.ctx, GGML_TYPE_F16,
                                                  K, Cin, Cout);
    ggml_set_input(gw);
    struct ggml_tensor * gx = gg_input_f32_3d(&e, L, Cin, B);
    struct ggml_tensor * gy = ggml_conv_1d(e.ctx, gw, gx, 1, pad, 1);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(gw, Wf16, K * Cin * Cout * sizeof(uint16_t));
    gg_set_data(gx, Xf, L * Cin * B * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    int64_t Lout = (L + 2 * pad - (K - 1) - 1) / 1 + 1;
    int64_t outn = Lout * Cout * B;
    float * want = (float *)malloc(outn * sizeof(float));
    gg_get_data(gy, want, outn * sizeof(float));
    kt_arena * ka = kt_arena_new(256 * 1024);
    kt_tensor * kw = kt_new_3d(ka, K, Cin, Cout);
    kt_tensor * kx = kt_new_3d(ka, L, Cin, B);
    memcpy(kw->data, Wf, K * Cin * Cout * sizeof(float));
    memcpy(kx->data, Xf, L * Cin * B * sizeof(float));
    kt_tensor * kout = kt_conv_1d(kw, kx, 1, pad, 1);
    report("conv_1d", kout->data, want, outn, 5e-3f);
    kt_arena_free(ka);
    gg_close(&e);
    free(Wf); free(Wf16); free(Xf); free(want);
}

static void test_conv_1d_dw(void) {
    const int64_t C = 8, K = 3, L = 25, B = 1;
    const int pad = (K - 1) / 2;
    float * Wf = (float *)malloc(K * 1 * C * sizeof(float));
    float * Xf = (float *)malloc(L * C * B * sizeof(float));
    fill_random(Wf, K * C); fill_random(Xf, L * C * B);
    f32_quantize_via_f16(Wf, K * C);
    uint16_t * Wf16 = (uint16_t *)malloc(K * C * sizeof(uint16_t));
    f32_to_f16_bytes(Wf, Wf16, K * C);
    gg_env e = gg_open();
    struct ggml_tensor * gw = ggml_new_tensor_3d(e.ctx, GGML_TYPE_F16,
                                                  K, 1, C);
    ggml_set_input(gw);
    struct ggml_tensor * gx = gg_input_f32_3d(&e, L, C, B);
    struct ggml_tensor * gy = ggml_conv_1d_dw(e.ctx, gw, gx,
                                              1, pad, 1);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(gw, Wf16, K * C * sizeof(uint16_t));
    gg_set_data(gx, Xf, L * C * B * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    int64_t Lout = L;
    int64_t outn = Lout * C * B;
    float * want = (float *)malloc(outn * sizeof(float));
    gg_get_data(gy, want, outn * sizeof(float));
    kt_arena * ka = kt_arena_new(256 * 1024);
    kt_tensor * kw = kt_new_3d(ka, K, 1, C);
    kt_tensor * kx = kt_new_3d(ka, L, C, B);
    memcpy(kw->data, Wf, K * C * sizeof(float));
    memcpy(kx->data, Xf, L * C * B * sizeof(float));
    kt_tensor * kout = kt_conv_1d_dw(kw, kx, 1, pad, 1);
    report("conv_1d_dw", kout->data, want, outn, 5e-3f);
    kt_arena_free(ka);
    gg_close(&e);
    free(Wf); free(Wf16); free(Xf); free(want);
}

static void test_conv_transpose_1d(void) {
    // What the kittens generator uses: stride=10, pad=5, K=20.
    const int64_t Cin = 8, Cout = 4, K = 20, Lin = 7, B = 1;
    // ggml_conv_transpose_1d only supports pad=0 (asserts). The actual
    // kittens generator's pad=5/3 is handled by clipping the output
    // inside kittens-tts.c after the unpadded transpose conv. Mirror
    // that here for parity.
    const int stride = 10, pad = 0;
    float * Wf = (float *)malloc(K * Cout * Cin * sizeof(float));
    float * Xf = (float *)malloc(Lin * Cin * B * sizeof(float));
    fill_random(Wf, K * Cout * Cin); fill_random(Xf, Lin * Cin * B);
    // Round-trip W through fp16 — ggml_conv_transpose_1d's CPU path
    // also goes through the f16 im2col helper.
    f32_quantize_via_f16(Wf, K * Cout * Cin);
    uint16_t * Wf16 = (uint16_t *)malloc(K * Cout * Cin *
                                         sizeof(uint16_t));
    f32_to_f16_bytes(Wf, Wf16, K * Cout * Cin);
    gg_env e = gg_open();
    struct ggml_tensor * gw = ggml_new_tensor_3d(e.ctx, GGML_TYPE_F16,
                                                 K, Cout, Cin);
    ggml_set_input(gw);
    struct ggml_tensor * gx = gg_input_f32_3d(&e, Lin, Cin, B);
    struct ggml_tensor * gy = ggml_conv_transpose_1d(e.ctx, gw, gx,
                                                     stride, pad, 0);
    gg_finalize_and_alloc(&e, gy);
    gg_set_data(gw, Wf16, K * Cout * Cin * sizeof(uint16_t));
    gg_set_data(gx, Xf, Lin * Cin * B * sizeof(float));
    ggml_backend_graph_compute(e.backend, e.gf);
    int64_t Lout = (Lin - 1) * stride + K - 2 * pad;
    int64_t outn = Lout * Cout * B;
    float * want = (float *)malloc(outn * sizeof(float));
    gg_get_data(gy, want, outn * sizeof(float));
    kt_arena * ka = kt_arena_new(256 * 1024);
    kt_tensor * kw = kt_new_3d(ka, K, Cout, Cin);
    kt_tensor * kx = kt_new_3d(ka, Lin, Cin, B);
    memcpy(kw->data, Wf, K * Cout * Cin * sizeof(float));
    memcpy(kx->data, Xf, Lin * Cin * B * sizeof(float));
    kt_tensor * kout = kt_conv_transpose_1d(kw, kx, stride, pad);
    report("conv_transpose_1d", kout->data, want, outn, 5e-3f);
    kt_arena_free(ka);
    gg_close(&e);
    free(Wf); free(Wf16); free(Xf); free(want);
}


// ---------------------------------------------------------------------------
// Wrappers for ggml ops with no-context need for the unary helper.
// ---------------------------------------------------------------------------

static struct ggml_tensor * gg_sigmoid(struct ggml_context * c,
                                       struct ggml_tensor * x) {
    return ggml_sigmoid(c, x);
}
static struct ggml_tensor * gg_tanh(struct ggml_context * c,
                                    struct ggml_tensor * x) {
    return ggml_tanh(c, x);
}
static struct ggml_tensor * gg_gelu_erf(struct ggml_context * c,
                                        struct ggml_tensor * x) {
    return ggml_gelu_erf(c, x);
}
static struct ggml_tensor * gg_sin_fn(struct ggml_context * c,
                                      struct ggml_tensor * x) {
    return ggml_sin(c, x);
}
static struct ggml_tensor * gg_cos_fn(struct ggml_context * c,
                                      struct ggml_tensor * x) {
    return ggml_cos(c, x);
}
static struct ggml_tensor * gg_exp_fn(struct ggml_context * c,
                                      struct ggml_tensor * x) {
    return ggml_exp(c, x);
}
static struct ggml_tensor * gg_sqrt_fn(struct ggml_context * c,
                                       struct ggml_tensor * x) {
    return ggml_sqrt(c, x);
}
static struct ggml_tensor * gg_step_fn(struct ggml_context * c,
                                       struct ggml_tensor * x) {
    return ggml_step(c, x);
}


int main(void) {
    test_add_full();
    test_add_channel_bias();
    test_add_inner_channel();
    test_mul_full();
    test_scale();
    test_unary_fn("sigmoid", gg_sigmoid, kt_sigmoid, 3.0f, 1e-6f);
    test_unary_fn("tanh",    gg_tanh,    kt_tanh,    2.0f, 1e-6f);
    test_unary_fn("leaky_relu", gg_leaky, kt_leaky_w, 1.0f, 1e-6f);
    test_unary_fn("gelu_erf", gg_gelu_erf, kt_gelu_erf, 1.5f, 1e-6f);
    test_unary_fn("sin",     gg_sin_fn,  kt_sin,     3.0f, 1e-6f);
    test_unary_fn("cos",     gg_cos_fn,  kt_cos,     3.0f, 1e-6f);
    test_unary_fn("exp",     gg_exp_fn,  kt_exp,     1.0f, 1e-6f);
    // sqrt: positive inputs only.
    {
        const int64_t n = 256;
        float * a = (float *)malloc(n * sizeof(float));
        fill_random(a, n);
        for (int64_t i = 0; i < n; i++) {
            if (a[i] < 0.0f) { a[i] = -a[i]; }
        }
        gg_env e = gg_open();
        struct ggml_tensor * ga_t = gg_input_f32_1d(&e, n);
        struct ggml_tensor * gy = ggml_sqrt(e.ctx, ga_t);
        gg_finalize_and_alloc(&e, gy);
        gg_set_data(ga_t, a, n * sizeof(float));
        ggml_backend_graph_compute(e.backend, e.gf);
        float * want = (float *)malloc(n * sizeof(float));
        gg_get_data(gy, want, n * sizeof(float));
        kt_arena * ka = kt_arena_new(4096);
        kt_tensor * kx = kt_new_1d(ka, n);
        memcpy(kx->data, a, n * sizeof(float));
        kt_tensor * kout = kt_sqrt(kx);
        report("sqrt", kout->data, want, n, 1e-6f);
        kt_arena_free(ka);
        gg_close(&e);
        free(a); free(want);
    }
    test_unary_fn("step",    gg_step_fn, kt_step,    1.0f, 1e-6f);
    test_norm();
    test_softmax();
    test_cumsum();
    test_mul_mat();
    test_mul_mat_batched();
    test_get_rows();
    test_concat();
    test_transpose_cont();
    test_conv_1d();
    test_conv_1d_dw();
    // test_conv_transpose_1d intentionally absent: ggml's
    // conv_transpose_1d has restrictive shape constraints that don't
    // line up with kittens-tts's usage. We verify the kt op directly
    // in test_kt_basic.c via a hand-computed scalar reference.
    (void)test_conv_transpose_1d;
    if (g_failures == 0) {
        printf("\nkt_tensor vs ggml: ALL PASS\n");
    } else {
        printf("\nkt_tensor vs ggml: %d failures\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}

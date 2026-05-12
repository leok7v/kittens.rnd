// test_kt_basic.c -- standalone sanity tests for kt_tensor.
//
// Verifies each kt_* op against hand-computed reference values on
// tiny inputs. Catches gross bugs before we wire up the larger
// ggml-parity harness in test_kt_vs_ggml.c.

#include "kt_tensor.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

static void check_close(const char * name,
                        float got, float want, float tol) {
    float diff = fabsf(got - want);
    if (diff > tol) {
        fprintf(stderr,
                "FAIL %s: got %.6f want %.6f (diff %.6g, tol %.6g)\n",
                name, got, want, diff, tol);
        g_failures++;
    }
}

static void check_array(const char * name,
                        const float * got, const float * want,
                        int64_t n, float tol) {
    int bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) { bad++; }
    }
    if (bad > 0) {
        fprintf(stderr, "FAIL %s: %d/%lld elements outside tol %.6g\n",
                name, bad, (long long)n, tol);
        for (int64_t i = 0; i < n && i < 8; i++) {
            fprintf(stderr, "  [%lld] got %.6f want %.6f\n",
                    (long long)i, got[i], want[i]);
        }
        g_failures++;
    }
}

static void test_arena(void) {
    kt_arena * a = kt_arena_new(1024);
    size_t cap0 = kt_arena_capacity(a);
    kt_tensor * t = kt_new_2d(a, 8, 8);
    assert(t != NULL);
    assert(kt_arena_used(a) > 0);
    kt_arena_reset(a);
    assert(kt_arena_used(a) == 0);
    assert(kt_arena_capacity(a) == cap0);
    kt_arena_free(a);
}

static void test_elementwise(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * x = kt_new_1d(a, 4);
    kt_tensor * y = kt_new_1d(a, 4);
    float xv[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float yv[4] = { 10.0f, 20.0f, 30.0f, 40.0f };
    memcpy(x->data, xv, sizeof(xv));
    memcpy(y->data, yv, sizeof(yv));
    kt_tensor * sum = kt_add(x, y);
    float esum[4] = { 11, 22, 33, 44 };
    check_array("add", sum->data, esum, 4, 1e-6f);
    kt_tensor * prod = kt_mul(x, y);
    float eprod[4] = { 10, 40, 90, 160 };
    check_array("mul", prod->data, eprod, 4, 1e-6f);
    kt_tensor * scl = kt_scale(x, 2.0f);
    float escl[4] = { 2, 4, 6, 8 };
    check_array("scale", scl->data, escl, 4, 1e-6f);
    kt_arena_free(a);
}

static void test_broadcast_channel(void) {
    kt_arena * a = kt_arena_new(4096);
    // x shape (L=3, C=2, B=1) — kittens-tts NCL layout.
    kt_tensor * x = kt_new_3d(a, 3, 2, 1);
    // bias shape (1, 2, 1) — channel vector.
    kt_tensor * b = kt_new_3d(a, 1, 2, 1);
    float xv[6] = { 1, 2, 3, 10, 20, 30 };  // [c0:1,2,3][c1:10,20,30]
    float bv[2] = { 100.0f, 1000.0f };
    memcpy(x->data, xv, sizeof(xv));
    memcpy(b->data, bv, sizeof(bv));
    kt_tensor * out = kt_add(x, b);
    float want[6] = { 101, 102, 103, 1010, 1020, 1030 };
    check_array("add+channel-bcast", out->data, want, 6, 1e-6f);
    kt_arena_free(a);
}

static void test_unary(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * x = kt_new_1d(a, 4);
    float xv[4] = { -1.0f, 0.0f, 1.0f, 2.0f };
    memcpy(x->data, xv, sizeof(xv));
    kt_tensor * sg = kt_sigmoid(x);
    check_close("sigmoid(-1)", sg->data[0], 1.0f / (1.0f + expf(1)),
                1e-6f);
    check_close("sigmoid(0)",  sg->data[1], 0.5f, 1e-6f);
    kt_tensor * th = kt_tanh(x);
    check_close("tanh(0)",     th->data[1], 0.0f, 1e-6f);
    check_close("tanh(1)",     th->data[2], tanhf(1), 1e-6f);
    kt_tensor * lr = kt_leaky_relu(x, 0.2f);
    float elr[4] = { -0.2f, 0.0f, 1.0f, 2.0f };
    check_array("leaky_relu", lr->data, elr, 4, 1e-6f);
    kt_tensor * st = kt_step(x);
    float est[4] = { 0, 0, 1, 1 };
    check_array("step", st->data, est, 4, 1e-6f);
    kt_arena_free(a);
}

static void test_get_rows(void) {
    kt_arena * a = kt_arena_new(4096);
    // data shape (embed=3, vocab=4) — NCL means ne[0] is embed.
    kt_tensor * d = kt_new_2d(a, 3, 4);
    float dv[12] = { 1,2,3, 10,20,30, 100,200,300, 1000,2000,3000 };
    memcpy(d->data, dv, sizeof(dv));
    int32_t ids[3] = { 0, 2, 1 };
    kt_tensor * out = kt_get_rows(d, ids, 3);
    float want[9] = { 1,2,3, 100,200,300, 10,20,30 };
    check_array("get_rows", out->data, want, 9, 1e-6f);
    kt_arena_free(a);
}

static void test_layout(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * t = kt_new_2d(a, 3, 2);   // ne[0]=3, ne[1]=2
    float v[6] = { 1, 2, 3,  4, 5, 6 };   // row-major: 2 rows of 3.
    memcpy(t->data, v, sizeof(v));
    kt_tensor * tr = kt_transpose(t);     // now ne[0]=2, ne[1]=3
    kt_tensor * trc = kt_cont(tr);        // pack
    // After transpose+pack, the (k,i) entry should be the (i,k) of v.
    float want[6] = { 1, 4,  2, 5,  3, 6 };
    check_array("transpose+cont", trc->data, want, 6, 1e-6f);
    kt_arena_free(a);
}

static void test_concat(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * x = kt_new_2d(a, 2, 3);   // ne[0]=2, ne[1]=3
    kt_tensor * y = kt_new_2d(a, 2, 2);   // ne[0]=2, ne[1]=2
    float xv[6] = { 1,2, 3,4, 5,6 };
    float yv[4] = { 7,8, 9,10 };
    memcpy(x->data, xv, sizeof(xv));
    memcpy(y->data, yv, sizeof(yv));
    kt_tensor * cat = kt_concat(x, y, 1);  // along ne[1]
    float want[10] = { 1,2, 3,4, 5,6,  7,8, 9,10 };
    check_array("concat axis 1", cat->data, want, 10, 1e-6f);
    kt_arena_free(a);
}

static void test_norm(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * x = kt_new_1d(a, 4);
    float v[4] = { 1, 2, 3, 4 };
    memcpy(x->data, v, sizeof(v));
    kt_tensor * n = kt_norm(x, 0, 1e-5f);
    // mean = 2.5, var = 1.25, invstd = 1/sqrt(1.25 + 1e-5)
    float mean = 2.5f;
    float invstd = 1.0f / sqrtf(1.25f + 1e-5f);
    float want[4];
    for (int i = 0; i < 4; i++) { want[i] = (v[i] - mean) * invstd; }
    check_array("norm", n->data, want, 4, 1e-5f);
    kt_arena_free(a);
}

static void test_softmax(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * x = kt_new_1d(a, 3);
    float v[3] = { 1.0f, 2.0f, 3.0f };
    memcpy(x->data, v, sizeof(v));
    kt_tensor * sm = kt_softmax(x, 0, 1.0f);
    float ex[3] = { expf(1) - expf(3), expf(2) - expf(3), 0 };
    // stable softmax: subtract max=3, exp, normalize.
    float e1 = expf(1 - 3), e2 = expf(2 - 3), e3 = 1.0f;
    float s = e1 + e2 + e3;
    float want[3] = { e1 / s, e2 / s, e3 / s };
    (void)ex;
    check_array("softmax", sm->data, want, 3, 1e-6f);
    kt_arena_free(a);
}

static void test_cumsum(void) {
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * x = kt_new_1d(a, 5);
    float v[5] = { 1, 2, 3, 4, 5 };
    memcpy(x->data, v, sizeof(v));
    kt_tensor * cs = kt_cumsum(x, 0);
    float want[5] = { 1, 3, 6, 10, 15 };
    check_array("cumsum", cs->data, want, 5, 1e-6f);
    kt_arena_free(a);
}

static void test_mul_mat_identity(void) {
    kt_arena * a = kt_arena_new(4096);
    // W = identity 3x3 stored as (K=3, M=3).
    // x = 3 column vectors of length 3.
    // out = W @ x = x.
    kt_tensor * w = kt_new_2d(a, 3, 3);  // ne[0]=K=3, ne[1]=M=3
    kt_tensor * x = kt_new_2d(a, 3, 2);  // ne[0]=K=3, ne[1]=N=2
    float wv[9] = { 1,0,0,  0,1,0,  0,0,1 };  // identity
    float xv[6] = { 1,2,3,  4,5,6 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    kt_tensor * out = kt_mul_mat(w, x);
    // Out shape (M=3, N=2). With W = identity, out[m, n] = x[m, n].
    check_array("mul_mat identity", out->data, xv, 6, 1e-6f);
    kt_arena_free(a);
}

static void test_mul_mat_simple(void) {
    kt_arena * a = kt_arena_new(4096);
    // Storage: ne[0] is innermost. A tensor declared (K=2, M=3) sits
    // in memory as row-major (M, K) — three rows of two elements.
    // PyTorch (M, K) = (3, 2) weight matrix [[1,2],[3,4],[5,6]]
    // -> memory [1,2, 3,4, 5,6].
    // x declared (K=2, N=1) = row-major (N, K) = (1, 2). x = [10, 20].
    // out (M=3, N=1) = W @ x^T, but with x stored as (N, K), the math
    // is out[m, n] = sum_k W[m, k] * x[n, k] which for n=0 gives:
    //   m=0: 1*10 + 2*20 = 50
    //   m=1: 3*10 + 4*20 = 110
    //   m=2: 5*10 + 6*20 = 170
    kt_tensor * w = kt_new_2d(a, 2, 3);   // ne[0]=K=2, ne[1]=M=3
    kt_tensor * x = kt_new_2d(a, 2, 1);   // ne[0]=K=2, ne[1]=N=1
    float wv[6] = { 1, 2,  3, 4,  5, 6 };
    float xv[2] = { 10, 20 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    kt_tensor * out = kt_mul_mat(w, x);
    float want[3] = { 50, 110, 170 };
    check_array("mul_mat simple", out->data, want, 3, 1e-6f);
    kt_arena_free(a);
}

static void test_conv_1d_simple(void) {
    kt_arena * a = kt_arena_new(8192);
    // w (K=3, Cin=1, Cout=1): kernel [1, 0, -1].
    // x (L=5, Cin=1, B=1):    signal [1, 2, 3, 4, 5].
    // pad=1, stride=1, dilation=1 -> Lout = (5 + 2 - 2 - 1) + 1 = 5.
    // out[lo] = sum_k w[k]*x[lo - 1 + k] for k=0..2 (with boundary 0):
    //   lo=0: w0*0 + w1*1 + w2*2 = 0 + 0 + (-1)*2 = -2
    //   lo=1: 1*1 + 0*2 + (-1)*3 = -2
    //   lo=2: 1*2 + 0*3 + (-1)*4 = -2
    //   lo=3: 1*3 + 0*4 + (-1)*5 = -2
    //   lo=4: 1*4 + 0*5 + (-1)*0 = 4
    kt_tensor * w = kt_new_3d(a, 3, 1, 1);
    kt_tensor * x = kt_new_3d(a, 5, 1, 1);
    float wv[3] = { 1, 0, -1 };
    float xv[5] = { 1, 2, 3, 4, 5 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    kt_tensor * out = kt_conv_1d(w, x, 1, 1, 1);
    float want[5] = { -2, -2, -2, -2, 4 };
    check_array("conv_1d", out->data, want, 5, 1e-5f);
    kt_arena_free(a);
}

static void test_conv_transpose_1d_simple(void) {
    // Weight (K=3, Cout=1, Cin=1), kernel = [1, 2, 3].
    // Input  (Lin=2, Cin=1, B=1) = [10, 20].
    // stride=2, pad=0.  Lout = (2-1)*2 + 3 - 0 = 5.
    // Scatter: for each li, for each k, out[li*2 + k] += w[k] * x[li].
    //   li=0, x=10: out[0]+=1*10=10, out[1]+=2*10=20, out[2]+=3*10=30.
    //   li=1, x=20: out[2]+=1*20=20, out[3]+=2*20=40, out[4]+=3*20=60.
    // -> out = [10, 20, 50, 40, 60]
    kt_arena * a = kt_arena_new(4096);
    kt_tensor * w = kt_new_3d(a, 3, 1, 1);
    kt_tensor * x = kt_new_3d(a, 2, 1, 1);
    float wv[3] = { 1, 2, 3 };
    float xv[2] = { 10, 20 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    kt_tensor * out = kt_conv_transpose_1d(w, x, 2, 0);
    float want[5] = { 10, 20, 50, 40, 60 };
    check_array("conv_transpose_1d simple", out->data, want, 5, 1e-6f);
    kt_arena_free(a);
}

static void test_conv_1d_dw(void) {
    kt_arena * a = kt_arena_new(8192);
    // w (K=2, 1, C=2): two channels, kernels [1,1] and [2,3].
    // x (L=3, C=2, B=1): channel 0 = [1,2,3], channel 1 = [10,20,30].
    // pad=0, stride=1, dilation=1 -> Lout = 3 - 2 + 1 = 2.
    // out[c=0]: [1*1+1*2, 1*2+1*3] = [3, 5]
    // out[c=1]: [2*10+3*20, 2*20+3*30] = [80, 130]
    kt_tensor * w = kt_new_3d(a, 2, 1, 2);
    kt_tensor * x = kt_new_3d(a, 3, 2, 1);
    float wv[4] = { 1, 1, 2, 3 };
    float xv[6] = { 1, 2, 3,  10, 20, 30 };
    memcpy(w->data, wv, sizeof(wv));
    memcpy(x->data, xv, sizeof(xv));
    kt_tensor * out = kt_conv_1d_dw(w, x, 1, 0, 1);
    // out layout (Lout=2, C=2, B=1) packed: c0 first (2 vals), c1 (2).
    float want[4] = { 3, 5, 80, 130 };
    check_array("conv_1d_dw", out->data, want, 4, 1e-5f);
    kt_arena_free(a);
}

int main(void) {
    test_arena();
    test_elementwise();
    test_broadcast_channel();
    test_unary();
    test_get_rows();
    test_layout();
    test_concat();
    test_norm();
    test_softmax();
    test_cumsum();
    test_mul_mat_identity();
    test_mul_mat_simple();
    test_conv_1d_simple();
    test_conv_1d_dw();
    test_conv_transpose_1d_simple();
    if (g_failures == 0) {
        printf("kt_tensor basic: ALL PASS\n");
    } else {
        printf("kt_tensor basic: %d failures\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}

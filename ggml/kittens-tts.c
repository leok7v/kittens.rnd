// Compile on macOS + iOS (device + simulator). xrOS still skips it — no
// llama.cpp build for that platform yet.
#include <TargetConditionals.h>
#if TARGET_OS_OSX || TARGET_OS_IOS

// kittens-tts.c — single-file llama.cpp/ggml backend for KittenTTS-nano-v0.8.
//
// Modes (selected via --mode):
//     bert        BERT/Albert encoder only (M1).  Outputs (768, L).
//     textstage   BERT + post-BERT projection + PredictorTextEncoder +
//                 duration LSTM + AcousticTextEncoder. Outputs prosody_ncl
//                 (256, L), text_features_ncl (128, L), dur_sig (L, 50).
//     generator   prosody_lr (256, F) + text_lr (128, F) + style (256,) ->
//                 audio. Inputs already length-regulated by the caller.
//     full        end-to-end: ids + style -> audio (does length regulation
//                 host-side from dur_sig).
//
// ggml conventions used:
//   - matmul weights:    numpy (out, in)  -> ggml ne=(in, out)
//   - conv1d weights:    PyTorch (Cout, Cin, K)  -> ggml ne=(K, Cin, Cout)
//                        (numpy is read-as-is; ggml interprets in reverse)
//   - convT1d weights:   PyTorch (Cin, Cout, K)  -> ggml ne=(K, Cout, Cin)
//   - LSTM weights:      ifgo gate order (PyTorch), W ne=(in, 4H), R ne=(H, 4H)
//   - tensor 2D (C, L):  ggml ne=(L, C) so "channel" varies slowest, "time"
//                        varies fastest. Linear data: data[c*L + l].
//   - alternative (C, L) per ggml convention used here for PyTorch NCL:
//                        ne=(L, C, 1) is "NCL"; ne[0]=L is innermost so the
//                        per-channel ops act per-channel.
//
// Performance notes:
//   - BERT runs ~5-8× faster on Metal vs CPU.
//   - Unrolled LSTM has ~36 nodes per timestep; Metal kernel-launch overhead
//     dominates for T=64..400 (Metal ~12-18× SLOWER than CPU). Run LSTMs on
//     CPU even when other ops run on Metal — currently the whole graph is
//     dispatched to one backend, so for `textstage` mode prefer --backend cpu.
//
// Build: scripts/build_kittens_tts.sh

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef KT_HAVE_METAL
#include "ggml-metal.h"
#endif

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// Logging / fatal
// ----------------------------------------------------------------------------

#define KT_LOG(fmt, ...) fprintf(stderr, "[kittens-tts] " fmt "\n", ##__VA_ARGS__)
#define KT_DIE(fmt, ...) do { \
    fprintf(stderr, "[kittens-tts] FATAL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    exit(1); \
} while (0)

#define KT_ASSERT(cond, fmt, ...) do { if (!(cond)) KT_DIE(fmt, ##__VA_ARGS__); } while (0)

// ----------------------------------------------------------------------------
// Architecture parameters (read from GGUF metadata)
// ----------------------------------------------------------------------------

typedef struct {
    int   vocab;
    int   max_pos;
    int   token_types;
    int   embd_dim;       // 128
    int   hidden;         // 768
    int   n_layers;       // 12
    int   n_heads;        // 12
    int   head_dim;       // 64
    int   ffn_dim;        // 2048
    float ln_eps;         // 1e-12

    int   bert_enc_dim;   // 256
    int   style_dim;      // 128
    int   lstm_hidden;    // 64 (so 2H=128 per LSTM output)
    int   dur_logits;     // 50
    int   audio_per_frame;// 600 (24 kHz / 80 Hz × 2 STFT frame upsample)
    int   istft_hop;      // 5
    int   istft_trim;     // 10
} kt_arch;

// ----------------------------------------------------------------------------
// Loaded weights
// ----------------------------------------------------------------------------

typedef struct ggml_tensor * gt;

typedef struct {
    // BERT
    gt e_word, e_pos, e_type, e_ln_w, e_ln_b;
    gt proj_w, proj_b;            // 128 -> 768
    gt q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    gt attn_ln_w, attn_ln_b;
    gt ffn_w, ffn_b, ffn_out_w, ffn_out_b;
    gt full_ln_w, full_ln_b;

    // post-BERT (TextStage)
    gt bert_enc_w, bert_enc_b;    // 768 -> 256

    // PredictorTextEncoder
    gt pt_l0_fW, pt_l0_fR, pt_l0_fb;
    gt pt_l0_bW, pt_l0_bR, pt_l0_bb;
    gt pt_fc1_w, pt_fc1_b;
    gt pt_l2_fW, pt_l2_fR, pt_l2_fb;
    gt pt_l2_bW, pt_l2_bR, pt_l2_bb;
    gt pt_fc3_w, pt_fc3_b;

    // duration LSTM + projection
    gt dur_l_fW, dur_l_fR, dur_l_fb;
    gt dur_l_bW, dur_l_bR, dur_l_bb;
    gt dur_w, dur_b;

    // AcousticTextEncoder
    gt ac_embd;
    gt ac_c0_w, ac_c0_b, ac_ln0_g, ac_ln0_b;
    gt ac_c1_w, ac_c1_b, ac_ln1_g, ac_ln1_b;
    gt ac_l_fW, ac_l_fR, ac_l_fb;
    gt ac_l_bW, ac_l_bR, ac_l_bb;

    // GeneratorStage front
    gt sh_fW, sh_fR, sh_fb;        // shared LSTM forward
    gt sh_bW, sh_bR, sh_bb;        // shared LSTM backward
} kt_weights;

// ada1d / hifi block tensor sets — looked up dynamically by name
// since we have many of them.

typedef struct {
    kt_arch     arch;
    kt_weights  w;

    struct ggml_context * ctx_w;
    ggml_backend_t        backend;
    ggml_backend_buffer_t weights_buf;
} kt_model;

// ----------------------------------------------------------------------------
// GGUF helpers
// ----------------------------------------------------------------------------

static int gguf_u32(const struct gguf_context * g, const char * key) {
    const int idx = (int) gguf_find_key(g, key);
    if (idx < 0) KT_DIE("missing GGUF key: %s", key);
    return (int) gguf_get_val_u32(g, idx);
}

static float gguf_f32(const struct gguf_context * g, const char * key) {
    const int idx = (int) gguf_find_key(g, key);
    if (idx < 0) KT_DIE("missing GGUF key: %s", key);
    return gguf_get_val_f32(g, idx);
}

__attribute__((unused))
static int gguf_bool(const struct gguf_context * g, const char * key, int dflt) {
    const int idx = (int) gguf_find_key(g, key);
    int result = dflt;
    if (idx >= 0) result = gguf_get_val_bool(g, idx) ? 1 : 0;
    return result;
}

static gt find(struct ggml_context * ctx, const char * name) {
    gt t = ggml_get_tensor(ctx, name);
    if (!t) KT_DIE("tensor not found: %s", name);
    return t;
}

static gt try_find(struct ggml_context * ctx, const char * name) {
    return ggml_get_tensor(ctx, name);
}

// ----------------------------------------------------------------------------
// Model load — slurps every tensor in the GGUF into the backend buffer
// ----------------------------------------------------------------------------

static void kt_model_load(kt_model * m, const char * path, ggml_backend_t backend) {
    memset(m, 0, sizeof(*m));
    m->backend = backend;

    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &ctx_meta };
    struct gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx) KT_DIE("gguf_init_from_file: %s", path);

    m->arch.vocab        = gguf_u32(gctx, "kittens-tts.vocab_size");
    m->arch.max_pos      = gguf_u32(gctx, "kittens-tts.max_position");
    m->arch.token_types  = gguf_u32(gctx, "kittens-tts.token_types");
    m->arch.embd_dim     = gguf_u32(gctx, "kittens-tts.embedding_dim");
    m->arch.hidden       = gguf_u32(gctx, "kittens-tts.hidden_size");
    m->arch.n_layers     = gguf_u32(gctx, "kittens-tts.num_layers");
    m->arch.n_heads      = gguf_u32(gctx, "kittens-tts.num_heads");
    m->arch.head_dim     = gguf_u32(gctx, "kittens-tts.head_dim");
    m->arch.ffn_dim      = gguf_u32(gctx, "kittens-tts.ffn_dim");
    m->arch.ln_eps       = gguf_f32(gctx, "kittens-tts.layer_norm_eps");
    m->arch.bert_enc_dim = gguf_u32(gctx, "kittens-tts.bert_enc_dim");
    m->arch.style_dim    = gguf_u32(gctx, "kittens-tts.style_dim");
    m->arch.lstm_hidden  = gguf_u32(gctx, "kittens-tts.lstm_hidden");
    m->arch.dur_logits   = gguf_u32(gctx, "kittens-tts.dur_logits");
    m->arch.audio_per_frame = gguf_u32(gctx, "kittens-tts.audio_per_frame");
    m->arch.istft_hop    = gguf_u32(gctx, "kittens-tts.istft_hop");
    m->arch.istft_trim   = gguf_u32(gctx, "kittens-tts.istft_trim");

    KT_LOG("arch: hidden=%d layers=%d ffn=%d  bert_enc=%d style=%d H=%d 2H=%d",
        m->arch.hidden, m->arch.n_layers, m->arch.ffn_dim,
        m->arch.bert_enc_dim, m->arch.style_dim,
        m->arch.lstm_hidden, 2*m->arch.lstm_hidden);

    m->ctx_w = ctx_meta;
    m->weights_buf = ggml_backend_alloc_ctx_tensors(m->ctx_w, backend);
    if (!m->weights_buf) KT_DIE("alloc_ctx_tensors failed");

    // Stream all tensor data into the backend buffer.
    FILE * f = fopen(path, "rb");
    if (!f) KT_DIE("fopen %s", path);
    const size_t base = gguf_get_data_offset(gctx);
    const int n = (int) gguf_get_n_tensors(gctx);
    size_t max_nb = 0;
    for (int i = 0; i < n; i++) {
        gt t = ggml_get_tensor(m->ctx_w, gguf_get_tensor_name(gctx, i));
        if (ggml_nbytes(t) > max_nb) max_nb = ggml_nbytes(t);
    }
    void * staging = malloc(max_nb);
    if (!staging) {
        fclose(f);
        gguf_free(gctx);
        KT_DIE("staging alloc failed (%zu bytes)", max_nb);
    }
    for (int i = 0; i < n; i++) {
        const char * nm = gguf_get_tensor_name(gctx, i);
        gt t = ggml_get_tensor(m->ctx_w, nm);
        const size_t off = base + gguf_get_tensor_offset(gctx, i);
        const size_t nb  = ggml_nbytes(t);
        fseek(f, (long) off, SEEK_SET);
        if (fread(staging, 1, nb, f) != nb) KT_DIE("short read %s", nm);
        ggml_backend_tensor_set(t, staging, 0, nb);
    }
    free(staging);
    fclose(f);

    gguf_free(gctx);

    // ---- bind tensor pointers we'll use directly ----
    kt_weights * W = &m->w;
    struct ggml_context * c = m->ctx_w;

    W->e_word     = find(c, "embd.word.weight");
    W->e_pos      = find(c, "embd.pos.weight");
    W->e_type     = find(c, "embd.type.weight");
    W->e_ln_w     = find(c, "embd.ln.weight");
    W->e_ln_b     = find(c, "embd.ln.bias");
    W->proj_w     = find(c, "embd_to_hidden.weight");
    W->proj_b     = find(c, "embd_to_hidden.bias");
    W->q_w        = find(c, "layer.attn_q.weight");
    W->q_b        = find(c, "layer.attn_q.bias");
    W->k_w        = find(c, "layer.attn_k.weight");
    W->k_b        = find(c, "layer.attn_k.bias");
    W->v_w        = find(c, "layer.attn_v.weight");
    W->v_b        = find(c, "layer.attn_v.bias");
    W->o_w        = find(c, "layer.attn_out.weight");
    W->o_b        = find(c, "layer.attn_out.bias");
    W->attn_ln_w  = find(c, "layer.attn_ln.weight");
    W->attn_ln_b  = find(c, "layer.attn_ln.bias");
    W->ffn_w      = find(c, "layer.ffn.weight");
    W->ffn_b      = find(c, "layer.ffn.bias");
    W->ffn_out_w  = find(c, "layer.ffn_out.weight");
    W->ffn_out_b  = find(c, "layer.ffn_out.bias");
    W->full_ln_w  = find(c, "layer.full_ln.weight");
    W->full_ln_b  = find(c, "layer.full_ln.bias");

    // post-BERT
    W->bert_enc_w = try_find(c, "bert_enc.weight");
    if (W->bert_enc_w) {
        W->bert_enc_b = find(c, "bert_enc.bias");

        W->pt_l0_fW = find(c, "pred_text.lstm0.fwd.W");
        W->pt_l0_fR = find(c, "pred_text.lstm0.fwd.R");
        W->pt_l0_fb = find(c, "pred_text.lstm0.fwd.b");
        W->pt_l0_bW = find(c, "pred_text.lstm0.bwd.W");
        W->pt_l0_bR = find(c, "pred_text.lstm0.bwd.R");
        W->pt_l0_bb = find(c, "pred_text.lstm0.bwd.b");
        W->pt_fc1_w = find(c, "pred_text.fc1.weight");
        W->pt_fc1_b = find(c, "pred_text.fc1.bias");
        W->pt_l2_fW = find(c, "pred_text.lstm2.fwd.W");
        W->pt_l2_fR = find(c, "pred_text.lstm2.fwd.R");
        W->pt_l2_fb = find(c, "pred_text.lstm2.fwd.b");
        W->pt_l2_bW = find(c, "pred_text.lstm2.bwd.W");
        W->pt_l2_bR = find(c, "pred_text.lstm2.bwd.R");
        W->pt_l2_bb = find(c, "pred_text.lstm2.bwd.b");
        W->pt_fc3_w = find(c, "pred_text.fc3.weight");
        W->pt_fc3_b = find(c, "pred_text.fc3.bias");

        W->dur_l_fW = find(c, "dur.lstm.fwd.W");
        W->dur_l_fR = find(c, "dur.lstm.fwd.R");
        W->dur_l_fb = find(c, "dur.lstm.fwd.b");
        W->dur_l_bW = find(c, "dur.lstm.bwd.W");
        W->dur_l_bR = find(c, "dur.lstm.bwd.R");
        W->dur_l_bb = find(c, "dur.lstm.bwd.b");
        W->dur_w = find(c, "dur_proj.weight");
        W->dur_b = find(c, "dur_proj.bias");

        W->ac_embd  = find(c, "acoustic.embd.weight");
        W->ac_c0_w  = find(c, "acoustic.cnn0.weight");
        W->ac_c0_b  = find(c, "acoustic.cnn0.bias");
        W->ac_ln0_g = find(c, "acoustic.ln0.gamma");
        W->ac_ln0_b = find(c, "acoustic.ln0.beta");
        W->ac_c1_w  = find(c, "acoustic.cnn1.weight");
        W->ac_c1_b  = find(c, "acoustic.cnn1.bias");
        W->ac_ln1_g = find(c, "acoustic.ln1.gamma");
        W->ac_ln1_b = find(c, "acoustic.ln1.beta");
        W->ac_l_fW  = find(c, "acoustic.lstm.fwd.W");
        W->ac_l_fR  = find(c, "acoustic.lstm.fwd.R");
        W->ac_l_fb  = find(c, "acoustic.lstm.fwd.b");
        W->ac_l_bW  = find(c, "acoustic.lstm.bwd.W");
        W->ac_l_bR  = find(c, "acoustic.lstm.bwd.R");
        W->ac_l_bb  = find(c, "acoustic.lstm.bwd.b");

        W->sh_fW = try_find(c, "shared.lstm.fwd.W");
        if (W->sh_fW) {
            W->sh_fR = find(c, "shared.lstm.fwd.R");
            W->sh_fb = find(c, "shared.lstm.fwd.b");
            W->sh_bW = find(c, "shared.lstm.bwd.W");
            W->sh_bR = find(c, "shared.lstm.bwd.R");
            W->sh_bb = find(c, "shared.lstm.bwd.b");
        }
    }
}

static void kt_model_free(kt_model * m) {
    if (m->weights_buf) ggml_backend_buffer_free(m->weights_buf);
    if (m->ctx_w)       ggml_free(m->ctx_w);
    memset(m, 0, sizeof(*m));
}

// ============================================================================
// HELPERS — primitives shared across stages
// ============================================================================

// LayerNorm over ne[0]: out = (norm(x) * w) + b.   eps configurable.
static gt kt_layer_norm(struct ggml_context * c, gt x, gt w, gt b, float eps) {
    x = ggml_norm(c, x, eps);
    if (w) x = ggml_mul(c, x, w);
    if (b) x = ggml_add(c, x, b);
    return x;
}

// AdaLayerNorm (NLC):
//   x: ne=(C, L)              feature dim is ne[0] (matches torch layer_norm_last)
//   style: ne=(style_dim,)
//   fcW: ne=(style_dim, 2C), fcB: ne=(2C,)
//
//   h = fcW * style + fcB         => (2C,)
//   gamma = h[0:C], beta = h[C:2C]
//   normed = layer_norm(x) over ne[0]=C
//   return normed * (1 + gamma) + beta   ((1+gamma) and beta broadcast over L)
static gt kt_ada_layer_norm(struct ggml_context * c, gt x, gt style,
                            gt fcW, gt fcB, int C) {
    gt h = ggml_mul_mat(c, fcW, style);
    h = ggml_add(c, h, fcB);
    const size_t fsz = sizeof(float);
    gt gamma = ggml_view_1d(c, h, C, 0);
    gt beta  = ggml_view_1d(c, h, C, (size_t) C * fsz);

    gt n = ggml_norm(c, x, 1e-5f);
    // Compute normed*(1+gamma)+beta as normed*gamma + normed + beta
    // (avoids materializing a "1" tensor; gamma and beta broadcast on ne[1]).
    gt n_g = ggml_mul(c, n, gamma);
    gt out = ggml_add(c, n_g, n);
    out = ggml_add(c, out, beta);
    return out;
}

// AdaIN1D on NLC tensors:  x ne=(C, L).
//
// Instance-norm normalizes over the time dim, which in NLC is ne[1].
// ggml_norm only normalizes over ne[0], so transpose, normalize, transpose
// back. Then apply per-channel norm.weight/bias (when present), scale by
// (1+gamma) and shift by beta where gamma/beta come from the style FC.
static gt kt_ada_in_1d(struct ggml_context * c, gt x, gt style,
                       gt fcW, gt fcB, gt nW, gt nB, int C) {
    gt h = ggml_mul_mat(c, fcW, style);   // (2C,)
    h = ggml_add(c, h, fcB);
    const size_t fsz = sizeof(float);
    gt gamma_1d = ggml_view_1d(c, h, C, 0);
    gt beta_1d  = ggml_view_1d(c, h, C, (size_t) C * fsz);
    // gamma/beta as 1D (C,). For ggml_mul/add against x ne=(C, L), broadcast
    // works (smaller has ne[1]=1 implicit for 1D tensors).
    gt gamma = ggml_cont(c, gamma_1d);
    gt beta  = ggml_cont(c, beta_1d);

    // Instance-norm: transpose NLC->NCL so ggml_norm normalizes over time.
    gt x_t = ggml_cont(c, ggml_transpose(c, x));    // (L, C)
    gt n_t = ggml_norm(c, x_t, 1e-5f);              // norm over ne[0]=L
    gt n   = ggml_cont(c, ggml_transpose(c, n_t));  // back to (C, L)

    if (nW) n = ggml_mul(c, n, nW);   // (C,) broadcasts over (C, L)
    if (nB) n = ggml_add(c, n, nB);

    // out = n*(1+gamma) + beta = n*gamma + n + beta
    gt n_g = ggml_mul(c, n, gamma);
    gt out = ggml_add(c, n_g, n);
    out = ggml_add(c, out, beta);
    return out;
}

// Snake activation on NLC tensors:  x + (1/alpha) * sin(alpha*x)^2.
// x ne=(C, L), alpha ne=(C,) — alpha broadcasts on ne[1] for both mul and div.
static gt kt_snake_1d(struct ggml_context * c, gt x, gt alpha) {
    gt ax = ggml_mul(c, x, alpha);
    gt s  = ggml_sin(c, ax);
    gt s2 = ggml_mul(c, s, s);
    gt s2_over_a = ggml_div(c, s2, alpha);
    return ggml_add(c, x, s2_over_a);
}

// Look up a tensor by formatted name (e.g. kt_get(m, "f0.0.c1.weight")).
// Returns NULL if not found.
static gt kt_get_fmt(const kt_model * m, const char * fmt, ...) {
    char buf[160];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return ggml_get_tensor(m->ctx_w, buf);
}

// Conv1d via explicit im2col+mul_mat with F32 dst_type — works with F32
// kernels (regular ggml_conv_1d hardcodes F16 dst_type which forces F16
// kernels). Used for STFT analysis where kernel precision matters.
static gt kt_conv1d_nlc_f32k(struct ggml_context * c, gt x, gt kw, gt kb,
                             int stride, int pad, int dilation)
{
    // x: NLC ne=(Cin, L)  -> NCL 3D ne=(L, Cin, 1)
    gt x_ncl = ggml_cont(c, ggml_transpose(c, x));
    gt x_3d  = ggml_reshape_3d(c, x_ncl, x_ncl->ne[0], x_ncl->ne[1], 1);
    // im2col with F32 dst
    gt im2col = ggml_im2col(c, kw, x_3d, stride, 0, pad, 0, dilation, 0,
                            /*is_2D=*/false, /*dst_type=*/GGML_TYPE_F32);
    // im2col ne=(Cin*K, OL, B). Reshape to 2D for mul_mat.
    gt im2col_2d = ggml_reshape_2d(c, im2col, im2col->ne[0],
                                   im2col->ne[1] * im2col->ne[2]);
    // kernel ne=(K, Cin, Cout)  -> reshape to (K*Cin, Cout)
    gt kw_2d = ggml_reshape_2d(c, kw, kw->ne[0] * kw->ne[1], kw->ne[2]);
    gt res = ggml_mul_mat(c, im2col_2d, kw_2d);   // ne=(Cout, OL*B)
    gt res_3d = ggml_reshape_3d(c, res, im2col->ne[1], kw->ne[2], im2col->ne[2]);
    if (kb) {
        gt b3 = ggml_reshape_3d(c, kb, 1, kb->ne[0], 1);
        res_3d = ggml_add(c, res_3d, b3);
    }
    gt y_2d = ggml_reshape_2d(c, res_3d, res_3d->ne[0], res_3d->ne[1]);
    return ggml_cont(c, ggml_transpose(c, y_2d));   // (Cout, L_out) NLC
}

// Conv1d on an NLC tensor x ne=(C, L). Returns NLC ne=(Cout, L_out).
// kernel `kw` is ggml ne=(K, Cin, Cout) (as packed by the converter).
// Pass pad=-1 for SAME padding ((K-1)/2).
static gt kt_conv1d_nlc(struct ggml_context * c, gt x, gt kw, gt kb,
                        int stride, int pad, int dilation)
{
    const int K = (int) kw->ne[0];
    if (pad < 0) pad = (K - 1) / 2;
    // NLC -> NCL (transpose + cont) and promote to 3D.
    gt x_ncl = ggml_cont(c, ggml_transpose(c, x));
    gt x_3d  = ggml_reshape_3d(c, x_ncl, x_ncl->ne[0], x_ncl->ne[1], 1);
    gt y_3d  = ggml_conv_1d(c, kw, x_3d, stride, pad, dilation);
    // Add bias broadcast over L.
    if (kb) {
        gt b3 = ggml_reshape_3d(c, kb, 1, kb->ne[0], 1);
        y_3d = ggml_add(c, y_3d, b3);
    }
    gt y_2d = ggml_reshape_2d(c, y_3d, y_3d->ne[0], y_3d->ne[1]);  // (L_out, Cout)
    return ggml_cont(c, ggml_transpose(c, y_2d));                  // (Cout, L_out) NLC
}

// Repeat-interleave along the last (L) axis 2x: (C, L) NLC -> (C, 2L) NLC.
// Each L-position is duplicated.  See dump_lstm_fixture.py adjacent test
// for the stack-then-permute trick.
static gt kt_repeat_interleave_2x_nlc(struct ggml_context * c, gt x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    // Reshape x to (1, C, L) so we can concat along dim 0.
    gt x3 = ggml_reshape_3d(c, x, C, L, 1);
    gt x3_top = ggml_reshape_4d(c, x3, x3->ne[0], x3->ne[1], 1, 1);
    // [x; x] along ne[2] (now-empty dim "2"): result ne=(C, L, 2).
    gt stacked = ggml_concat(c, x3_top, x3_top, 2);
    // Permute (C, L, 2) -> (C, 2, L): want new ne[0]=C(old[0]), new ne[1]=2(old[2]), new ne[2]=L(old[1]).
    gt p = ggml_permute(c, stacked, 0, 2, 1, 3);
    p = ggml_cont(c, p);
    return ggml_reshape_2d(c, p, C, 2 * L);
}

// Insert one zero between every pair of adjacent L-positions of x:
//   (C, L) NLC -> (C, 2L) NLC,  output[c, 2t] = x[c, t], output[c, 2t+1] = 0.
// Used to fake depthwise conv-transpose-1d (stride=2) via ggml_conv_1d_dw.
static gt kt_insert_zeros_2x_nlc(struct ggml_context * c, gt x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    gt zeros = ggml_scale(c, x, 0.0f);  // same shape, all zeros
    gt x4    = ggml_reshape_4d(c, x, C, L, 1, 1);
    gt z4    = ggml_reshape_4d(c, zeros, C, L, 1, 1);
    gt stacked = ggml_concat(c, x4, z4, 2);          // (C, L, 2, 1)
    gt p = ggml_permute(c, stacked, 0, 2, 1, 3);     // (C, 2, L, 1)
    p = ggml_cont(c, p);
    return ggml_reshape_2d(c, p, C, 2 * L);
}

// Depthwise conv-transpose-1d 2x (stride=2, padding=1, output_padding=1, K=3).
// Implemented via insert-zeros + ggml_conv_1d_dw on a K-flipped kernel.
// `pool_w` is ggml ne=(K, 1, C) with K-axis pre-flipped at convert time.
// Output: NLC ne=(C, 2L).
static gt kt_upsample_2x_dwT(struct ggml_context * c, gt x, gt pool_w, gt pool_b)
{
    const int K = (int) pool_w->ne[0];
    KT_ASSERT(K == 3, "kt_upsample_2x_dwT only verified for K=3 (got K=%d)", K);

    gt xup = kt_insert_zeros_2x_nlc(c, x);             // (C, 2L) NLC
    // ggml_conv_1d_dw wants NCL b ne=(L, Cin, B). Transform.
    gt xup_ncl = ggml_cont(c, ggml_transpose(c, xup)); // (2L, C)
    gt xup_3d  = ggml_reshape_3d(c, xup_ncl, xup_ncl->ne[0], xup_ncl->ne[1], 1);
    gt y_3d    = ggml_conv_1d_dw(c, pool_w, xup_3d, /*s0=*/1, /*p0=*/1, /*d0=*/1);
    if (pool_b) {
        gt b3 = ggml_reshape_3d(c, pool_b, 1, pool_b->ne[0], 1);
        y_3d = ggml_add(c, y_3d, b3);
    }
    gt y_2d = ggml_reshape_2d(c, y_3d, y_3d->ne[0], y_3d->ne[1]);
    return ggml_cont(c, ggml_transpose(c, y_2d));      // (C, 2L) NLC
}

// Build one AdaINResBlock1D from named-prefix tensors.
//   x:           (Cin, L) NLC (the data input)
//   shortcut_in: optional shortcut tensor — if NULL, use x.
//   divide:      if true, divide output by sqrt(2).
// Detects upsample/has_conv1x1 from tensor presence.
static gt build_ada_block_1d(struct ggml_context * c, const kt_model * m,
                             const char * prefix, gt x, gt style,
                             gt shortcut_in, int divide)
{
    gt n1_fcW = kt_get_fmt(m, "%s.n1.fcW", prefix);
    gt n1_fcB = kt_get_fmt(m, "%s.n1.fcB", prefix);
    gt n1_nW  = kt_get_fmt(m, "%s.n1.nW",  prefix);
    gt n1_nB  = kt_get_fmt(m, "%s.n1.nB",  prefix);
    gt n2_fcW = kt_get_fmt(m, "%s.n2.fcW", prefix);
    gt n2_fcB = kt_get_fmt(m, "%s.n2.fcB", prefix);
    gt n2_nW  = kt_get_fmt(m, "%s.n2.nW",  prefix);
    gt n2_nB  = kt_get_fmt(m, "%s.n2.nB",  prefix);
    gt c1_w   = kt_get_fmt(m, "%s.c1.weight", prefix);
    gt c1_b   = kt_get_fmt(m, "%s.c1.bias",   prefix);
    gt c2_w   = kt_get_fmt(m, "%s.c2.weight", prefix);
    gt c2_b   = kt_get_fmt(m, "%s.c2.bias",   prefix);
    gt sv_w   = kt_get_fmt(m, "%s.sv.weight", prefix);   // optional
    gt sv_b   = kt_get_fmt(m, "%s.sv.bias",   prefix);
    gt pool_w = kt_get_fmt(m, "%s.pool.weight", prefix); // optional (upsample)
    gt pool_b = kt_get_fmt(m, "%s.pool.bias",   prefix);

    KT_ASSERT(n1_fcW && n2_fcW && c1_w && c2_w,
              "missing AdaINResBlock1D tensors under prefix %s", prefix);

    const int upsample    = (pool_w != NULL);
    const int has_conv1x1 = (sv_w   != NULL);

    const int Cin = (int) x->ne[0];
    gt h = kt_ada_in_1d(c, x, style, n1_fcW, n1_fcB, n1_nW, n1_nB, Cin);
    h = ggml_leaky_relu(c, h, 0.2f, false);

    if (upsample) h = kt_upsample_2x_dwT(c, h, pool_w, pool_b);

    h = kt_conv1d_nlc(c, h, c1_w, c1_b, /*s=*/1, /*pad=*/-1, /*d=*/1);

    const int Cmid = (int) c1_w->ne[2];
    h = kt_ada_in_1d(c, h, style, n2_fcW, n2_fcB, n2_nW, n2_nB, Cmid);
    h = ggml_leaky_relu(c, h, 0.2f, false);

    h = kt_conv1d_nlc(c, h, c2_w, c2_b, 1, -1, 1);

    gt shortcut = shortcut_in ? shortcut_in : x;
    gt res;
    if (upsample) {
        gt sup = kt_repeat_interleave_2x_nlc(c, shortcut);
        res = kt_conv1d_nlc(c, sup, sv_w, sv_b, 1, 0, 1);
    } else if (has_conv1x1) {
        res = kt_conv1d_nlc(c, shortcut, sv_w, sv_b, 1, 0, 1);
    } else {
        res = shortcut;
    }

    gt out = ggml_add(c, h, res);
    if (divide) out = ggml_scale(c, out, 1.0f / sqrtf(2.0f));
    return out;
}

// Reflection-pad LEFT only by `n` samples on an NLC tensor x ne=(C, L).
// Prepends x[:, 1:n+1].flip(-1) to x (= n elements). For n=1 the flip is a
// no-op so we just take x[:, 1:2] and concat. Larger n unsupported (the
// model uses n=1 in exactly one place).
static gt kt_reflection_pad_left(struct ggml_context * c, gt x, int n) {
    gt result = x;
    if (n > 0) {
        if (n != 1) {
            KT_DIE("kt_reflection_pad_left only supports n=1 "
                   "(got %d)", n);
        }
        const int64_t C = x->ne[0];
        // View x[:, 1:2] — slice along ne[1] (the L axis).
        // ggml_view_2d: (ctx, t, ne0, ne1, nb1, offset).
        gt slice = ggml_view_2d(c, x, C, 1,
                                x->nb[1], (size_t) 1 * x->nb[1]);
        slice = ggml_cont(c, slice);                  // (C, 1)
        result = ggml_concat(c, slice, x, /*dim=*/1); // (C, L+1)
    }
    return result;
}

// Build positional ids tensor [0, 1, 2, ..., L-1] as I32 input tensor.
//
// Returns the input tensor (caller fills via ggml_backend_tensor_set later).

// ============================================================================
// LSTM helpers
// ============================================================================

// One direction of a bidir LSTM.
//
// x:   ne=(in, T)   F32   (in = in_size, T = sequence length)
// W:   ne=(in, 4H)  F32   (input-hidden weights, gate order ifgo)
// R:   ne=(H,  4H)  F32   (recurrent weights)
// b:   ne=(4H,)     F32   (combined bias)
// h0/c0: ne=(H,)    F32   (initial states; pass zero tensors)
//
// Returns hidden states with ne=(H, T). cgraph nodes for the per-timestep
// cpys are appended via ggml_build_forward_expand to ensure scheduling.
static gt kt_lstm_dir(struct ggml_context * c, struct ggml_cgraph * gf,
                     gt x, gt W, gt R, gt b, gt h0, gt c0,
                     int H, int T, int reverse, const char * name)
{
    gt Wx = ggml_mul_mat(c, W, x);              // (4H, T)
    Wx = ggml_add(c, Wx, b);                     // broadcast (4H,) over T

    gt out = ggml_new_tensor_2d(c, GGML_TYPE_F32, H, T);
    if (name) ggml_set_name(out, name);

    gt h_prev = h0;
    gt c_prev = c0;
    const size_t fsz = sizeof(float);

    for (int step = 0; step < T; step++) {
        const int t = reverse ? (T - 1 - step) : step;

        gt Wx_t = ggml_view_1d(c, Wx, 4 * H, (size_t) t * 4 * H * fsz);
        gt Rh   = ggml_mul_mat(c, R, h_prev);
        gt z    = ggml_add(c, Wx_t, Rh);

        gt zi = ggml_view_1d(c, z, H, 0);
        gt zf = ggml_view_1d(c, z, H, 1ull * H * fsz);
        gt zg = ggml_view_1d(c, z, H, 2ull * H * fsz);
        gt zo = ggml_view_1d(c, z, H, 3ull * H * fsz);

        gt gi  = ggml_sigmoid(c, zi);
        gt gf_ = ggml_sigmoid(c, zf);
        gt gg  = ggml_tanh   (c, zg);
        gt go  = ggml_sigmoid(c, zo);

        gt fc = ggml_mul(c, gf_, c_prev);
        gt ig = ggml_mul(c, gi,  gg);
        gt c_t = ggml_add(c, fc, ig);
        gt h_t = ggml_mul(c, go, ggml_tanh(c, c_t));

        gt dest = ggml_view_1d(c, out, H, (size_t) t * H * fsz);
        ggml_build_forward_expand(gf, ggml_cpy(c, h_t, dest));

        h_prev = h_t;
        c_prev = c_t;
    }
    return out;
}

// Bidirectional LSTM. Returns (2H, T) — forward (H, T) concatenated with
// backward (H, T) along channel dim.
static gt kt_bidir_lstm(struct ggml_context * c, struct ggml_cgraph * gf,
                        gt x, gt fW, gt fR, gt fb, gt bW, gt bR, gt bb,
                        gt h0, gt c0, int H, int T)
{
    gt fwd = kt_lstm_dir(c, gf, x, fW, fR, fb, h0, c0, H, T, 0, NULL);
    gt bwd = kt_lstm_dir(c, gf, x, bW, bR, bb, h0, c0, H, T, 1, NULL);
    return ggml_concat(c, fwd, bwd, 0);   // (2H, T)
}

// ============================================================================
// BERT/Albert encoder (M1)
// ============================================================================

// Returns (hidden=768, L) — final encoder hidden states.
static gt build_albert(struct ggml_context * c, struct ggml_cgraph * gf,
                       const kt_model * m, int L,
                       gt input_ids, gt pos_ids, gt type_ids)
{
    const kt_arch * a = &m->arch;
    const kt_weights * W = &m->w;
    gt h = ggml_get_rows(c, W->e_word, input_ids);
    gt p = ggml_get_rows(c, W->e_pos,  pos_ids);
    gt t = ggml_get_rows(c, W->e_type, type_ids);
    h = ggml_add(c, h, p);
    h = ggml_add(c, h, t);
    h = kt_layer_norm(c, h, W->e_ln_w, W->e_ln_b, a->ln_eps);
    h = ggml_mul_mat(c, W->proj_w, h);
    h = ggml_add(c, h, W->proj_b);

    const float kq_scale = 1.0f / sqrtf((float) a->head_dim);

    for (int il = 0; il < a->n_layers; il++) {
        gt residual = h;

        gt q = ggml_add(c, ggml_mul_mat(c, W->q_w, h), W->q_b);
        gt k = ggml_add(c, ggml_mul_mat(c, W->k_w, h), W->k_b);
        gt v = ggml_add(c, ggml_mul_mat(c, W->v_w, h), W->v_b);

        q = ggml_reshape_4d(c, q, a->head_dim, a->n_heads, L, 1);
        k = ggml_reshape_4d(c, k, a->head_dim, a->n_heads, L, 1);
        v = ggml_reshape_4d(c, v, a->head_dim, a->n_heads, L, 1);

        q = ggml_permute(c, q, 0, 2, 1, 3);
        k = ggml_permute(c, k, 0, 2, 1, 3);
        v = ggml_permute(c, v, 0, 2, 1, 3);
        v = ggml_cont(c, ggml_transpose(c, v));

        gt kq = ggml_mul_mat(c, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(c, kq, NULL, kq_scale, 0.0f);

        gt kqv = ggml_mul_mat(c, v, kq);
        kqv = ggml_permute(c, kqv, 0, 2, 1, 3);
        kqv = ggml_cont_2d(c, kqv, a->hidden, L);

        gt att_out = ggml_add(c, ggml_mul_mat(c, W->o_w, kqv), W->o_b);

        h = ggml_add(c, att_out, residual);
        h = kt_layer_norm(c, h, W->attn_ln_w, W->attn_ln_b, a->ln_eps);
        gt mid = h;

        gt ffn = ggml_add(c, ggml_mul_mat(c, W->ffn_w, h), W->ffn_b);
        ffn = ggml_gelu_erf(c, ffn);
        ffn = ggml_add(c, ggml_mul_mat(c, W->ffn_out_w, ffn), W->ffn_out_b);

        h = ggml_add(c, ffn, mid);
        h = kt_layer_norm(c, h, W->full_ln_w, W->full_ln_b, a->ln_eps);
    }
    (void) gf;
    return h;
}

// ============================================================================
// PredictorTextEncoder (TextStage)
// ============================================================================
//
// Inputs:
//   bert_out: ne=(256, L)        — bert_enc(bert_out) projection (NLC layout,
//                                   feature varies fastest = ne[0]=256)
//   prosody_style: ne=(128,)
// Output: prosody (NLC layout, ne=(128, L))

// Build a "style broadcast" tensor of shape (C, L): each column is a copy of
// `style` (length C). Equivalent to torch's prosody_style.unsqueeze(1) +
// torch.zeros_like(...). We use ggml_repeat against a target shape.
static gt kt_style_bcast_CxL(struct ggml_context * c, gt style, int C, int L) {
    // Build a target tensor (C, L) — anything with that shape will do; ggml_repeat
    // copies `style` into the broadcast dims to fill it.
    gt s_2d = ggml_reshape_2d(c, style, C, 1);
    gt target = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, L);
    return ggml_repeat(c, s_2d, target);
}

static gt build_pred_text(struct ggml_context * c, struct ggml_cgraph * gf,
                          const kt_model * m, gt bert_out, gt style,
                          gt h0_lstm, gt c0_lstm, int L)
{
    const kt_arch * a = &m->arch;
    const kt_weights * W = &m->w;
    const int C  = 128;                     // both LSTM outputs are 2H=128
    const int H  = a->lstm_hidden;          // 64

    gt s_bcast = kt_style_bcast_CxL(c, style, C, L);          // (128, L)

    // Concat bert_out (256, L) + s_bcast (128, L) along channel -> (384, L)
    gt x = ggml_concat(c, bert_out, s_bcast, 0);

    // ----- LSTM 0: (384, L) -> (128, L) -----
    gt y = kt_bidir_lstm(c, gf, x,
                        W->pt_l0_fW, W->pt_l0_fR, W->pt_l0_fb,
                        W->pt_l0_bW, W->pt_l0_bR, W->pt_l0_bb,
                        h0_lstm, c0_lstm, H, L);

    // ----- AdaLayerNorm using fc1 -----
    gt y1 = kt_ada_layer_norm(c, y, style, W->pt_fc1_w, W->pt_fc1_b, C);

    // Concat y1 (128, L) + s_bcast (128, L) -> (256, L)
    gt x2 = ggml_concat(c, y1, s_bcast, 0);

    // ----- LSTM 2: (256, L) -> (128, L) -----
    gt y2 = kt_bidir_lstm(c, gf, x2,
                         W->pt_l2_fW, W->pt_l2_fR, W->pt_l2_fb,
                         W->pt_l2_bW, W->pt_l2_bR, W->pt_l2_bb,
                         h0_lstm, c0_lstm, H, L);

    // ----- AdaLayerNorm using fc3 -----
    gt y3 = kt_ada_layer_norm(c, y2, style, W->pt_fc3_w, W->pt_fc3_b, C);
    return y3;  // (128, L)
}

// ============================================================================
// AcousticTextEncoder
// ============================================================================
// Layout convention here:
//   We track tensors as NLC (channels in ne[0], time in ne[1]) consistent
//   with the rest of TextStage. ggml_get_rows naturally returns NLC. For
//   conv1d (which wants NCL), we transpose+cont in/out around the conv.
//
// Inputs:  input_ids ne=(L,)
// Output:  ne=(128, L)  NLC  (channel varies fastest)
static gt build_acoustic(struct ggml_context * c, struct ggml_cgraph * gf,
                         const kt_model * m, gt input_ids,
                         gt h0_lstm, gt c0_lstm, int L)
{
    const kt_weights * W = &m->w;
    const int H = m->arch.lstm_hidden;

    gt x = ggml_get_rows(c, W->ac_embd, input_ids);   // ne=(128, L) NLC

    for (int i = 0; i < 2; i++) {
        gt cnnW = (i == 0) ? W->ac_c0_w : W->ac_c1_w;
        gt cnnB = (i == 0) ? W->ac_c0_b : W->ac_c1_b;
        gt lnG  = (i == 0) ? W->ac_ln0_g : W->ac_ln1_g;
        gt lnB  = (i == 0) ? W->ac_ln0_b : W->ac_ln1_b;
        const int K = (int) cnnW->ne[0];
        const int pad = (K - 1) / 2;

        // NLC -> NCL for conv1d: swap ne[0] and ne[1] then materialize.
        gt x_ncl = ggml_cont(c, ggml_transpose(c, x));  // ne=(L, C)

        // Promote to 3D (B=1) for ggml_conv_1d's expected input shape.
        gt x_ncl_3d = ggml_reshape_3d(c, x_ncl, x_ncl->ne[0], x_ncl->ne[1], 1);
        gt y_ncl_3d = ggml_conv_1d(c, cnnW, x_ncl_3d, /*s0=*/1, /*p0=*/pad, /*d0=*/1);
        // y_ncl_3d ne=(L, Cout, 1). Add bias (Cout,) broadcast over L.
        gt bias3 = ggml_reshape_3d(c, cnnB, 1, cnnB->ne[0], 1);
        y_ncl_3d = ggml_add(c, y_ncl_3d, bias3);

        // Drop batch dim -> (L, Cout)
        gt y_ncl = ggml_reshape_2d(c, y_ncl_3d, y_ncl_3d->ne[0], y_ncl_3d->ne[1]);

        // NCL -> NLC: transpose so channel is back in ne[0] for LN over channels.
        x = ggml_cont(c, ggml_transpose(c, y_ncl));        // ne=(C, L) NLC

        // LayerNorm over channels (ne[0]) + multiplicative gamma + beta.
        x = kt_layer_norm(c, x, lnG, lnB, 1e-5f);
        x = ggml_leaky_relu(c, x, 0.2f, false);
    }

    // x is NLC ne=(128, L). LSTM wants (in_size, T) which IS NLC. Pass as-is.
    gt y = kt_bidir_lstm(c, gf, x,
                        W->ac_l_fW, W->ac_l_fR, W->ac_l_fb,
                        W->ac_l_bW, W->ac_l_bR, W->ac_l_bb,
                        h0_lstm, c0_lstm, H, L);
    return y;   // (128, L) NLC — caller converts to NCL if desired
}

// ============================================================================
// TextStage builder
// ============================================================================

typedef struct {
    gt prosody_ncl;    // (256, L)  channel-major (NCL with C=ne[1], L=ne[0]?)
    gt text_ncl;       // (128, L)
    gt dur_sig;        // (50, L)   actually torch returns (1, L, 50); we keep (50, L) and reshape host-side
} kt_textstage_outs;

static kt_textstage_outs build_textstage(
    struct ggml_context * c, struct ggml_cgraph * gf,
    const kt_model * m, int L,
    gt input_ids, gt pos_ids, gt type_ids,
    gt style_prosodic, gt h0_lstm, gt c0_lstm)
{
    const kt_weights * W = &m->w;

    gt bert = build_albert(c, gf, m, L, input_ids, pos_ids, type_ids);  // (768, L)
    gt bert_proj = ggml_add(c, ggml_mul_mat(c, W->bert_enc_w, bert), W->bert_enc_b);  // (256, L)

    gt prosody = build_pred_text(c, gf, m, bert_proj, style_prosodic,
                                 h0_lstm, c0_lstm, L);                  // (128, L)

    // prosody_ncl (256, L) = concat(prosody (128, L), s_bcast (128, L)) on channel.
    gt s_bcast = kt_style_bcast_CxL(c, style_prosodic,
                                    m->arch.style_dim, L);
    gt prosody256 = ggml_concat(c, prosody, s_bcast, 0);                 // (256, L)

    // duration LSTM: in=256 -> out=128 (per timestep)
    gt dlstm = kt_bidir_lstm(c, gf, prosody256,
                             W->dur_l_fW, W->dur_l_fR, W->dur_l_fb,
                             W->dur_l_bW, W->dur_l_bR, W->dur_l_bb,
                             h0_lstm, c0_lstm,
                             m->arch.lstm_hidden, L);                    // (128, L)

    // dur_proj: 128 -> 50, then sigmoid. ggml result ne=(50, L).
    gt dur_logits = ggml_add(c, ggml_mul_mat(c, W->dur_w, dlstm), W->dur_b);
    gt dur_sig    = ggml_sigmoid(c, dur_logits);

    gt text_ncl = build_acoustic(c, gf, m, input_ids, h0_lstm, c0_lstm, L);  // (128, L)

    kt_textstage_outs r = { prosody256, text_ncl, dur_sig };
    return r;
}

// ============================================================================
// GeneratorStage front: shared LSTM + 6 AdaINResBlock1D + f0_proj/n_proj
// ============================================================================
//
// Inputs:
//   prosody_lr_ncl: ne=(L_lr, 256) NCL  (post length-regulation)  -- BUT we
//                   accept it as ggml ne=(256, F) NLC where F = n_frames.
//   style_prosodic: ne=(128,)
//   F = n_frames (length-regulated frame count)
//
// Outputs:
//   f0_proj: ne=(1, 2F) NLC  — single-channel f0 contour at 2× frames
//   n_proj:  ne=(1, 2F) NLC

typedef struct {
    gt f0_proj;   // (1, 2F)
    gt n_proj;    // (1, 2F)
} kt_genfront_outs;

static kt_genfront_outs build_generator_front(
    struct ggml_context * c, struct ggml_cgraph * gf,
    const kt_model * m, gt prosody_lr_nlc, gt style,
    gt h0_lstm, gt c0_lstm, int F)
{
    const kt_weights * W = &m->w;
    const int H = m->arch.lstm_hidden;

    // shared bidir LSTM: in=256, out=2H=128 -> ne=(128, F) NLC
    gt sh = kt_bidir_lstm(c, gf, prosody_lr_nlc,
                          W->sh_fW, W->sh_fR, W->sh_fb,
                          W->sh_bW, W->sh_bR, W->sh_bb,
                          h0_lstm, c0_lstm, H, F);

    // F0 path: 3 AdaINResBlock1D blocks, middle one upsamples (×2 along F).
    gt f0 = build_ada_block_1d(c, m, "f0.0", sh,  style, sh,  /*divide=*/1);
    f0    = build_ada_block_1d(c, m, "f0.1", f0,  style, f0,  1);   // upsample -> (C, 2F)
    f0    = build_ada_block_1d(c, m, "f0.2", f0,  style, f0,  1);

    // f0_proj conv1d: (Cmid, 2F) NLC -> (1, 2F) NLC
    gt f0_proj_w = kt_get_fmt(m, "f0_proj.weight");
    gt f0_proj_b = kt_get_fmt(m, "f0_proj.bias");
    gt f0p = kt_conv1d_nlc(c, f0, f0_proj_w, f0_proj_b, 1, -1, 1);

    // N path: same structure but separate weights
    gt nx = build_ada_block_1d(c, m, "n.0", sh,  style, sh,  1);
    nx    = build_ada_block_1d(c, m, "n.1", nx,  style, nx,  1);   // upsample
    nx    = build_ada_block_1d(c, m, "n.2", nx,  style, nx,  1);

    gt n_proj_w = kt_get_fmt(m, "n_proj.weight");
    gt n_proj_b = kt_get_fmt(m, "n_proj.bias");
    gt np = kt_conv1d_nlc(c, nx, n_proj_w, n_proj_b, 1, -1, 1);

    kt_genfront_outs r = { f0p, np };
    return r;
}

// ConvTranspose1d on NLC tensor x ne=(Cin, L). Returns NLC ne=(Cout, L_out_torch).
// ggml_conv_transpose_1d only supports padding=0, dilation=1. We pass p0=0
// and crop 2*pad samples from the output (pad each side) to match torch's
// padded output. Output_padding is not supported (would be a non-symmetric
// crop) — pass 0; our model's u0/u1 use OP=0.
//
// Kernel `kw` PyTorch shape (Cin, Cout, K) -> numpy -> ggml ne=(K, Cout, Cin).
// Bias `kb` shape (Cout,).
static gt kt_conv_transpose_1d_nlc(struct ggml_context * c, gt x, gt kw, gt kb,
                                   int stride, int pad)
{
    // ggml requires b to be 2D. Convert NLC ne=(Cin, L) -> 2D ne=(L, Cin) NCL.
    gt b = ggml_cont(c, ggml_transpose(c, x));   // (L, Cin)
    gt y = ggml_conv_transpose_1d(c, kw, b, stride, /*p0=*/0, /*d0=*/1);
    // y has ne=(L_out_ggml, Cout, 1, 1). Crop 2*pad samples (pad from each side).
    if (pad > 0) {
        const int64_t L_out = y->ne[0] - 2 * pad;
        const int64_t Cout  = y->ne[1];
        gt y_view = ggml_view_3d(c, y, L_out, Cout, 1,
                                 y->nb[1], y->nb[2],
                                 (size_t) pad * y->nb[0]);
        y = ggml_cont(c, y_view);
    }
    if (kb) {
        gt b3 = ggml_reshape_3d(c, kb, 1, kb->ne[0], 1);
        y = ggml_add(c, y, b3);
    }
    gt y_2d = ggml_reshape_2d(c, y, y->ne[0], y->ne[1]);
    return ggml_cont(c, ggml_transpose(c, y_2d));   // (Cout, L_out) NLC
}

// Custom CPU op for atan2(a, b) — ggml has no native atan2.
static void kt_atan2_op(struct ggml_tensor * dst,
                        const struct ggml_tensor * a,
                        const struct ggml_tensor * b,
                        int ith, int nth, void * ud)
{
    (void) ud;
    const int64_t n = ggml_nelements(dst);
    const int64_t per = (n + nth - 1) / nth;
    const int64_t i0 = ith * per;
    const int64_t i1 = (i0 + per < n) ? i0 + per : n;
    const float * aa = (const float *) a->data;
    const float * bb = (const float *) b->data;
    float * dd = (float *) dst->data;
    for (int64_t i = i0; i < i1; i++) {
        dd[i] = atan2f(aa[i], bb[i]);
    }
}

// AdaINResBlockHiFiGAN with snake activation. Output shape == input shape.
// Block has 3 dilations (1, 3, 5); each iteration is:
//   h = ada_in(a1.k, x) -> snake(al1.k) -> conv1d(c1.k, dilation=d, pad=d*(K-1)/2)
//     -> ada_in(a2.k, h) -> snake(al2.k) -> conv1d(c2.k, pad=(K2-1)/2)
//   out += h
// x is NLC ne=(C, L).
static gt build_hifi_block(struct ggml_context * c, const kt_model * m,
                           const char * prefix, gt x, gt style)
{
    const int C = (int) x->ne[0];
    static const int dilations[3] = { 1, 3, 5 };
    gt out = x;
    for (int k = 0; k < 3; k++) {
        const int d = dilations[k];
        gt a1_fcW = kt_get_fmt(m, "%s.a1.%d.fcW", prefix, k);
        gt a1_fcB = kt_get_fmt(m, "%s.a1.%d.fcB", prefix, k);
        gt a1_nW  = kt_get_fmt(m, "%s.a1.%d.nW",  prefix, k);
        gt a1_nB  = kt_get_fmt(m, "%s.a1.%d.nB",  prefix, k);
        gt a2_fcW = kt_get_fmt(m, "%s.a2.%d.fcW", prefix, k);
        gt a2_fcB = kt_get_fmt(m, "%s.a2.%d.fcB", prefix, k);
        gt a2_nW  = kt_get_fmt(m, "%s.a2.%d.nW",  prefix, k);
        gt a2_nB  = kt_get_fmt(m, "%s.a2.%d.nB",  prefix, k);
        gt al1    = kt_get_fmt(m, "%s.al1.%d",    prefix, k);
        gt al2    = kt_get_fmt(m, "%s.al2.%d",    prefix, k);
        gt c1_w   = kt_get_fmt(m, "%s.c1.%d.weight", prefix, k);
        gt c1_b   = kt_get_fmt(m, "%s.c1.%d.bias",   prefix, k);
        gt c2_w   = kt_get_fmt(m, "%s.c2.%d.weight", prefix, k);
        gt c2_b   = kt_get_fmt(m, "%s.c2.%d.bias",   prefix, k);

        const int K1 = (int) c1_w->ne[0];
        const int K2 = (int) c2_w->ne[0];

        gt h = kt_ada_in_1d(c, out, style, a1_fcW, a1_fcB, a1_nW, a1_nB, C);
        h = kt_snake_1d(c, h, al1);
        h = kt_conv1d_nlc(c, h, c1_w, c1_b, /*s=*/1, /*pad=*/d * (K1 - 1) / 2, /*d=*/d);
        h = kt_ada_in_1d(c, h, style, a2_fcW, a2_fcB, a2_nW, a2_nB, C);
        h = kt_snake_1d(c, h, al2);
        h = kt_conv1d_nlc(c, h, c2_w, c2_b, 1, (K2 - 1) / 2, 1);
        out = ggml_add(c, out, h);
    }
    return out;
}

// ============================================================================
// DecoderPipeline
// ============================================================================
//
// Inputs (all NLC):
//   text_lr_nlc:   ne=(128, F)  — length-regulated text features
//   f0_proj_nlc:   ne=(1, 2F)   — from generator front
//   n_proj_nlc:    ne=(1, 2F)
//   style_aco:     ne=(128,)    — acoustic style (style256[0:128])
//
// Output: NLC ne=(C_out, 2F)   (last decode block upsamples)
static gt build_decoder(struct ggml_context * c, const kt_model * m,
                        gt text_lr, gt f0_proj, gt n_proj, gt style_aco)
{
    gt asrW = kt_get_fmt(m, "dec.asr.weight");
    gt asrB = kt_get_fmt(m, "dec.asr.bias");
    gt f0W  = kt_get_fmt(m, "dec.f0_conv.weight");
    gt f0B  = kt_get_fmt(m, "dec.f0_conv.bias");
    gt nW   = kt_get_fmt(m, "dec.n_conv.weight");
    gt nB   = kt_get_fmt(m, "dec.n_conv.bias");

    // asr_res.0 is padding=0 in torch — keep that.
    gt asr   = kt_conv1d_nlc(c, text_lr, asrW, asrB, /*s=*/1, /*pad=*/0, 1);
    // F0/N downsample by 2
    gt f0_dn = kt_conv1d_nlc(c, f0_proj, f0W, f0B, /*s=*/2, /*pad=*/1, 1);
    gt n_dn  = kt_conv1d_nlc(c, n_proj,  nW,  nB,  /*s=*/2, /*pad=*/1, 1);

    // enc_in = cat([text_lr, f0_dn, n_dn]) along channels (ggml dim 0).
    gt enc_in = ggml_concat(c, text_lr, f0_dn, 0);
    enc_in    = ggml_concat(c, enc_in,  n_dn,  0);

    gt x = build_ada_block_1d(c, m, "dec.encode", enc_in, style_aco, enc_in, /*divide=*/1);

    for (int i = 0; i < 4; i++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "dec.decode.%d", i);
        gt x_cat = ggml_concat(c, x,     asr,   0);
        x_cat    = ggml_concat(c, x_cat, f0_dn, 0);
        x_cat    = ggml_concat(c, x_cat, n_dn,  0);
        x = build_ada_block_1d(c, m, prefix, x_cat, style_aco, x_cat, /*divide=*/1);
    }
    return x;
}

// ============================================================================
// compute_noise_contribs — sine-excitation generator + STFT + noise_res blocks
// ============================================================================
//
// Inputs (NLC):
//   f0_proj:    ne=(1, 2F)            single-channel f0 contour at frame rate
//   style_aco:  ne=(128,)
//   harmonics:  ne=(9,)               filled at host with [1.0, 2.0, ..., 9.0]
//   s_range:    ne=(300,)             filled at host with [0, 1, ..., 299]
//   eps_t:      ne=(1,)               filled at host with [1e-9]
//
// Per LLAMA.BACKEND.md design choice: phase_jitter and uv_noise are zeroed
// (deterministic output across backends). The torch reference also zeros
// these, so parity is measured against zero-noise torch.
//
// Returns:
//   nr0: NLC ne=(128, 20F)
//   nr1: NLC ne=(64,  120F+1)
typedef struct { gt nr0; gt nr1; } kt_noise_outs;

static kt_noise_outs build_noise_contribs(
    struct ggml_context * c, const kt_model * m,
    gt f0_proj, gt style_aco,
    gt harmonics, gt s_range, gt eps_t, int F)
{
    const int T_frames = 2 * F;
    const int hop      = 300;
    const int T_audio  = T_frames * hop;          // 600F
    const float sr     = 24000.0f;
    const float two_pi = 2.0f * (float) M_PI;

    // ---- nearest-neighbor upsample f0_proj (1, 2F) -> f0_audio (1, T_audio) ----
    // Reshape to (1, 1, T_frames) and ggml_repeat to (1, 300, T_frames). After
    // reshape to (1, T_audio), linear data[T] = f0[T/300]. (Verified by hand.)
    gt f0_3d  = ggml_reshape_3d(c, f0_proj, 1, 1, T_frames);
    gt f0_targ = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, hop, T_frames);
    gt f0_audio_3d = ggml_repeat(c, f0_3d, f0_targ);     // (1, hop, T_frames)
    gt f0_audio = ggml_reshape_2d(c, f0_audio_3d, 1, T_audio);  // NLC (1, T_audio)

    // ---- voiced mask = step(f0_audio) ----
    gt voiced = ggml_step(c, f0_audio);             // (1, T_audio)

    // ---- f0_per_frame[h, t] = f0[t] * (h+1) ----
    // Repeat f0_proj (1, 2F) -> (9, 2F) via ggml_repeat (smaller's ne[0]=1
    // divides target's ne[0]=9).
    gt f0_rep_target = ggml_new_tensor_2d(c, GGML_TYPE_F32, 9, T_frames);
    gt f0_repeated = ggml_repeat(c, f0_proj, f0_rep_target);   // (9, 2F)
    // harmonics (9,) reshaped to (9, 1) — broadcasts across time on ne[1].
    gt harm_2d = ggml_reshape_2d(c, harmonics, 9, 1);
    gt f0_per_frame = ggml_mul(c, f0_repeated, harm_2d);       // (9, 2F)

    // ---- step = f0_per_frame * (hop/sr); phase_start = (cumsum(step)-step)*2π ----
    gt step_nlc = ggml_scale(c, f0_per_frame, (float) hop / sr);
    // cumsum is along ne[0]; transpose to put time in ne[0].
    gt step_ncl = ggml_cont(c, ggml_transpose(c, step_nlc));   // (2F, 9) NCL
    gt cs       = ggml_cumsum(c, step_ncl);                    // cumsum over time
    gt ps_ncl   = ggml_sub(c, cs, step_ncl);                   // exclusive cumsum
    ps_ncl      = ggml_scale(c, ps_ncl, two_pi);
    gt phase_start_nlc = ggml_cont(c, ggml_transpose(c, ps_ncl));   // (9, 2F) NLC

    // ---- phase_within[h, t, s] = f0_per_frame[h, t] * s * (2π / sr) ----
    // ggml_mul/add only broadcast in ONE direction (smaller->bigger), so we
    // must explicitly expand both factors to the common shape (9, 2F, hop)
    // via ggml_repeat before multiplying.
    gt fpf_3d = ggml_reshape_3d(c, f0_per_frame, 9, T_frames, 1);  // (9, 2F, 1)
    gt s_3d   = ggml_reshape_3d(c, s_range, 1, 1, hop);            // (1, 1, hop)
    gt within_targ = ggml_new_tensor_3d(c, GGML_TYPE_F32, 9, T_frames, hop);
    gt fpf_x = ggml_repeat(c, fpf_3d, within_targ);                // (9, 2F, hop)
    gt s_x   = ggml_repeat(c, s_3d,   within_targ);                // (9, 2F, hop)
    gt within = ggml_mul(c, fpf_x, s_x);
    within = ggml_scale(c, within, two_pi / sr);

    // ---- expand phase_start to (9, 2F, hop), add to phase_within ----
    gt ps_3d = ggml_reshape_3d(c, phase_start_nlc, 9, T_frames, 1);
    gt ps_targ = ggml_new_tensor_3d(c, GGML_TYPE_F32, 9, T_frames, hop);
    gt ps_expanded = ggml_repeat(c, ps_3d, ps_targ);        // (9, 2F, hop)
    gt phase = ggml_add(c, ps_expanded, within);            // (9, 2F, hop)

    // Reorder to (9, hop, 2F) so reshape to (9, T_audio) gives correct layout
    // (consecutive samples within same frame, then advance frame).
    phase = ggml_permute(c, phase, 0, 2, 1, 3);              // (9, hop, 2F)
    phase = ggml_cont(c, phase);
    phase = ggml_reshape_2d(c, phase, 9, T_audio);           // (9, T_audio)

    // ---- sines = sin(phase) * sine_amp ----
    gt sines = ggml_scale(c, ggml_sin(c, phase), 0.1f);     // (9, T_audio)

    // ---- sin_gen = sines * voiced (broadcast 1-channel voiced over 9) ----
    gt sin_gen = ggml_mul(c, sines, voiced);                // (9, T_audio)

    // ---- mixed = l_lin(sin_gen): (9, T_audio) -> (1, T_audio) ----
    gt l_lin_w = kt_get_fmt(m, "l_lin.weight");
    gt l_lin_b = kt_get_fmt(m, "l_lin.bias");
    gt mixed = ggml_add(c, ggml_mul_mat(c, l_lin_w, sin_gen), l_lin_b);  // (1, T_audio)

    // ---- excitation = tanh(mixed) ----
    gt excitation = ggml_tanh(c, mixed);                    // (1, T_audio)

    // ---- STFT analysis: F32 kernel via custom im2col+mul_mat path ----
    gt stft_fr = kt_get_fmt(m, "stft_fwd.real");
    gt stft_fi = kt_get_fmt(m, "stft_fwd.imag");
    gt stft_real = kt_conv1d_nlc_f32k(c, excitation, stft_fr, NULL, /*s=*/5, /*pad=*/10, 1);
    gt stft_imag = kt_conv1d_nlc_f32k(c, excitation, stft_fi, NULL,     5,         10, 1);
    // stft_real/imag: NLC (11, 120F+1)

    // ---- mag = sqrt(real² + imag² + 1e-9) ----
    gt re2 = ggml_mul(c, stft_real, stft_real);
    gt im2 = ggml_mul(c, stft_imag, stft_imag);
    gt mag2 = ggml_add(c, re2, im2);
    mag2 = ggml_add(c, mag2, eps_t);                         // broadcast (1,) over (11, T_stft)
    gt mag = ggml_sqrt(c, mag2);

    // ---- phi = atan2(imag, real) via custom op ----
    gt phi = ggml_map_custom2(c, stft_imag, stft_real, kt_atan2_op,
                              GGML_N_TASKS_MAX, NULL);

    // ---- stft_out = concat([mag, phi]) along channels (ne[0]) ----
    gt stft_out = ggml_concat(c, mag, phi, 0);              // (22, T_stft)

    // ---- noise_convs ----
    gt nc0_w = kt_get_fmt(m, "nc0.weight");
    gt nc0_b = kt_get_fmt(m, "nc0.bias");
    gt nc1_w = kt_get_fmt(m, "nc1.weight");
    gt nc1_b = kt_get_fmt(m, "nc1.bias");
    gt nc0 = kt_conv1d_nlc(c, stft_out, nc0_w, nc0_b, /*s=*/6, /*pad=*/3, 1);   // (128, 20F)
    gt nc1 = kt_conv1d_nlc(c, stft_out, nc1_w, nc1_b, /*s=*/1, /*pad=*/0, 1);   // (64, 120F+1)

    // ---- noise_res HiFi-GAN blocks ----
    gt nr0 = build_hifi_block(c, m, "nr0", nc0, style_aco);
    gt nr1 = build_hifi_block(c, m, "nr1", nc1, style_aco);

    kt_noise_outs r = { nr0, nr1 };
    return r;
}

// ============================================================================
// GeneratorPipeline + iSTFT head
// ============================================================================
//
// Inputs (all NLC):
//   dec_out:    ne=(256, 2F)   from M2.D
//   noise_res0: ne=(128, 20F)  from compute_noise_contribs (zero for now)
//   noise_res1: ne=(64,  120F+1)
//   style_aco:  ne=(128,)
// Output: audio waveform float[T] (length depends on F).
static gt build_generator(struct ggml_context * c, const kt_model * m,
                          gt dec_out, gt nr0, gt nr1, gt style_aco)
{
    gt u0_w = kt_get_fmt(m, "gen.u0.weight");
    gt u0_b = kt_get_fmt(m, "gen.u0.bias");
    gt u1_w = kt_get_fmt(m, "gen.u1.weight");
    gt u1_b = kt_get_fmt(m, "gen.u1.bias");
    gt cp_w = kt_get_fmt(m, "gen.cp.weight");
    gt cp_b = kt_get_fmt(m, "gen.cp.bias");
    gt sb_r = kt_get_fmt(m, "stft_bwd.real");
    gt sb_i = kt_get_fmt(m, "stft_bwd.imag");

    gt x = ggml_leaky_relu(c, dec_out, 0.1f, false);
    x = kt_conv_transpose_1d_nlc(c, x, u0_w, u0_b, /*stride=*/10, /*pad=*/5);
    x = ggml_add(c, x, nr0);

    gt r0 = build_hifi_block(c, m, "gen.r0", x, style_aco);
    gt r1 = build_hifi_block(c, m, "gen.r1", x, style_aco);
    x = ggml_scale(c, ggml_add(c, r0, r1), 0.5f);
    x = ggml_leaky_relu(c, x, 0.1f, false);

    x = kt_conv_transpose_1d_nlc(c, x, u1_w, u1_b, /*stride=*/6, /*pad=*/3);
    x = kt_reflection_pad_left(c, x, 1);
    x = ggml_add(c, x, nr1);

    gt r2 = build_hifi_block(c, m, "gen.r2", x, style_aco);
    gt r3 = build_hifi_block(c, m, "gen.r3", x, style_aco);
    x = ggml_scale(c, ggml_add(c, r2, r3), 0.5f);
    x = ggml_leaky_relu(c, x, 0.1f, false);

    // conv_post: NLC ne=(64, L) -> ne=(22, L). K=7, padding=3 (auto half).
    x = kt_conv1d_nlc(c, x, cp_w, cp_b, /*s=*/1, /*pad=*/-1, /*d=*/1);

    // ----- iSTFT head -----
    // mag_logits = x[0:11, :], phase = x[11:22, :].
    const int64_t L = x->ne[1];
    gt mag_log = ggml_view_2d(c, x, 11, L, x->nb[1], 0);
    gt phase   = ggml_view_2d(c, x, 11, L, x->nb[1], (size_t) 11 * x->nb[0]);
    mag_log = ggml_cont(c, mag_log);
    phase   = ggml_cont(c, phase);
    gt mag = ggml_exp(c, mag_log);
    gt inner = ggml_sin(c, phase);
    gt re = ggml_mul(c, mag, ggml_cos(c, inner));
    gt im = ggml_mul(c, mag, ggml_sin(c, inner));

    // ConvTranspose1d (real and imag) stride=5, padding=0.
    gt audio_r = kt_conv_transpose_1d_nlc(c, re, sb_r, /*kb=*/NULL, /*stride=*/5, /*pad=*/0);
    gt audio_i = kt_conv_transpose_1d_nlc(c, im, sb_i, /*kb=*/NULL, /*stride=*/5, /*pad=*/0);

    // audio = audio_r - audio_i
    gt audio = ggml_sub(c, audio_r, audio_i);   // ne=(1, T_audio) NLC

    // Trim `trim` samples from each end. audio shape (1, T). Slice ne[1].
    const int trim = m->arch.istft_trim;   // 10
    const int64_t T = audio->ne[1];
    KT_ASSERT(T > 2 * trim, "audio too short to trim (T=%lld trim=%d)", (long long) T, trim);
    // (Earlier 2D-view-based trim was discarded; the simpler 1D-view path
    // below handles the (1, T) case correctly.)
    // For a 2D NLC with ne=(1, T), trim along ne[1]:
    // offset = trim * sizeof(float) (along time), since ne[0]=1.
    // Let me redo: ne[0]=1 (channel), ne[1]=T (time). Element (0, t) at byte
    // offset t * nb[0] = t * 4 (since nb[0]=sizeof(f32)). Wait no — for NLC
    // ne=(1, T), nb[0]=4, nb[1]=4*1=4. So linear data is data[t]. Trim along
    // time = view starting at offset = trim * 4.
    return ggml_cont(c, ggml_view_1d(c, audio, T - 2 * trim, (size_t) trim * sizeof(float)));
}

// Debug: outputs (excitation, mag, phi, nc0, nc1) — set by environment
// variable so we don't have to plumb a full mode for each.

// Run noise contribs and output nr0 / nr1.  Used for parity isolation.
typedef struct {
    gt nr0;
    gt nr1;
    gt phase;       // (9, T_audio)  — for debugging
    gt excitation;  // (1, T_audio)
    gt stft_real;   // (11, T_stft)
    gt stft_imag;
} kt_noise_dbg;

// Forward decl — defined later, used here in kt_run_fullgen.
static gt build_full_generator(
    struct ggml_context * c, const kt_model * m,
    gt dec_out, gt f0_proj, gt style_aco,
    gt harmonics, gt s_range, gt eps_t, int F);

// ============================================================================
// Per-stage runners — wrap one ggml graph per stage, take host arrays in/out.
// ============================================================================

// Run TextStage: ids[L] + style256[256] -> prosody256[256*L], text[128*L],
// dur_sig[50*L]. Output buffers must be pre-allocated by caller.
static void kt_run_textstage(kt_model * m, const int32_t * ids, int L,
                             const float * style256,
                             float * out_prosody, float * out_text, float * out_dur)
{
    const kt_arch * a = &m->arch;
    const size_t mem_size = ggml_tensor_overhead() * 200000ull
                          + ggml_graph_overhead_custom(200000, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

    gt input_ids = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(input_ids);
    gt pos_ids   = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(pos_ids);
    gt type_ids  = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(type_ids);
    gt style_pr  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->style_dim); ggml_set_input(style_pr);
    gt h0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(h0);
    gt c0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(c0);

    kt_textstage_outs o = build_textstage(gfc, gf, m, L,
                                          input_ids, pos_ids, type_ids,
                                          style_pr, h0, c0);
    ggml_set_output(o.prosody_ncl); ggml_set_output(o.text_ncl); ggml_set_output(o.dur_sig);
    ggml_build_forward_expand(gf, o.prosody_ncl);
    ggml_build_forward_expand(gf, o.text_ncl);
    ggml_build_forward_expand(gf, o.dur_sig);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    ggml_gallocr_alloc_graph(ga, gf);

    int32_t * pos = malloc(sizeof(int32_t) * L);
    int32_t * tp  = malloc(sizeof(int32_t) * L);
    for (int i = 0; i < L; i++) { pos[i] = i; tp[i] = 0; }
    ggml_backend_tensor_set(input_ids, ids, 0, sizeof(int32_t) * L);
    ggml_backend_tensor_set(pos_ids,   pos, 0, sizeof(int32_t) * L);
    ggml_backend_tensor_set(type_ids,  tp,  0, sizeof(int32_t) * L);
    free(pos); free(tp);
    ggml_backend_tensor_set(style_pr, style256 + a->style_dim, 0, sizeof(float) * a->style_dim);
    float * z = calloc(a->lstm_hidden, sizeof(float));
    ggml_backend_tensor_set(h0, z, 0, sizeof(float) * a->lstm_hidden);
    ggml_backend_tensor_set(c0, z, 0, sizeof(float) * a->lstm_hidden);
    free(z);

    ggml_backend_graph_compute(m->backend, gf);
    ggml_backend_tensor_get(o.prosody_ncl, out_prosody, 0, sizeof(float) * 256 * L);
    ggml_backend_tensor_get(o.text_ncl,    out_text,    0, sizeof(float) * 128 * L);
    ggml_backend_tensor_get(o.dur_sig,     out_dur,     0, sizeof(float) * 50 * L);

    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
}

// Run GenFront: prosody_lr[256*F] + style_pr[128] -> f0_proj[2F], n_proj[2F].
static void kt_run_genfront(kt_model * m, int F,
                            const float * prosody_lr, const float * style_pr,
                            float * out_f0p, float * out_np)
{
    const kt_arch * a = &m->arch;
    const size_t mem_size = ggml_tensor_overhead() * 200000ull
                          + ggml_graph_overhead_custom(200000, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

    gt pr_t  = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 256, F); ggml_set_input(pr_t);
    gt sp_t  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->style_dim); ggml_set_input(sp_t);
    gt h0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(h0);
    gt c0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(c0);

    kt_genfront_outs o = build_generator_front(gfc, gf, m, pr_t, sp_t, h0, c0, F);
    ggml_set_output(o.f0_proj); ggml_set_output(o.n_proj);
    ggml_build_forward_expand(gf, o.f0_proj);
    ggml_build_forward_expand(gf, o.n_proj);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    ggml_gallocr_alloc_graph(ga, gf);

    ggml_backend_tensor_set(pr_t, prosody_lr, 0, sizeof(float) * 256 * F);
    ggml_backend_tensor_set(sp_t, style_pr,   0, sizeof(float) * a->style_dim);
    float * z = calloc(a->lstm_hidden, sizeof(float));
    ggml_backend_tensor_set(h0, z, 0, sizeof(float) * a->lstm_hidden);
    ggml_backend_tensor_set(c0, z, 0, sizeof(float) * a->lstm_hidden);
    free(z);

    ggml_backend_graph_compute(m->backend, gf);
    ggml_backend_tensor_get(o.f0_proj, out_f0p, 0, sizeof(float) * 2 * F);
    ggml_backend_tensor_get(o.n_proj,  out_np,  0, sizeof(float) * 2 * F);

    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
}

// Run Decoder: text_lr[128*F] + f0_proj[2F] + n_proj[2F] + style_aco[128]
//              -> dec_out[256*2F] (caller pre-allocates 256*2F floats).
static void kt_run_decoder(kt_model * m, int F,
                           const float * text_lr,
                           const float * f0p, const float * np_,
                           const float * style_aco,
                           float * out_dec)
{
    const size_t mem_size = ggml_tensor_overhead() * 200000ull
                          + ggml_graph_overhead_custom(200000, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

    gt text_t  = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 128, F);   ggml_set_input(text_t);
    gt f0p_t   = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1,   2*F); ggml_set_input(f0p_t);
    gt np_t    = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1,   2*F); ggml_set_input(np_t);
    gt style_t = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);      ggml_set_input(style_t);

    gt out = build_decoder(gfc, m, text_t, f0p_t, np_t, style_t);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    ggml_gallocr_alloc_graph(ga, gf);

    ggml_backend_tensor_set(text_t,  text_lr,   0, sizeof(float) * 128 * F);
    ggml_backend_tensor_set(f0p_t,   f0p,       0, sizeof(float) * 2 * F);
    ggml_backend_tensor_set(np_t,    np_,       0, sizeof(float) * 2 * F);
    ggml_backend_tensor_set(style_t, style_aco, 0, sizeof(float) * 128);

    ggml_backend_graph_compute(m->backend, gf);
    ggml_backend_tensor_get(out, out_dec, 0, sizeof(float) * out->ne[0] * out->ne[1]);

    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
}

// Run full generator (with noise contribs): dec_out[256*2F] + f0p[2F] + style_aco[128]
// -> audio[T_audio]. Returns audio length in *out_T_audio.
static float * kt_run_fullgen(kt_model * m, int F,
                              const float * dec_out, const float * f0p,
                              const float * style_aco, int * out_T_audio)
{
    const size_t mem_size = ggml_tensor_overhead() * 200000ull
                          + ggml_graph_overhead_custom(200000, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

    gt dec_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 256, 2*F); ggml_set_input(dec_t);
    gt f0_t  = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1,   2*F); ggml_set_input(f0_t);
    gt sty_a = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);      ggml_set_input(sty_a);
    gt harm  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 9);        ggml_set_input(harm);
    gt s_rng = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 300);      ggml_set_input(s_rng);
    gt eps_t = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 1);        ggml_set_input(eps_t);

    gt audio = build_full_generator(gfc, m, dec_t, f0_t, sty_a, harm, s_rng, eps_t, F);
    ggml_set_output(audio);
    ggml_build_forward_expand(gf, audio);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    ggml_gallocr_alloc_graph(ga, gf);

    ggml_backend_tensor_set(dec_t, dec_out,   0, sizeof(float) * 256 * 2 * F);
    ggml_backend_tensor_set(f0_t,  f0p,       0, sizeof(float) * 2 * F);
    ggml_backend_tensor_set(sty_a, style_aco, 0, sizeof(float) * 128);
    float harm_buf[9]; for (int i = 0; i < 9; i++) harm_buf[i] = (float)(i + 1);
    ggml_backend_tensor_set(harm, harm_buf, 0, sizeof(harm_buf));
    float s_buf[300]; for (int i = 0; i < 300; i++) s_buf[i] = (float) i;
    ggml_backend_tensor_set(s_rng, s_buf, 0, sizeof(s_buf));
    float eps_buf[1] = { 1e-9f };
    ggml_backend_tensor_set(eps_t, eps_buf, 0, sizeof(eps_buf));

    ggml_backend_graph_compute(m->backend, gf);

    const int T = (int) audio->ne[0];
    float * abuf = malloc(sizeof(float) * T);
    ggml_backend_tensor_get(audio, abuf, 0, sizeof(float) * T);

    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
    *out_T_audio = T;
    return abuf;
}

// End-to-end generator: dec_out + f0_proj + style -> audio.
// Internally builds noise contribs and chains into build_generator.
static gt build_full_generator(
    struct ggml_context * c, const kt_model * m,
    gt dec_out, gt f0_proj, gt style_aco,
    gt harmonics, gt s_range, gt eps_t, int F)
{
    kt_noise_outs nz = build_noise_contribs(c, m, f0_proj, style_aco,
                                            harmonics, s_range, eps_t, F);
    return build_generator(c, m, dec_out, nz.nr0, nz.nr1, style_aco);
}

// ============================================================================
// Public C API (kt_create / kt_destroy / kt_synthesize / kt_audio_free)
// ============================================================================

#include "include/KittensGGML.h"

struct kt_ctx {
    kt_model       model;
    ggml_backend_t backend;
    char           err[256];
};

kt_ctx * kt_create(const char * gguf_path,
                   const char * backend_name) {
    kt_ctx * ctx = NULL;
    if (gguf_path) {
        ggml_backend_t b = NULL;
        if (!backend_name || !strcmp(backend_name, "cpu")) {
            b = ggml_backend_cpu_init();
        }
#ifdef KT_HAVE_METAL
        else if (!strcmp(backend_name, "metal")) {
            b = ggml_backend_metal_init();
        }
#endif
        if (b) {
            ctx = (kt_ctx *) calloc(1, sizeof(kt_ctx));
            if (ctx) {
                ctx->backend = b;
                kt_model_load(&ctx->model, gguf_path, b);
            } else {
                ggml_backend_free(b);
            }
        }
    }
    return ctx;
}

void kt_destroy(kt_ctx * ctx) {
    if (ctx) {
        kt_model_free(&ctx->model);
        if (ctx->backend) ggml_backend_free(ctx->backend);
        free(ctx);
    }
}

void kt_audio_free(kt_audio a) {
    if (a.samples) free(a.samples);
}

const char * kt_last_error(const kt_ctx * ctx) {
    return ctx ? ctx->err : "no context";
}

// Cosine fade-in helper.
static void kt_fade_in(float * x, int n, int fade) {
    if (fade > 0 && fade <= n) {
        for (int i = 0; i < fade; i++) {
            const float t = (float) i
                          / (float)(fade - 1 > 0 ? fade - 1 : 1);
            x[i] *= 0.5f - 0.5f * cosf((float) M_PI * t);
        }
    }
}

static void kt_fade_out(float * x, int n, int fade) {
    if (fade > 0 && fade <= n) {
        const int start = n - fade;
        for (int i = 0; i < fade; i++) {
            const float t = (float) i
                          / (float)(fade - 1 > 0 ? fade - 1 : 1);
            x[start + i] *= 0.5f + 0.5f * cosf((float) M_PI * t);
        }
    }
}

// Shared cleanup for kt_synthesize failures. Free-NULL is a no-op,
// so callers may pass any subset of allocated pointers; the rest may
// already be NULL or freed-and-NULL'd by the happy path.
static void kt_synth_fail(kt_ctx * ctx,
                          float * prosody, float * text, float * dur,
                          int * durs,
                          float * prosody_lr, float * text_lr,
                          float * f0p, float * np_, float * dec_out)
{
    free(prosody); free(text); free(dur); free(durs);
    free(prosody_lr); free(text_lr);
    free(f0p); free(np_); free(dec_out);
    snprintf(ctx->err, sizeof(ctx->err),
             "kt_synthesize: allocation or stage failure");
}

kt_audio kt_synthesize(kt_ctx * ctx, const int32_t * ids, int L,
                       const float * style256, float speed)
{
    kt_audio out = { NULL, 0 };
    if (!ctx || !ids || L <= 0 || !style256 || speed <= 0) return out;
    kt_model * m = &ctx->model;

    // All bufs declared up front + initialised to NULL so the
    // failure cleanup can free unconditionally regardless of which
    // stage we exit from.
    float * prosody = NULL, * text = NULL, * dur = NULL;
    int   * durs = NULL;
    float * prosody_lr = NULL, * text_lr = NULL;
    float * f0p = NULL, * np_ = NULL;
    float * dec_out = NULL;

    // ---- Stage 1: TextStage ----
    prosody = (float *) malloc(sizeof(float) * 256 * L);
    text    = (float *) malloc(sizeof(float) * 128 * L);
    dur     = (float *) malloc(sizeof(float) * 50 * L);
    if (!prosody || !text || !dur) {
        kt_synth_fail(ctx, prosody, text, dur, durs,
                      prosody_lr, text_lr, f0p, np_, dec_out);
        return out;
    }
    kt_run_textstage(m, ids, L, style256, prosody, text, dur);

    // ---- Length regulation: durations and expansion ----
    durs = (int *) malloc(sizeof(int) * L);
    if (!durs) {
        kt_synth_fail(ctx, prosody, text, dur, durs,
                      prosody_lr, text_lr, f0p, np_, dec_out);
        return out;
    }
    int F = 0;
    for (int i = 0; i < L; i++) {
        // dur layout: NLC ne=(50, L) so data[t*50 + j]. But actually
        // run_textstage writes ggml NLC linear which is data[l*50+j].
        // For NLC ne=(50, L), linear index of (j, l) is j + l*50. So
        // dur[l*50 + j] is sigmoid output for token l at logit
        // index j.
        float sum = 0;
        for (int j = 0; j < 50; j++) sum += dur[i * 50 + j];
        int d = (int) lrintf(sum / speed);
        if (d < 1) d = 1;
        durs[i] = d;
        F += d;
    }

    prosody_lr = (float *) calloc((size_t) 256 * F, sizeof(float));
    text_lr    = (float *) calloc((size_t) 128 * F, sizeof(float));
    if (!prosody_lr || !text_lr) {
        kt_synth_fail(ctx, prosody, text, dur, durs,
                      prosody_lr, text_lr, f0p, np_, dec_out);
        return out;
    }
    {
        // prosody NLC ne=(256, L) -> data[l*256 + c]. Same for text
        // NLC (128, L). Output prosody_lr ne=(256, F) ->
        // data[t*256 + c]. Expand each token l into durs[l]
        // consecutive frames.
        int t = 0;
        for (int l = 0; l < L; l++) {
            const int d = durs[l];
            for (int k = 0; k < d; k++, t++) {
                memcpy(prosody_lr + t * 256,
                       prosody + l * 256,
                       sizeof(float) * 256);
                memcpy(text_lr    + t * 128,
                       text    + l * 128,
                       sizeof(float) * 128);
            }
        }
    }
    free(prosody); prosody = NULL;
    free(text);    text    = NULL;
    free(dur);     dur     = NULL;
    free(durs);    durs    = NULL;

    // ---- Stage 2: GenFront ----
    f0p = (float *) malloc(sizeof(float) * 2 * F);
    np_ = (float *) malloc(sizeof(float) * 2 * F);
    if (!f0p || !np_) {
        kt_synth_fail(ctx, prosody, text, dur, durs,
                      prosody_lr, text_lr, f0p, np_, dec_out);
        return out;
    }
    kt_run_genfront(m, F, prosody_lr, style256 + 128, f0p, np_);
    free(prosody_lr); prosody_lr = NULL;

    // ---- Stage 3: Decoder -> dec_out ne=(256, 2F) flattened NLC ----
    dec_out = (float *) malloc(sizeof(float) * 256 * 2 * F);
    if (!dec_out) {
        kt_synth_fail(ctx, prosody, text, dur, durs,
                      prosody_lr, text_lr, f0p, np_, dec_out);
        return out;
    }
    kt_run_decoder(m, F, text_lr, f0p, np_, style256, dec_out);
    free(text_lr); text_lr = NULL;
    free(np_);     np_     = NULL;

    // ---- Stage 4: Full generator (with noise) -> audio ----
    int T_audio = 0;
    float * audio = kt_run_fullgen(m, F, dec_out, f0p,
                                   style256, &T_audio);
    free(dec_out); dec_out = NULL;
    free(f0p);     f0p     = NULL;
    if (!audio) {
        kt_synth_fail(ctx, prosody, text, dur, durs,
                      prosody_lr, text_lr, f0p, np_, dec_out);
        return out;
    }

    // ---- Tail-drop 3 frames + fade-in 3 ms + fade-out 40 ms ----
    int n = T_audio;
    const int tail_drop = 3 * 600;       // 3 frames × 600 samples
    if (n > tail_drop) n -= tail_drop;
    kt_fade_in (audio, n, 72);            // 3 ms
    kt_fade_out(audio, n, 960);           // 40 ms

    out.samples   = audio;
    out.n_samples = (uint64_t) n;
    return out;
}

// ============================================================================
// CLI dispatch
// ============================================================================

__attribute__((unused))
static ggml_backend_t kt_backend_init(const char * name) {
    ggml_backend_t b = NULL;
    if (!name || !strcmp(name, "cpu")) {
        b = ggml_backend_cpu_init();
    }
#ifdef KT_HAVE_METAL
    else if (!strcmp(name, "metal")) {
        b = ggml_backend_metal_init();
    }
#endif
    else {
        KT_DIE("unknown backend: %s", name);
    }
    return b;
}

// Run a textstage forward. Writes (prosody256_NL, text_NL, dur_sig_NL) into
// `out_concat` in that order, total size = (256+128+50)*L floats.
__attribute__((unused))
static double run_textstage(kt_model * m, const int32_t * ids, int L,
                            const float * style256, float * out_concat)
{
    const kt_arch * a = &m->arch;
    const size_t mem_size = ggml_tensor_overhead() * 200000ull
                            + ggml_graph_overhead_custom(200000, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, /*no_alloc=*/ true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

    gt input_ids = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(input_ids);
    gt pos_ids   = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(pos_ids);
    gt type_ids  = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(type_ids);
    gt style_pr  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->style_dim);
    ggml_set_input(style_pr);
    gt h0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(h0);
    gt c0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(c0);

    kt_textstage_outs o = build_textstage(gfc, gf, m, L,
                                          input_ids, pos_ids, type_ids,
                                          style_pr, h0, c0);
    ggml_set_output(o.prosody_ncl);
    ggml_set_output(o.text_ncl);
    ggml_set_output(o.dur_sig);
    ggml_build_forward_expand(gf, o.prosody_ncl);
    ggml_build_forward_expand(gf, o.text_ncl);
    ggml_build_forward_expand(gf, o.dur_sig);

    KT_LOG("textstage graph nodes=%d", ggml_graph_n_nodes(gf));

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

    int32_t * pos = malloc(sizeof(int32_t) * L);
    int32_t * tp  = malloc(sizeof(int32_t) * L);
    for (int i = 0; i < L; i++) { pos[i] = i; tp[i] = 0; }
    ggml_backend_tensor_set(input_ids, ids, 0, sizeof(int32_t) * L);
    ggml_backend_tensor_set(pos_ids,   pos, 0, sizeof(int32_t) * L);
    ggml_backend_tensor_set(type_ids,  tp,  0, sizeof(int32_t) * L);
    free(pos); free(tp);

    // style[128:256] is the prosodic half
    ggml_backend_tensor_set(style_pr, style256 + a->style_dim, 0,
                            sizeof(float) * a->style_dim);
    float * zeros = calloc(a->lstm_hidden, sizeof(float));
    ggml_backend_tensor_set(h0, zeros, 0, sizeof(float) * a->lstm_hidden);
    ggml_backend_tensor_set(c0, zeros, 0, sizeof(float) * a->lstm_hidden);
    free(zeros);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;

    ggml_backend_tensor_get(o.prosody_ncl, out_concat, 0,
                            sizeof(float) * 256 * L);
    ggml_backend_tensor_get(o.text_ncl, out_concat + 256 * L, 0,
                            sizeof(float) * 128 * L);
    ggml_backend_tensor_get(o.dur_sig,  out_concat + (256+128) * L, 0,
                            sizeof(float) * 50 * L);

    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
    return ms;
}

// Generator-front mode. Input: i32 F (n_frames), then 256*F f32 prosody_lr,
// then 128 f32 prosodic style. Output: 2F f32 (f0_proj) + 2F f32 (n_proj).
__attribute__((unused))
static double run_genfront(kt_model * m, int F, const float * prosody_lr,
                           const float * style128, float * out_concat)
{
    const kt_arch * a = &m->arch;
    const size_t mem_size = ggml_tensor_overhead() * 200000ull
                          + ggml_graph_overhead_custom(200000, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, /*no_alloc=*/ true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

    gt prosody_in = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 256, F);
    ggml_set_input(prosody_in);
    gt style_pr = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->style_dim);
    ggml_set_input(style_pr);
    gt h0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(h0);
    gt c0 = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, a->lstm_hidden); ggml_set_input(c0);

    kt_genfront_outs o = build_generator_front(gfc, gf, m, prosody_in, style_pr, h0, c0, F);
    ggml_set_output(o.f0_proj);
    ggml_set_output(o.n_proj);
    ggml_build_forward_expand(gf, o.f0_proj);
    ggml_build_forward_expand(gf, o.n_proj);
    KT_LOG("genfront graph nodes=%d", ggml_graph_n_nodes(gf));

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

    ggml_backend_tensor_set(prosody_in, prosody_lr, 0, sizeof(float) * 256 * F);
    ggml_backend_tensor_set(style_pr,   style128,   0, sizeof(float) * a->style_dim);
    float * zeros = calloc(a->lstm_hidden, sizeof(float));
    ggml_backend_tensor_set(h0, zeros, 0, sizeof(float) * a->lstm_hidden);
    ggml_backend_tensor_set(c0, zeros, 0, sizeof(float) * a->lstm_hidden);
    free(zeros);

    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    ggml_backend_tensor_get(o.f0_proj, out_concat,             0, sizeof(float) * 2 * F);
    ggml_backend_tensor_get(o.n_proj,  out_concat + 2 * F,     0, sizeof(float) * 2 * F);

    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
    return ms;
}

// ----- BERT mode (M1 compat) -----

__attribute__((unused))
static double run_bert(kt_model * m, const int32_t * ids, int L, float * out) {
    const size_t mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    void * mem = malloc(mem_size);
    struct ggml_init_params ip = { mem_size, mem, /*no_alloc=*/ true };
    struct ggml_context * gfc = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 8192, false);

    gt input_ids = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(input_ids);
    gt pos_ids   = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(pos_ids);
    gt type_ids  = ggml_new_tensor_1d(gfc, GGML_TYPE_I32, L); ggml_set_input(type_ids);

    gt h = build_albert(gfc, gf, m, L, input_ids, pos_ids, type_ids);
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

    int32_t * pos = malloc(sizeof(int32_t) * L);
    int32_t * tp  = malloc(sizeof(int32_t) * L);
    for (int i = 0; i < L; i++) { pos[i] = i; tp[i] = 0; }
    ggml_backend_tensor_set(input_ids, ids, 0, sizeof(int32_t) * L);
    ggml_backend_tensor_set(pos_ids,   pos, 0, sizeof(int32_t) * L);
    ggml_backend_tensor_set(type_ids,  tp,  0, sizeof(int32_t) * L);
    free(pos); free(tp);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (ggml_backend_graph_compute(m->backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;

    ggml_backend_tensor_get(h, out, 0, sizeof(float) * m->arch.hidden * L);
    ggml_gallocr_free(ga);
    ggml_free(gfc);
    free(mem);
    return ms;
}

// CLI dispatch is gated behind KT_BUILD_CLI so the file can be linked into
// an app target without colliding with the app's own main(). Build the
// standalone CLI with -DKT_BUILD_CLI (see scripts/build_kittens_tts.sh).
#ifdef KT_BUILD_CLI

static void usage(const char * a0) {
    fprintf(stderr,
        "Usage: %s --gguf X --mode bert|textstage|generator|full --input X --output X [--backend cpu|metal] [--repeat N]\n", a0);
}

int main(int argc, char ** argv) {
    const char * gguf_path = NULL;
    const char * input_path = NULL;
    const char * output_path = NULL;
    const char * backend_name = "cpu";
    const char * mode = "bert";
    int repeat = 1;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf")    && i+1<argc) gguf_path    = argv[++i];
        else if (!strcmp(argv[i], "--input")   && i+1<argc) input_path   = argv[++i];
        else if (!strcmp(argv[i], "--output")  && i+1<argc) output_path  = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i+1<argc) backend_name = argv[++i];
        else if (!strcmp(argv[i], "--mode")    && i+1<argc) mode         = argv[++i];
        else if (!strcmp(argv[i], "--repeat")  && i+1<argc) repeat       = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (!gguf_path || !input_path || !output_path) { usage(argv[0]); return 1; }

    ggml_backend_t backend = kt_backend_init(backend_name);
    KT_LOG("backend=%s  mode=%s", backend_name, mode);

    kt_model m;
    kt_model_load(&m, gguf_path, backend);

    if (!strcmp(mode, "bert")) {
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t L; if (fread(&L, 4, 1, f) != 1) KT_DIE("read L");
        int32_t * ids = malloc(sizeof(int32_t) * L);
        if (fread(ids, 4, L, f) != (size_t) L) KT_DIE("read ids");
        fclose(f);

        float * out = malloc(sizeof(float) * m.arch.hidden * L);
        for (int r = 0; r < repeat; r++) {
            const double ms = run_bert(&m, ids, L, out);
            KT_LOG("forward[%d/%d] %.2f ms", r+1, repeat, ms);
        }
        FILE * fo = fopen(output_path, "wb");
        fwrite(out, sizeof(float), (size_t) m.arch.hidden * L, fo);
        fclose(fo);
        free(out); free(ids);
    } else if (!strcmp(mode, "stft")) {
        // Debug: take excitation as input, output stft_real, stft_imag, mag, phi.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t T_audio; if (fread(&T_audio, 4, 1, f) != 1) KT_DIE("read T");
        float * exc = malloc(sizeof(float) * T_audio);
        if (fread(exc, sizeof(float), T_audio, f) != (size_t) T_audio) KT_DIE("read exc");
        fclose(f);

        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt exc_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1, T_audio); ggml_set_input(exc_t);
        gt eps_t = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 1);          ggml_set_input(eps_t);

        gt stft_fr = kt_get_fmt(&m, "stft_fwd.real");
        gt stft_fi = kt_get_fmt(&m, "stft_fwd.imag");
        gt sr = kt_conv1d_nlc_f32k(gfc, exc_t, stft_fr, NULL, 5, 10, 1);
        gt si = kt_conv1d_nlc_f32k(gfc, exc_t, stft_fi, NULL, 5, 10, 1);
        gt mag2 = ggml_add(gfc, ggml_add(gfc, ggml_mul(gfc, sr, sr),
                                              ggml_mul(gfc, si, si)),
                                eps_t);
        gt mag = ggml_sqrt(gfc, mag2);
        gt phi = ggml_map_custom2(gfc, si, sr, kt_atan2_op, GGML_N_TASKS_MAX, NULL);

        ggml_set_output(sr);
        ggml_set_output(si);
        ggml_set_output(mag);
        ggml_set_output(phi);
        ggml_build_forward_expand(gf, sr);
        ggml_build_forward_expand(gf, si);
        ggml_build_forward_expand(gf, mag);
        ggml_build_forward_expand(gf, phi);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");
        ggml_backend_tensor_set(exc_t, exc, 0, sizeof(float) * T_audio);
        float eps_buf[1] = { 1e-9f };
        ggml_backend_tensor_set(eps_t, eps_buf, 0, sizeof(eps_buf));

        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");

        const int T_stft = (int) sr->ne[1];
        const size_t n = 11 * T_stft;
        float * b_sr = malloc(sizeof(float) * n);
        float * b_si = malloc(sizeof(float) * n);
        float * b_m  = malloc(sizeof(float) * n);
        float * b_p  = malloc(sizeof(float) * n);
        ggml_backend_tensor_get(sr,  b_sr, 0, sizeof(float) * n);
        ggml_backend_tensor_get(si,  b_si, 0, sizeof(float) * n);
        ggml_backend_tensor_get(mag, b_m,  0, sizeof(float) * n);
        ggml_backend_tensor_get(phi, b_p,  0, sizeof(float) * n);

        FILE * fo = fopen(output_path, "wb");
        int32_t T_stft_i = T_stft;
        fwrite(&T_stft_i, sizeof(int32_t), 1, fo);
        fwrite(b_sr, sizeof(float), n, fo);
        fwrite(b_si, sizeof(float), n, fo);
        fwrite(b_m,  sizeof(float), n, fo);
        fwrite(b_p,  sizeof(float), n, fo);
        fclose(fo);
        free(b_sr); free(b_si); free(b_m); free(b_p);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(exc);
    } else if (!strcmp(mode, "excitation")) {
        // Debug: dump excitation (1, T_audio).
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        float * f0 = malloc(sizeof(float) * 2 * F);
        if (fread(f0, sizeof(float), 2 * F, f) != (size_t)(2*F)) KT_DIE("read f0");
        fclose(f);
        const int T_audio = 2 * F * 300;

        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt f0_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1, 2*F);   ggml_set_input(f0_t);
        gt sty  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);      ggml_set_input(sty);
        gt harm = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 9);        ggml_set_input(harm);
        gt s_rng = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 300);     ggml_set_input(s_rng);
        gt eps_t = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 1);       ggml_set_input(eps_t);

        // Replicate the computation up to excitation by inlining build_noise_contribs partially.
        // Easier: call build_noise_contribs but expose excitation. Hack: just compute it here.
        const int T_frames = 2 * F, hop = 300;
        const float sr = 24000.0f, two_pi = 2.0f * (float) M_PI;

        gt f0_audio_3d_t = ggml_new_tensor_3d(gfc, GGML_TYPE_F32, 1, hop, T_frames);
        gt f0_3d = ggml_reshape_3d(gfc, f0_t, 1, 1, T_frames);
        gt f0_audio = ggml_reshape_2d(gfc, ggml_repeat(gfc, f0_3d, f0_audio_3d_t), 1, T_audio);
        gt voiced = ggml_step(gfc, f0_audio);

        gt f0_rep_target = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 9, T_frames);
        gt f0_rep = ggml_repeat(gfc, f0_t, f0_rep_target);
        gt harm_2d = ggml_reshape_2d(gfc, harm, 9, 1);
        gt fpf = ggml_mul(gfc, f0_rep, harm_2d);
        gt step_nlc = ggml_scale(gfc, fpf, (float) hop / sr);
        gt step_ncl = ggml_cont(gfc, ggml_transpose(gfc, step_nlc));
        gt cs = ggml_cumsum(gfc, step_ncl);
        gt ps_ncl = ggml_scale(gfc, ggml_sub(gfc, cs, step_ncl), two_pi);
        gt ps_nlc = ggml_cont(gfc, ggml_transpose(gfc, ps_ncl));

        gt fpf_3d = ggml_reshape_3d(gfc, fpf, 9, T_frames, 1);
        gt s_3d = ggml_reshape_3d(gfc, s_rng, 1, 1, hop);
        gt within_targ = ggml_new_tensor_3d(gfc, GGML_TYPE_F32, 9, T_frames, hop);
        gt within = ggml_mul(gfc, ggml_repeat(gfc, fpf_3d, within_targ),
                                  ggml_repeat(gfc, s_3d,   within_targ));
        within = ggml_scale(gfc, within, two_pi / sr);
        gt ps_3d = ggml_reshape_3d(gfc, ps_nlc, 9, T_frames, 1);
        gt ps_targ = ggml_new_tensor_3d(gfc, GGML_TYPE_F32, 9, T_frames, hop);
        gt phase = ggml_add(gfc, ggml_repeat(gfc, ps_3d, ps_targ), within);
        phase = ggml_cont(gfc, ggml_permute(gfc, phase, 0, 2, 1, 3));
        phase = ggml_reshape_2d(gfc, phase, 9, T_audio);

        gt sines = ggml_scale(gfc, ggml_sin(gfc, phase), 0.1f);
        gt sin_gen = ggml_mul(gfc, sines, voiced);

        gt l_lin_w = kt_get_fmt(&m, "l_lin.weight");
        gt l_lin_b = kt_get_fmt(&m, "l_lin.bias");
        gt mixed = ggml_add(gfc, ggml_mul_mat(gfc, l_lin_w, sin_gen), l_lin_b);
        gt excitation = ggml_tanh(gfc, mixed);
        (void) sty; (void) eps_t;

        ggml_set_output(excitation);
        ggml_build_forward_expand(gf, excitation);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

        ggml_backend_tensor_set(f0_t, f0, 0, sizeof(float) * 2 * F);
        float harm_buf[9]; for (int i = 0; i < 9; i++) harm_buf[i] = (float)(i + 1);
        ggml_backend_tensor_set(harm, harm_buf, 0, sizeof(harm_buf));
        float s_buf[300]; for (int i = 0; i < 300; i++) s_buf[i] = (float) i;
        ggml_backend_tensor_set(s_rng, s_buf, 0, sizeof(s_buf));

        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");

        float * out = malloc(sizeof(float) * T_audio);
        ggml_backend_tensor_get(excitation, out, 0, sizeof(float) * T_audio);
        FILE * fo = fopen(output_path, "wb");
        int32_t T_i = (int32_t) T_audio;
        fwrite(&T_i, sizeof(int32_t), 1, fo);
        fwrite(out, sizeof(float), T_audio, fo);
        fclose(fo);
        free(out);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(f0);
    } else if (!strcmp(mode, "phase")) {
        // Debug: dump just the phase tensor (9, T_audio) before sin().
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        float * f0 = malloc(sizeof(float) * 2 * F);
        if (fread(f0, sizeof(float), 2 * F, f) != (size_t)(2*F)) KT_DIE("read f0");
        fclose(f);
        const int T_frames = 2 * F, hop = 300, T_audio = T_frames * hop;
        const float sr = 24000.0f, two_pi = 2.0f * (float) M_PI;

        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt f0_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1, T_frames);   ggml_set_input(f0_t);
        gt harm = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 9);              ggml_set_input(harm);
        gt s_rng = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, hop);           ggml_set_input(s_rng);

        // Replicate inside-noise math to get phase
        gt f0_rep_target = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 9, T_frames);
        gt f0_repeated = ggml_repeat(gfc, f0_t, f0_rep_target);
        gt harm_2d = ggml_reshape_2d(gfc, harm, 9, 1);
        gt f0_per_frame = ggml_mul(gfc, f0_repeated, harm_2d);
        gt step_nlc = ggml_scale(gfc, f0_per_frame, (float) hop / sr);
        gt step_ncl = ggml_cont(gfc, ggml_transpose(gfc, step_nlc));
        gt cs = ggml_cumsum(gfc, step_ncl);
        gt ps_ncl = ggml_sub(gfc, cs, step_ncl);
        ps_ncl = ggml_scale(gfc, ps_ncl, two_pi);
        gt phase_start_nlc = ggml_cont(gfc, ggml_transpose(gfc, ps_ncl));
        gt fpf_3d = ggml_reshape_3d(gfc, f0_per_frame, 9, T_frames, 1);
        gt s_3d = ggml_reshape_3d(gfc, s_rng, 1, 1, hop);
        gt within_targ = ggml_new_tensor_3d(gfc, GGML_TYPE_F32, 9, T_frames, hop);
        gt fpf_x = ggml_repeat(gfc, fpf_3d, within_targ);
        gt s_x = ggml_repeat(gfc, s_3d, within_targ);
        gt within = ggml_mul(gfc, fpf_x, s_x);
        within = ggml_scale(gfc, within, two_pi / sr);
        gt ps_3d = ggml_reshape_3d(gfc, phase_start_nlc, 9, T_frames, 1);
        gt ps_targ = ggml_new_tensor_3d(gfc, GGML_TYPE_F32, 9, T_frames, hop);
        gt ps_expanded = ggml_repeat(gfc, ps_3d, ps_targ);
        gt phase = ggml_add(gfc, ps_expanded, within);
        phase = ggml_permute(gfc, phase, 0, 2, 1, 3);
        phase = ggml_cont(gfc, phase);
        phase = ggml_reshape_2d(gfc, phase, 9, T_audio);

        ggml_set_output(phase);
        ggml_build_forward_expand(gf, phase);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

        ggml_backend_tensor_set(f0_t, f0, 0, sizeof(float) * T_frames);
        float harm_buf[9]; for (int i = 0; i < 9; i++) harm_buf[i] = (float)(i + 1);
        ggml_backend_tensor_set(harm, harm_buf, 0, sizeof(harm_buf));
        float s_buf[300]; for (int i = 0; i < 300; i++) s_buf[i] = (float) i;
        ggml_backend_tensor_set(s_rng, s_buf, 0, sizeof(s_buf));

        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");

        float * out = malloc(sizeof(float) * 9 * T_audio);
        ggml_backend_tensor_get(phase, out, 0, sizeof(float) * 9 * T_audio);
        FILE * fo = fopen(output_path, "wb");
        int32_t hdr[2] = { 9, T_audio };
        fwrite(hdr, sizeof(int32_t), 2, fo);
        fwrite(out, sizeof(float), 9 * T_audio, fo);
        fclose(fo);
        free(out);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(f0);
    } else if (!strcmp(mode, "tts")) {
        // End-to-end TTS via kt_synthesize. Input: i32 L, L i32 ids, 256 f32 style,
        // f32 speed. Output: i32 T_audio, T_audio f32 PCM.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t L; if (fread(&L, 4, 1, f) != 1) KT_DIE("read L");
        int32_t * ids = malloc(sizeof(int32_t) * L);
        if (fread(ids, 4, L, f) != (size_t) L) KT_DIE("read ids");
        float style[256];
        if (fread(style, sizeof(float), 256, f) != 256) KT_DIE("read style");
        float speed = 1.0f;
        if (fread(&speed, sizeof(float), 1, f) != 1) speed = 1.0f;
        fclose(f);

        // Build a kt_ctx out of the already-loaded model. We re-init ctx to
        // share storage (the model is loaded in `m`). Simpler: allocate a
        // fresh ctx that takes ownership.
        kt_ctx ctx = { .backend = m.backend };
        memcpy(&ctx.model, &m, sizeof(kt_model));
        // Note: we DON'T destroy ctx at the end since it shares m's storage —
        // m's destructor at end of main handles cleanup.

        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        kt_audio a = kt_synthesize(&ctx, ids, L, style, speed);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        KT_LOG("tts forward %.2f ms  audio=%.2f s", ms, (double) a.n_samples / 24000.0);

        FILE * fo = fopen(output_path, "wb");
        int32_t T_i = (int32_t) a.n_samples;
        fwrite(&T_i, sizeof(int32_t), 1, fo);
        fwrite(a.samples, sizeof(float), a.n_samples, fo);
        fclose(fo);
        kt_audio_free(a);
        free(ids);
    } else if (!strcmp(mode, "noise")) {
        // Standalone test for compute_noise_contribs.
        // Input: i32 F, 2F f32 f0_proj, 128 f32 acoustic_style.
        // Output: header (i32 C0, i32 L0, i32 C1, i32 L1) then nr0 then nr1.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        size_t need = sizeof(float) * (2 * F + 128);
        char * buf = malloc(need);
        if (fread(buf, 1, need, f) != need) KT_DIE("read inputs");
        fclose(f);
        float * f0    = (float *)(buf);
        float * style = f0 + 2 * F;

        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt f0_t  = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1, 2*F);   ggml_set_input(f0_t);
        gt sty_a = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);      ggml_set_input(sty_a);
        gt harm  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 9);        ggml_set_input(harm);
        gt s_rng = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 300);      ggml_set_input(s_rng);
        gt eps_t = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 1);        ggml_set_input(eps_t);

        kt_noise_outs nz = build_noise_contribs(gfc, &m, f0_t, sty_a, harm, s_rng, eps_t, F);
        ggml_set_output(nz.nr0);
        ggml_set_output(nz.nr1);
        ggml_build_forward_expand(gf, nz.nr0);
        ggml_build_forward_expand(gf, nz.nr1);
        KT_LOG("noise graph nodes=%d", ggml_graph_n_nodes(gf));

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

        ggml_backend_tensor_set(f0_t,  f0,    0, sizeof(float) * 2 * F);
        ggml_backend_tensor_set(sty_a, style, 0, sizeof(float) * 128);
        float harm_buf[9]; for (int i = 0; i < 9; i++) harm_buf[i] = (float)(i + 1);
        ggml_backend_tensor_set(harm, harm_buf, 0, sizeof(harm_buf));
        float s_buf[300]; for (int i = 0; i < 300; i++) s_buf[i] = (float) i;
        ggml_backend_tensor_set(s_rng, s_buf, 0, sizeof(s_buf));
        float eps_buf[1] = { 1e-9f };
        ggml_backend_tensor_set(eps_t, eps_buf, 0, sizeof(eps_buf));

        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");

        const int C0 = (int) nz.nr0->ne[0], L0 = (int) nz.nr0->ne[1];
        const int C1 = (int) nz.nr1->ne[0], L1 = (int) nz.nr1->ne[1];
        const size_t n0 = (size_t) C0 * L0;
        const size_t n1 = (size_t) C1 * L1;
        float * o0 = malloc(sizeof(float) * n0);
        float * o1 = malloc(sizeof(float) * n1);
        ggml_backend_tensor_get(nz.nr0, o0, 0, sizeof(float) * n0);
        ggml_backend_tensor_get(nz.nr1, o1, 0, sizeof(float) * n1);

        FILE * fo = fopen(output_path, "wb");
        int32_t hdr[4] = { C0, L0, C1, L1 };
        fwrite(hdr, sizeof(int32_t), 4, fo);
        fwrite(o0, sizeof(float), n0, fo);
        fwrite(o1, sizeof(float), n1, fo);
        fclose(fo);

        free(o0); free(o1);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(buf);
    } else if (!strcmp(mode, "fullgen")) {
        // Input: i32 F, then 256*2F f32 dec_out, 2F f32 f0_proj, 128 f32 style.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        const size_t L_dec = 2 * F;
        size_t need = sizeof(float) * (256 * L_dec + 2 * F + 128);
        char * buf = malloc(need);
        if (fread(buf, 1, need, f) != need) KT_DIE("read inputs");
        fclose(f);
        float * dec   = (float *)(buf);
        float * f0    = dec + 256 * L_dec;
        float * style = f0 + 2 * F;

        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt dec_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 256, L_dec); ggml_set_input(dec_t);
        gt f0_t  = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1,   2*F);   ggml_set_input(f0_t);
        gt sty_a = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);        ggml_set_input(sty_a);
        gt harm  = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 9);          ggml_set_input(harm);
        gt s_rng = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 300);        ggml_set_input(s_rng);
        gt eps_t = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 1);          ggml_set_input(eps_t);

        gt audio = build_full_generator(gfc, &m, dec_t, f0_t, sty_a,
                                        harm, s_rng, eps_t, F);
        ggml_set_output(audio);
        ggml_set_name(audio, "audio");
        ggml_build_forward_expand(gf, audio);
        KT_LOG("fullgen graph nodes=%d  audio.ne[0]=%lld",
               ggml_graph_n_nodes(gf), (long long) audio->ne[0]);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

        ggml_backend_tensor_set(dec_t, dec,   0, sizeof(float) * 256 * L_dec);
        ggml_backend_tensor_set(f0_t,  f0,    0, sizeof(float) * 2 * F);
        ggml_backend_tensor_set(sty_a, style, 0, sizeof(float) * 128);
        // Constants
        float harm_buf[9]; for (int i = 0; i < 9; i++) harm_buf[i] = (float)(i + 1);
        ggml_backend_tensor_set(harm, harm_buf, 0, sizeof(harm_buf));
        float s_buf[300]; for (int i = 0; i < 300; i++) s_buf[i] = (float) i;
        ggml_backend_tensor_set(s_rng, s_buf, 0, sizeof(s_buf));
        float eps_buf[1] = { 1e-9f };
        ggml_backend_tensor_set(eps_t, eps_buf, 0, sizeof(eps_buf));

        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        KT_LOG("fullgen forward %.2f ms", ms);

        const size_t T = (size_t) audio->ne[0];
        float * abuf = malloc(sizeof(float) * T);
        ggml_backend_tensor_get(audio, abuf, 0, sizeof(float) * T);
        FILE * fo = fopen(output_path, "wb");
        int32_t T_i = (int32_t) T;
        fwrite(&T_i, sizeof(int32_t), 1, fo);
        fwrite(abuf, sizeof(float), T, fo);
        fclose(fo);
        free(abuf);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(buf);
    } else if (!strcmp(mode, "generator")) {
        // Input: i32 F, 256*2F f32 dec_out, 128*20F f32 nr0, 64*(120F+1) f32 nr1, 128 f32 style.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        const size_t L_dec = 2 * F;
        const size_t L_n0  = 20 * F;
        const size_t L_n1  = 120 * F + 1;
        size_t need = sizeof(float) * (256 * L_dec + 128 * L_n0 + 64 * L_n1 + 128);
        char * buf = malloc(need);
        if (fread(buf, 1, need, f) != need) KT_DIE("read inputs");
        fclose(f);
        float * dec  = (float *)(buf);
        float * nr0_h = dec + 256 * L_dec;
        float * nr1_h = nr0_h + 128 * L_n0;
        float * style = nr1_h + 64 * L_n1;

        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt dec_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 256, L_dec); ggml_set_input(dec_t);
        gt nr0_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 128, L_n0);  ggml_set_input(nr0_t);
        gt nr1_t = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 64,  L_n1);  ggml_set_input(nr1_t);
        gt style_a = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);      ggml_set_input(style_a);

        gt audio = build_generator(gfc, &m, dec_t, nr0_t, nr1_t, style_a);
        ggml_set_output(audio);
        ggml_set_name(audio, "audio");
        ggml_build_forward_expand(gf, audio);
        KT_LOG("generator graph nodes=%d  audio.ne[0]=%lld",
               ggml_graph_n_nodes(gf), (long long) audio->ne[0]);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

        ggml_backend_tensor_set(dec_t,   dec,   0, sizeof(float) * 256 * L_dec);
        ggml_backend_tensor_set(nr0_t,   nr0_h, 0, sizeof(float) * 128 * L_n0);
        ggml_backend_tensor_set(nr1_t,   nr1_h, 0, sizeof(float) * 64  * L_n1);
        ggml_backend_tensor_set(style_a, style, 0, sizeof(float) * 128);

        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        KT_LOG("generator forward %.2f ms", ms);

        const size_t T = (size_t) audio->ne[0];
        float * abuf = malloc(sizeof(float) * T);
        ggml_backend_tensor_get(audio, abuf, 0, sizeof(float) * T);
        FILE * fo = fopen(output_path, "wb");
        int32_t T_i = (int32_t) T;
        fwrite(&T_i, sizeof(int32_t), 1, fo);
        fwrite(abuf, sizeof(float), T, fo);
        fclose(fo);
        free(abuf);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(buf);
    } else if (!strcmp(mode, "decoder")) {
        // Input: i32 F, then 128*F f32 text_lr, then 2F f32 f0_proj, then 2F f32 n_proj, then 128 f32 style.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        float * tl = malloc(sizeof(float) * 128 * F);
        if (fread(tl, sizeof(float), 128*F, f) != (size_t)(128*F)) KT_DIE("read text_lr");
        float * f0 = malloc(sizeof(float) * 2 * F);
        if (fread(f0, sizeof(float), 2*F, f) != (size_t)(2*F)) KT_DIE("read f0");
        float * np_ = malloc(sizeof(float) * 2 * F);
        if (fread(np_, sizeof(float), 2*F, f) != (size_t)(2*F)) KT_DIE("read n");
        float * style = malloc(sizeof(float) * 128);
        if (fread(style, sizeof(float), 128, f) != 128) KT_DIE("read style");
        fclose(f);

        // Build & run a graph for decoder
        const size_t mem_size = ggml_tensor_overhead() * 200000ull
                              + ggml_graph_overhead_custom(200000, false);
        void * mem = malloc(mem_size);
        struct ggml_init_params ip = { mem_size, mem, true };
        struct ggml_context * gfc = ggml_init(ip);
        struct ggml_cgraph * gf = ggml_new_graph_custom(gfc, 200000, false);

        gt text_lr = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 128, F);   ggml_set_input(text_lr);
        gt f0p     = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1,   2*F); ggml_set_input(f0p);
        gt nnp     = ggml_new_tensor_2d(gfc, GGML_TYPE_F32, 1,   2*F); ggml_set_input(nnp);
        gt style_a = ggml_new_tensor_1d(gfc, GGML_TYPE_F32, 128);      ggml_set_input(style_a);

        gt out = build_decoder(gfc, &m, text_lr, f0p, nnp, style_a);
        ggml_set_output(out);
        ggml_set_name(out, "dec_out");
        ggml_build_forward_expand(gf, out);
        KT_LOG("decoder graph nodes=%d  out shape=(%lld, %lld)",
               ggml_graph_n_nodes(gf), (long long) out->ne[0], (long long) out->ne[1]);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) KT_DIE("gallocr_alloc_graph");

        ggml_backend_tensor_set(text_lr, tl,    0, sizeof(float) * 128 * F);
        ggml_backend_tensor_set(f0p,     f0,    0, sizeof(float) * 2 * F);
        ggml_backend_tensor_set(nnp,     np_,   0, sizeof(float) * 2 * F);
        ggml_backend_tensor_set(style_a, style, 0, sizeof(float) * 128);

        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        if (ggml_backend_graph_compute(m.backend, gf) != GGML_STATUS_SUCCESS) KT_DIE("compute");
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        KT_LOG("decoder forward %.2f ms", ms);

        const size_t total = (size_t) out->ne[0] * out->ne[1];
        float * obuf = malloc(sizeof(float) * total);
        ggml_backend_tensor_get(out, obuf, 0, sizeof(float) * total);
        FILE * fo = fopen(output_path, "wb");
        // Write a small header: i32 C_out, i32 L_out, then floats
        int32_t C_out = (int32_t) out->ne[0];
        int32_t L_out = (int32_t) out->ne[1];
        fwrite(&C_out, sizeof(int32_t), 1, fo);
        fwrite(&L_out, sizeof(int32_t), 1, fo);
        fwrite(obuf, sizeof(float), total, fo);
        fclose(fo);
        free(obuf);
        ggml_gallocr_free(ga);
        ggml_free(gfc);
        free(mem);
        free(tl); free(f0); free(np_); free(style);
    } else if (!strcmp(mode, "genfront")) {
        // Input: i32 F, then 256*F f32 prosody_lr, then 128 f32 style.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t F; if (fread(&F, 4, 1, f) != 1) KT_DIE("read F");
        float * pr = malloc(sizeof(float) * 256 * F);
        if (fread(pr, sizeof(float), 256 * F, f) != (size_t)(256*F)) KT_DIE("read prosody_lr");
        float * style = malloc(sizeof(float) * 128);
        if (fread(style, sizeof(float), 128, f) != 128) KT_DIE("read style");
        fclose(f);

        const size_t total = 4 * F;   // 2F f0 + 2F n
        float * out = malloc(sizeof(float) * total);
        for (int r = 0; r < repeat; r++) {
            const double ms = run_genfront(&m, F, pr, style, out);
            KT_LOG("genfront[%d/%d] %.2f ms", r+1, repeat, ms);
        }
        FILE * fo = fopen(output_path, "wb");
        fwrite(out, sizeof(float), total, fo);
        fclose(fo);
        free(out); free(style); free(pr);
    } else if (!strcmp(mode, "textstage")) {
        // Input: i32 L, L i32 ids, 256 f32 style.
        FILE * f = fopen(input_path, "rb");
        if (!f) KT_DIE("fopen %s", input_path);
        int32_t L; if (fread(&L, 4, 1, f) != 1) KT_DIE("read L");
        int32_t * ids = malloc(sizeof(int32_t) * L);
        if (fread(ids, 4, L, f) != (size_t) L) KT_DIE("read ids");
        float * style = malloc(sizeof(float) * 256);
        if (fread(style, sizeof(float), 256, f) != 256) KT_DIE("read style");
        fclose(f);

        const size_t total = (size_t)(256 + 128 + 50) * L;
        float * out = malloc(sizeof(float) * total);
        for (int r = 0; r < repeat; r++) {
            const double ms = run_textstage(&m, ids, L, style, out);
            KT_LOG("textstage[%d/%d] %.2f ms", r+1, repeat, ms);
        }
        FILE * fo = fopen(output_path, "wb");
        fwrite(out, sizeof(float), total, fo);
        fclose(fo);
        free(out); free(style); free(ids);
    } else {
        KT_LOG("mode '%s' not yet wired up", mode);
    }

    kt_model_free(&m);
    ggml_backend_free(backend);
    return 0;
}

#endif // KT_BUILD_CLI

#endif // TARGET_OS_OSX || TARGET_OS_IOS

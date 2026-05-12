// kittens-tts-cpu.c -- pure-C / cblas KittenTTS backend.
//
// Status: SCAFFOLD. The context loads the GGUF, binds every named
// tensor that kittens-tts.c (the ggml backend) binds, and exposes the
// same C API as KittensGGML.h. kt_cpu_synthesize currently returns a
// short silent buffer — the four stages (textstage / genfront /
// decoder / fullgen) need to be ported from ggml/kittens-tts.c onto
// kt_tensor.
//
// Porting plan (each stage roughly mirrors the existing
// build_albert / build_pred_text / build_acoustic / build_decoder /
// build_generator chain in ggml/kittens-tts.c):
//
//   1. Translate each ggml_* call to its kt_* equivalent. The eager
//      kt API removes ggml_set_input/output, ggml_backend_tensor_set,
//      ggml_gallocr_t plumbing — just pass tensors through.
//   2. The LSTM helper (kt_lstm_dir / kt_bidir_lstm) is a tight loop
//      already; port verbatim.
//   3. kt_atan2_op (the ggml_map_custom2 wrapper) becomes a direct
//      kt_atan2 call.
//   4. Reflection pad / repeat-interleave / insert-zeros / depthwise
//      conv-transpose stay structurally the same.
//
// SESE cleanup goal: each ported builder is one entry, one exit; the
// monolithic graph-builder shape in kittens-tts.c becomes a sequence
// of small named helpers each returning a fully-realized tensor.

#include "KittensCPU.h"
#include "kt_tensor.h"
#include "kt_gguf.h"

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// ---------------------------------------------------------------------------
// Error reporting (per-context buffer; library is single-threaded)
// ---------------------------------------------------------------------------

typedef struct kt_cpu_arch {
    int vocab, max_pos, token_types;
    int embd_dim, hidden, n_layers, n_heads, head_dim, ffn_dim;
    float ln_eps;
    int bert_enc_dim, style_dim, lstm_hidden, dur_logits;
    int audio_per_frame, istft_hop, istft_trim;
} kt_cpu_arch;

// Bound weight pointers. Mirrors kt_weights in ggml/kittens-tts.c.
// Loaded lazily by kt_cpu_load_weight() into the weights_arena.
typedef struct kt_cpu_weights {
    // Albert / BERT
    kt_tensor * e_word, * e_pos, * e_type;
    kt_tensor * e_ln_w, * e_ln_b;
    kt_tensor * proj_w, * proj_b;
    kt_tensor * q_w, * q_b, * k_w, * k_b, * v_w, * v_b, * o_w, * o_b;
    kt_tensor * attn_ln_w, * attn_ln_b;
    kt_tensor * ffn_w, * ffn_b, * ffn_out_w, * ffn_out_b;
    kt_tensor * full_ln_w, * full_ln_b;
    // post-BERT projection
    kt_tensor * bert_enc_w, * bert_enc_b;
    // PredictorTextEncoder
    kt_tensor * pt_l0_fW, * pt_l0_fR, * pt_l0_fb;
    kt_tensor * pt_l0_bW, * pt_l0_bR, * pt_l0_bb;
    kt_tensor * pt_fc1_w, * pt_fc1_b;
    kt_tensor * pt_l2_fW, * pt_l2_fR, * pt_l2_fb;
    kt_tensor * pt_l2_bW, * pt_l2_bR, * pt_l2_bb;
    kt_tensor * pt_fc3_w, * pt_fc3_b;
    // Duration head
    kt_tensor * dur_l_fW, * dur_l_fR, * dur_l_fb;
    kt_tensor * dur_l_bW, * dur_l_bR, * dur_l_bb;
    kt_tensor * dur_w, * dur_b;
    // Acoustic text encoder
    kt_tensor * ac_embd;
    kt_tensor * ac_c0_w, * ac_c0_b, * ac_ln0_g, * ac_ln0_b;
    kt_tensor * ac_c1_w, * ac_c1_b, * ac_ln1_g, * ac_ln1_b;
    kt_tensor * ac_l_fW, * ac_l_fR, * ac_l_fb;
    kt_tensor * ac_l_bW, * ac_l_bR, * ac_l_bb;
    // GenFront shared LSTM
    kt_tensor * sh_fW, * sh_fR, * sh_fb;
    kt_tensor * sh_bW, * sh_bR, * sh_bb;
    // F0 / N / decoder / generator blocks are looked up dynamically by
    // formatted name (e.g. "f0.0.c1.weight", "dec.decode.3.pool.weight",
    // "gen.r0.c1.0.weight"). See kt_cpu_get_named().
} kt_cpu_weights;

typedef struct {
    char        name[64];
    kt_tensor * t;
} kt_named_entry;

struct kt_cpu_ctx {
    kt_gguf *         gguf;
    kt_arena *        weights_arena;    // model-lifetime
    kt_arena *        scratch_arena;    // reset each kt_cpu_synthesize
    kt_cpu_arch       arch;
    kt_cpu_weights    W;
    // Lazy name->tensor cache for dynamically named block weights
    // (e.g. "f0.0.c1.weight", "gen.r0.c1.0.weight"). Populated on first
    // lookup; entries live in weights_arena so they survive arena resets.
    kt_named_entry *  cache;
    int               cache_count;
    int               cache_cap;
    char              err[256];
};


static void set_ctx_err(kt_cpu_ctx * ctx, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(ctx->err, sizeof(ctx->err), fmt, ap);
    va_end(ap);
}

const char * kt_cpu_last_error(const kt_cpu_ctx * ctx) {
    return ctx != NULL ? ctx->err : kt_gguf_last_error();
}


// ---------------------------------------------------------------------------
// Weight binding
// ---------------------------------------------------------------------------

// Looked up by name; returns NULL if not present, caller decides.
static kt_tensor * kt_cpu_bind(kt_cpu_ctx * ctx, const char * name) {
    kt_tensor * t = kt_gguf_load_tensor(ctx->gguf, ctx->weights_arena, name);
    if (t == NULL) {
        set_ctx_err(ctx, "missing GGUF tensor: %s", name);
    }
    return t;
}

// Required: assert it exists.
static kt_tensor * kt_cpu_bind_req(kt_cpu_ctx * ctx, const char * name) {
    kt_tensor * t = kt_cpu_bind(ctx, name);
    if (t == NULL) {
        fprintf(stderr, "kt_cpu: required tensor missing: %s\n", name);
        abort();
    }
    return t;
}

static int kt_cpu_load_arch(kt_cpu_ctx * ctx) {
    int ok = 1;
    uint32_t v;
    if (!kt_gguf_get_u32(ctx->gguf, "kittens-tts.vocab_size", &v)) {
        set_ctx_err(ctx, "missing arch KV: kittens-tts.vocab_size");
        ok = 0;
    } else {
        ctx->arch.vocab = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.max_position", &v);
        ctx->arch.max_pos = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.token_types", &v);
        ctx->arch.token_types = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.embedding_dim", &v);
        ctx->arch.embd_dim = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.hidden_size", &v);
        ctx->arch.hidden = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.num_layers", &v);
        ctx->arch.n_layers = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.num_heads", &v);
        ctx->arch.n_heads = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.head_dim", &v);
        ctx->arch.head_dim = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.ffn_dim", &v);
        ctx->arch.ffn_dim = (int)v;
        kt_gguf_get_f32(ctx->gguf, "kittens-tts.layer_norm_eps",
                        &ctx->arch.ln_eps);
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.bert_enc_dim", &v);
        ctx->arch.bert_enc_dim = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.style_dim", &v);
        ctx->arch.style_dim = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.lstm_hidden", &v);
        ctx->arch.lstm_hidden = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.dur_logits", &v);
        ctx->arch.dur_logits = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.audio_per_frame", &v);
        ctx->arch.audio_per_frame = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.istft_hop", &v);
        ctx->arch.istft_hop = (int)v;
        kt_gguf_get_u32(ctx->gguf, "kittens-tts.istft_trim", &v);
        ctx->arch.istft_trim = (int)v;
    }
    return ok;
}

static void kt_cpu_bind_weights(kt_cpu_ctx * ctx) {
    kt_cpu_weights * W = &ctx->W;
    // Albert
    W->e_word   = kt_cpu_bind_req(ctx, "embd.word.weight");
    W->e_pos    = kt_cpu_bind_req(ctx, "embd.pos.weight");
    W->e_type   = kt_cpu_bind_req(ctx, "embd.type.weight");
    W->e_ln_w   = kt_cpu_bind_req(ctx, "embd.ln.weight");
    W->e_ln_b   = kt_cpu_bind_req(ctx, "embd.ln.bias");
    W->proj_w   = kt_cpu_bind_req(ctx, "embd_to_hidden.weight");
    W->proj_b   = kt_cpu_bind_req(ctx, "embd_to_hidden.bias");
    W->q_w      = kt_cpu_bind_req(ctx, "layer.attn_q.weight");
    W->q_b      = kt_cpu_bind_req(ctx, "layer.attn_q.bias");
    W->k_w      = kt_cpu_bind_req(ctx, "layer.attn_k.weight");
    W->k_b      = kt_cpu_bind_req(ctx, "layer.attn_k.bias");
    W->v_w      = kt_cpu_bind_req(ctx, "layer.attn_v.weight");
    W->v_b      = kt_cpu_bind_req(ctx, "layer.attn_v.bias");
    W->o_w      = kt_cpu_bind_req(ctx, "layer.attn_out.weight");
    W->o_b      = kt_cpu_bind_req(ctx, "layer.attn_out.bias");
    W->attn_ln_w= kt_cpu_bind_req(ctx, "layer.attn_ln.weight");
    W->attn_ln_b= kt_cpu_bind_req(ctx, "layer.attn_ln.bias");
    W->ffn_w    = kt_cpu_bind_req(ctx, "layer.ffn.weight");
    W->ffn_b    = kt_cpu_bind_req(ctx, "layer.ffn.bias");
    W->ffn_out_w= kt_cpu_bind_req(ctx, "layer.ffn_out.weight");
    W->ffn_out_b= kt_cpu_bind_req(ctx, "layer.ffn_out.bias");
    W->full_ln_w= kt_cpu_bind_req(ctx, "layer.full_ln.weight");
    W->full_ln_b= kt_cpu_bind_req(ctx, "layer.full_ln.bias");
    // post-BERT
    W->bert_enc_w = kt_cpu_bind_req(ctx, "bert_enc.weight");
    W->bert_enc_b = kt_cpu_bind_req(ctx, "bert_enc.bias");
    // PredictorTextEncoder
    W->pt_l0_fW = kt_cpu_bind_req(ctx, "pred_text.lstm0.fwd.W");
    W->pt_l0_fR = kt_cpu_bind_req(ctx, "pred_text.lstm0.fwd.R");
    W->pt_l0_fb = kt_cpu_bind_req(ctx, "pred_text.lstm0.fwd.b");
    W->pt_l0_bW = kt_cpu_bind_req(ctx, "pred_text.lstm0.bwd.W");
    W->pt_l0_bR = kt_cpu_bind_req(ctx, "pred_text.lstm0.bwd.R");
    W->pt_l0_bb = kt_cpu_bind_req(ctx, "pred_text.lstm0.bwd.b");
    W->pt_fc1_w = kt_cpu_bind_req(ctx, "pred_text.fc1.weight");
    W->pt_fc1_b = kt_cpu_bind_req(ctx, "pred_text.fc1.bias");
    W->pt_l2_fW = kt_cpu_bind_req(ctx, "pred_text.lstm2.fwd.W");
    W->pt_l2_fR = kt_cpu_bind_req(ctx, "pred_text.lstm2.fwd.R");
    W->pt_l2_fb = kt_cpu_bind_req(ctx, "pred_text.lstm2.fwd.b");
    W->pt_l2_bW = kt_cpu_bind_req(ctx, "pred_text.lstm2.bwd.W");
    W->pt_l2_bR = kt_cpu_bind_req(ctx, "pred_text.lstm2.bwd.R");
    W->pt_l2_bb = kt_cpu_bind_req(ctx, "pred_text.lstm2.bwd.b");
    W->pt_fc3_w = kt_cpu_bind_req(ctx, "pred_text.fc3.weight");
    W->pt_fc3_b = kt_cpu_bind_req(ctx, "pred_text.fc3.bias");
    // Duration head
    W->dur_l_fW = kt_cpu_bind_req(ctx, "dur.lstm.fwd.W");
    W->dur_l_fR = kt_cpu_bind_req(ctx, "dur.lstm.fwd.R");
    W->dur_l_fb = kt_cpu_bind_req(ctx, "dur.lstm.fwd.b");
    W->dur_l_bW = kt_cpu_bind_req(ctx, "dur.lstm.bwd.W");
    W->dur_l_bR = kt_cpu_bind_req(ctx, "dur.lstm.bwd.R");
    W->dur_l_bb = kt_cpu_bind_req(ctx, "dur.lstm.bwd.b");
    W->dur_w    = kt_cpu_bind_req(ctx, "dur_proj.weight");
    W->dur_b    = kt_cpu_bind_req(ctx, "dur_proj.bias");
    // Acoustic text encoder
    W->ac_embd  = kt_cpu_bind_req(ctx, "acoustic.embd.weight");
    W->ac_c0_w  = kt_cpu_bind_req(ctx, "acoustic.cnn0.weight");
    W->ac_c0_b  = kt_cpu_bind_req(ctx, "acoustic.cnn0.bias");
    W->ac_ln0_g = kt_cpu_bind_req(ctx, "acoustic.ln0.gamma");
    W->ac_ln0_b = kt_cpu_bind_req(ctx, "acoustic.ln0.beta");
    W->ac_c1_w  = kt_cpu_bind_req(ctx, "acoustic.cnn1.weight");
    W->ac_c1_b  = kt_cpu_bind_req(ctx, "acoustic.cnn1.bias");
    W->ac_ln1_g = kt_cpu_bind_req(ctx, "acoustic.ln1.gamma");
    W->ac_ln1_b = kt_cpu_bind_req(ctx, "acoustic.ln1.beta");
    W->ac_l_fW  = kt_cpu_bind_req(ctx, "acoustic.lstm.fwd.W");
    W->ac_l_fR  = kt_cpu_bind_req(ctx, "acoustic.lstm.fwd.R");
    W->ac_l_fb  = kt_cpu_bind_req(ctx, "acoustic.lstm.fwd.b");
    W->ac_l_bW  = kt_cpu_bind_req(ctx, "acoustic.lstm.bwd.W");
    W->ac_l_bR  = kt_cpu_bind_req(ctx, "acoustic.lstm.bwd.R");
    W->ac_l_bb  = kt_cpu_bind_req(ctx, "acoustic.lstm.bwd.b");
    // Shared LSTM (GenFront)
    W->sh_fW    = kt_cpu_bind_req(ctx, "shared.lstm.fwd.W");
    W->sh_fR    = kt_cpu_bind_req(ctx, "shared.lstm.fwd.R");
    W->sh_fb    = kt_cpu_bind_req(ctx, "shared.lstm.fwd.b");
    W->sh_bW    = kt_cpu_bind_req(ctx, "shared.lstm.bwd.W");
    W->sh_bR    = kt_cpu_bind_req(ctx, "shared.lstm.bwd.R");
    W->sh_bb    = kt_cpu_bind_req(ctx, "shared.lstm.bwd.b");
    // The F0/N/decoder/generator block tensors are looked up by
    // formatted name inside the inference path; not bound up-front.
}


// ---------------------------------------------------------------------------
// Public lifecycle
// ---------------------------------------------------------------------------

kt_cpu_ctx * kt_cpu_create(const char * gguf_path) {
    kt_cpu_ctx * ctx = (kt_cpu_ctx *)calloc(1, sizeof(*ctx));
    int success = 0;
    if (ctx == NULL) {
        // No context to set_err on; fall through.
    } else {
        ctx->gguf = kt_gguf_open(gguf_path);
        if (ctx->gguf == NULL) {
            snprintf(ctx->err, sizeof(ctx->err),
                     "kt_gguf_open(%s): %s",
                     gguf_path, kt_gguf_last_error());
        } else {
            ctx->weights_arena = kt_arena_new(64 * 1024 * 1024);
            // Scratch starts tiny (1 MB) and doubles on demand. Each
            // synthesize call ends with kt_arena_reset, so resting
            // size is just the first slab. With initial=64MB the app
            // sat on 64MB of mostly-empty scratch even at idle; 1MB
            // is plenty for the smallest stages and grows as needed.
            ctx->scratch_arena = kt_arena_new(1 * 1024 * 1024);
            if (kt_cpu_load_arch(ctx)) {
                kt_cpu_bind_weights(ctx);
                success = 1;
            }
        }
    }
    kt_cpu_ctx * result = NULL;
    if (success) {
        result = ctx;
    } else if (ctx != NULL) {
        kt_cpu_destroy(ctx);
    }
    return result;
}

void kt_cpu_destroy(kt_cpu_ctx * ctx) {
    if (ctx != NULL) {
        if (ctx->scratch_arena != NULL) {
            kt_arena_free(ctx->scratch_arena);
        }
        if (ctx->weights_arena != NULL) {
            kt_arena_free(ctx->weights_arena);
        }
        if (ctx->gguf != NULL) {
            kt_gguf_close(ctx->gguf);
        }
        free(ctx->cache);
        free(ctx);
    }
}

// ---------------------------------------------------------------------------
// Lazy lookup of block weights by formatted name.
// ---------------------------------------------------------------------------

static kt_tensor * kt_cpu_named(kt_cpu_ctx * ctx, const char * name) {
    for (int i = 0; i < ctx->cache_count; i++) {
        if (strcmp(ctx->cache[i].name, name) == 0) {
            return ctx->cache[i].t;
        }
    }
    kt_tensor * t = kt_gguf_load_tensor(ctx->gguf,
                                        ctx->weights_arena, name);
    if (t == NULL) { return NULL; }
    if (ctx->cache_count == ctx->cache_cap) {
        int nc = ctx->cache_cap == 0 ? 128 : ctx->cache_cap * 2;
        ctx->cache = (kt_named_entry *)realloc(ctx->cache,
                                  (size_t)nc * sizeof(kt_named_entry));
        ctx->cache_cap = nc;
    }
    snprintf(ctx->cache[ctx->cache_count].name,
             sizeof(ctx->cache[0].name), "%s", name);
    ctx->cache[ctx->cache_count].t = t;
    ctx->cache_count++;
    return t;
}

static kt_tensor * kt_cpu_named_fmt(kt_cpu_ctx * ctx,
                                    const char * fmt, ...) {
    char buf[64];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return kt_cpu_named(ctx, buf);
}

void kt_cpu_audio_free(kt_cpu_audio a) {
    free(a.samples);
}


// ===========================================================================
// Primitive helpers (mirror ggml/kittens-tts.c)
// ===========================================================================

// LayerNorm over ne[0] with optional gamma/beta. eps configurable.
static kt_tensor * kt_layer_norm(kt_tensor * x, kt_tensor * w,
                                 kt_tensor * b, float eps) {
    kt_tensor * h = kt_norm(x, 0, eps);
    if (w != NULL) { h = kt_mul(h, w); }
    if (b != NULL) { h = kt_add(h, b); }
    return h;
}

// AdaLayerNorm on (C, L) tensors. style:(style_dim,), fcW:(style_dim, 2C),
// fcB:(2C,). Splits the projected style into gamma|beta and applies
// out = normed * (1 + gamma) + beta with broadcast over L.
static kt_tensor * kt_ada_layer_norm(kt_tensor * x, kt_tensor * style,
                                     kt_tensor * fcW, kt_tensor * fcB,
                                     int C) {
    kt_tensor * h = kt_mul_mat(fcW, style);     // (2C,)
    h = kt_add(h, fcB);
    const size_t fsz = sizeof(float);
    kt_tensor * gamma = kt_view_1d(h, C, 0);
    kt_tensor * beta  = kt_view_1d(h, C, (size_t)C * fsz);
    kt_tensor * n   = kt_norm(x, 0, 1e-5f);
    kt_tensor * n_g = kt_mul(n, gamma);
    kt_tensor * out = kt_add(n_g, n);
    out = kt_add(out, beta);
    return out;
}

// AdaIN1D on NLC tensors x:(C, L). style:(style_dim,), fcW/fcB project
// to 2C (gamma|beta). nW/nB are per-channel multiplicative norm
// (may be NULL).
//
// Instance norm normalizes over the time axis. With ne[0]=C and ne[1]=L
// we transpose to put L innermost, normalize, then transpose back.
static kt_tensor * kt_ada_in_1d(kt_tensor * x, kt_tensor * style,
                                kt_tensor * fcW, kt_tensor * fcB,
                                kt_tensor * nW, kt_tensor * nB, int C) {
    kt_tensor * h = kt_mul_mat(fcW, style);
    h = kt_add(h, fcB);
    const size_t fsz = sizeof(float);
    kt_tensor * gamma = kt_view_1d(h, C, 0);
    kt_tensor * beta  = kt_view_1d(h, C, (size_t)C * fsz);
    kt_tensor * gamma_c = kt_cont(gamma);
    kt_tensor * beta_c  = kt_cont(beta);

    kt_tensor * x_t = kt_cont(kt_transpose(x));   // (L, C)
    kt_tensor * n_t = kt_norm(x_t, 0, 1e-5f);
    kt_tensor * n   = kt_cont(kt_transpose(n_t)); // (C, L)

    if (nW != NULL) { n = kt_mul(n, nW); }
    if (nB != NULL) { n = kt_add(n, nB); }

    kt_tensor * n_g = kt_mul(n, gamma_c);
    kt_tensor * out = kt_add(n_g, n);
    out = kt_add(out, beta_c);
    return out;
}

// Snake activation: x + (1/alpha) * sin(alpha*x)^2. x:(C, L), alpha:(C,)
static kt_tensor * kt_snake_1d(kt_tensor * x, kt_tensor * alpha) {
    kt_tensor * ax = kt_mul(x, alpha);
    kt_tensor * s  = kt_sin(ax);
    kt_tensor * s2 = kt_mul(s, s);
    kt_tensor * s2_over_a = kt_div(s2, alpha);
    return kt_add(x, s2_over_a);
}

// Conv1d on NLC tensor x:(C, L). Returns NLC (Cout, Lout). Kernel kw
// stored ne=(K, Cin, Cout). pad < 0 means SAME ((K-1)/2).
static kt_tensor * kt_conv1d_nlc(kt_tensor * x, kt_tensor * kw,
                                 kt_tensor * kb,
                                 int stride, int pad, int dilation) {
    const int K = (int)kw->ne[0];
    if (pad < 0) { pad = (K - 1) / 2; }
    kt_tensor * x_ncl = kt_cont(kt_transpose(x));   // (L, C)
    kt_tensor * x_3d = kt_reshape_3d(x_ncl,
                                     x_ncl->ne[0], x_ncl->ne[1], 1);
    kt_tensor * y_3d = kt_conv_1d(kw, x_3d, stride, pad, dilation);
    if (kb != NULL) {
        // bias (Cout,) broadcast along L via ne[1] of y_3d which is Cout.
        // y_3d ne=(Lout, Cout, 1). kb ne=(Cout,). We want to add per
        // channel: build a (1, Cout, 1) view.
        kt_tensor * b3 = kt_reshape_3d(kb, 1, kb->ne[0], 1);
        y_3d = kt_add(y_3d, b3);
    }
    kt_tensor * y_2d = kt_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return kt_cont(kt_transpose(y_2d));             // (Cout, Lout) NLC
}

// Repeat-interleave 2x along the L axis: (C, L) -> (C, 2L).
// output[c, 2l]   = x[c, l]
// output[c, 2l+1] = x[c, l]
static kt_tensor * kt_repeat_interleave_2x_nlc(kt_tensor * x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    kt_tensor * x4 = kt_reshape_4d(x, C, L, 1, 1);
    kt_tensor * stacked = kt_concat(x4, x4, 2);     // (C, L, 2, 1)
    kt_tensor * p = kt_permute(stacked, 0, 2, 1, 3); // (C, 2, L, 1)
    kt_tensor * pc = kt_cont(p);
    return kt_reshape_2d(pc, C, 2 * L);
}

// Insert one zero between every pair of adjacent L-positions:
// (C, L) -> (C, 2L), output[c, 2t]=x[c,t], output[c, 2t+1]=0.
static kt_tensor * kt_insert_zeros_2x_nlc(kt_tensor * x) {
    const int64_t C = x->ne[0];
    const int64_t L = x->ne[1];
    kt_tensor * zeros = kt_scale(x, 0.0f);
    kt_tensor * x4 = kt_reshape_4d(x, C, L, 1, 1);
    kt_tensor * z4 = kt_reshape_4d(zeros, C, L, 1, 1);
    kt_tensor * stacked = kt_concat(x4, z4, 2);
    kt_tensor * p = kt_permute(stacked, 0, 2, 1, 3);
    kt_tensor * pc = kt_cont(p);
    return kt_reshape_2d(pc, C, 2 * L);
}

// Depthwise conv-transpose-1d 2x (stride=2, padding=1, K=3) implemented
// via insert-zeros + depthwise conv on the K-flipped kernel. pool_w
// stored ne=(K, 1, C) with K-axis pre-flipped at convert time.
static kt_tensor * kt_upsample_2x_dwT(kt_tensor * x,
                                      kt_tensor * pool_w,
                                      kt_tensor * pool_b) {
    const int K = (int)pool_w->ne[0];
    assert(K == 3);
    kt_tensor * xup = kt_insert_zeros_2x_nlc(x);
    kt_tensor * xup_ncl = kt_cont(kt_transpose(xup));
    kt_tensor * xup_3d  = kt_reshape_3d(xup_ncl,
                                        xup_ncl->ne[0],
                                        xup_ncl->ne[1], 1);
    kt_tensor * y_3d = kt_conv_1d_dw(pool_w, xup_3d, 1, 1, 1);
    if (pool_b != NULL) {
        kt_tensor * b3 = kt_reshape_3d(pool_b, 1, pool_b->ne[0], 1);
        y_3d = kt_add(y_3d, b3);
    }
    kt_tensor * y_2d = kt_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return kt_cont(kt_transpose(y_2d));
}

// ConvTranspose1d on NLC: x:(Cin, L) -> (Cout, Lout). Kernel kw stored
// ne=(K, Cout, Cin) (PyTorch (Cin, Cout, K) packed by the converter).
// kt_conv_transpose_1d handles symmetric padding internally.
static kt_tensor * kt_conv_transpose_1d_nlc(kt_tensor * x,
                                            kt_tensor * kw,
                                            kt_tensor * kb,
                                            int stride, int pad) {
    kt_tensor * b = kt_cont(kt_transpose(x));   // (L, Cin)
    kt_tensor * b_3d = kt_reshape_3d(b, b->ne[0], b->ne[1], 1);
    kt_tensor * y_3d = kt_conv_transpose_1d(kw, b_3d, stride, pad);
    if (kb != NULL) {
        kt_tensor * bb = kt_reshape_3d(kb, 1, kb->ne[0], 1);
        y_3d = kt_add(y_3d, bb);
    }
    kt_tensor * y_2d = kt_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
    return kt_cont(kt_transpose(y_2d));
}

// Reflection-pad LEFT only by 1 sample on NLC tensor x:(C, L).
// Prepends x[:, 1:2] to x -> (C, L+1). Only n=1 supported (only use
// in the model is iSTFT n=1).
static kt_tensor * kt_reflection_pad_left(kt_tensor * x, int n) {
    kt_tensor * result = x;
    if (n > 0) {
        assert(n == 1);
        const int64_t C = x->ne[0];
        kt_tensor * slice = kt_view_2d(x, C, 1, (size_t)x->nb[1],
                                       (size_t)x->nb[1]);
        kt_tensor * slice_c = kt_cont(slice);
        result = kt_concat(slice_c, x, 1);
    }
    return result;
}

// Broadcast a 1D style (C,) to a (C, L) tensor — every column a copy
// of style.
static kt_tensor * kt_style_bcast_CxL(kt_arena * a, kt_tensor * style,
                                      int C, int L) {
    (void)a;
    kt_tensor * s2 = kt_reshape_2d(style, C, 1);
    return kt_repeat_to(s2, 2, C, L, 1, 1);
}


// ===========================================================================
// LSTM helpers (eager — no graph plumbing)
// ===========================================================================
//
// One direction of a bidir LSTM. ifgo gate order.
//   x:  (in_size, T)
//   W:  (in_size, 4H)
//   R:  (H,       4H)
//   b:  (4H,)
//   h0, c0: (H,) — initial states (zero tensors)
//   returns (H, T) packed
static kt_tensor * kt_lstm_dir(kt_arena * a,
                               kt_tensor * x, kt_tensor * W,
                               kt_tensor * R, kt_tensor * b,
                               kt_tensor * h0, kt_tensor * c0,
                               int H, int T, int reverse) {
    kt_tensor * Wx_full = kt_mul_mat(W, x);   // (4H, T)
    Wx_full = kt_add(Wx_full, b);             // broadcast (4H,) over T

    kt_tensor * out = kt_new_2d(a, H, T);

    kt_tensor * h_prev = h0;
    kt_tensor * c_prev = c0;
    const size_t fsz = sizeof(float);

    for (int step = 0; step < T; step++) {
        const int t = reverse ? (T - 1 - step) : step;

        kt_tensor * Wx_t = kt_view_1d(Wx_full, 4 * H,
                                      (size_t)t * 4 * H * fsz);
        kt_tensor * Rh   = kt_mul_mat(R, h_prev);
        kt_tensor * z    = kt_add(Wx_t, Rh);

        kt_tensor * zi = kt_view_1d(z, H, 0);
        kt_tensor * zf = kt_view_1d(z, H, (size_t)1 * H * fsz);
        kt_tensor * zg = kt_view_1d(z, H, (size_t)2 * H * fsz);
        kt_tensor * zo = kt_view_1d(z, H, (size_t)3 * H * fsz);

        kt_tensor * gi  = kt_sigmoid(zi);
        kt_tensor * gf_ = kt_sigmoid(zf);
        kt_tensor * gg  = kt_tanh   (zg);
        kt_tensor * go  = kt_sigmoid(zo);

        kt_tensor * fc  = kt_mul(gf_, c_prev);
        kt_tensor * ig  = kt_mul(gi,  gg);
        kt_tensor * c_t = kt_add(fc, ig);
        kt_tensor * h_t = kt_mul(go, kt_tanh(c_t));

        // Write h_t into out[:, t]. out is packed (H, T), so the dest
        // column starts at byte t*H*4 in linear data.
        memcpy((char *)out->data + (size_t)t * H * fsz,
               h_t->data, (size_t)H * fsz);

        h_prev = h_t;
        c_prev = c_t;
    }
    return out;
}

// Bidirectional LSTM: returns (2H, T), forward concatenated with
// backward along ne[0].
static kt_tensor * kt_bidir_lstm(kt_arena * a,
                                 kt_tensor * x,
                                 kt_tensor * fW, kt_tensor * fR,
                                 kt_tensor * fb,
                                 kt_tensor * bW, kt_tensor * bR,
                                 kt_tensor * bb,
                                 kt_tensor * h0, kt_tensor * c0,
                                 int H, int T) {
    kt_tensor * fwd = kt_lstm_dir(a, x, fW, fR, fb, h0, c0, H, T, 0);
    kt_tensor * bwd = kt_lstm_dir(a, x, bW, bR, bb, h0, c0, H, T, 1);
    return kt_concat(fwd, bwd, 0);
}


// ===========================================================================
// AdaINResBlock1D builder (used in F0/N paths and decoder)
// ===========================================================================

static kt_tensor * build_ada_block_1d(kt_cpu_ctx * ctx, const char * prefix,
                                      kt_tensor * x, kt_tensor * style,
                                      kt_tensor * shortcut_in, int divide) {
    kt_tensor * n1_fcW = kt_cpu_named_fmt(ctx, "%s.n1.fcW", prefix);
    kt_tensor * n1_fcB = kt_cpu_named_fmt(ctx, "%s.n1.fcB", prefix);
    kt_tensor * n1_nW  = kt_cpu_named_fmt(ctx, "%s.n1.nW",  prefix);
    kt_tensor * n1_nB  = kt_cpu_named_fmt(ctx, "%s.n1.nB",  prefix);
    kt_tensor * n2_fcW = kt_cpu_named_fmt(ctx, "%s.n2.fcW", prefix);
    kt_tensor * n2_fcB = kt_cpu_named_fmt(ctx, "%s.n2.fcB", prefix);
    kt_tensor * n2_nW  = kt_cpu_named_fmt(ctx, "%s.n2.nW",  prefix);
    kt_tensor * n2_nB  = kt_cpu_named_fmt(ctx, "%s.n2.nB",  prefix);
    kt_tensor * c1_w   = kt_cpu_named_fmt(ctx, "%s.c1.weight", prefix);
    kt_tensor * c1_b   = kt_cpu_named_fmt(ctx, "%s.c1.bias",   prefix);
    kt_tensor * c2_w   = kt_cpu_named_fmt(ctx, "%s.c2.weight", prefix);
    kt_tensor * c2_b   = kt_cpu_named_fmt(ctx, "%s.c2.bias",   prefix);
    kt_tensor * sv_w   = kt_cpu_named_fmt(ctx, "%s.sv.weight", prefix);
    kt_tensor * sv_b   = kt_cpu_named_fmt(ctx, "%s.sv.bias",   prefix);
    kt_tensor * pool_w = kt_cpu_named_fmt(ctx, "%s.pool.weight", prefix);
    kt_tensor * pool_b = kt_cpu_named_fmt(ctx, "%s.pool.bias",   prefix);

    assert(n1_fcW != NULL && n2_fcW != NULL
           && c1_w != NULL && c2_w != NULL);

    const int upsample    = (pool_w != NULL);
    const int has_conv1x1 = (sv_w   != NULL);
    const int Cin = (int)x->ne[0];

    kt_tensor * h = kt_ada_in_1d(x, style, n1_fcW, n1_fcB,
                                 n1_nW, n1_nB, Cin);
    h = kt_leaky_relu(h, 0.2f);

    if (upsample) { h = kt_upsample_2x_dwT(h, pool_w, pool_b); }

    h = kt_conv1d_nlc(h, c1_w, c1_b, 1, -1, 1);

    const int Cmid = (int)c1_w->ne[2];
    h = kt_ada_in_1d(h, style, n2_fcW, n2_fcB,
                     n2_nW, n2_nB, Cmid);
    h = kt_leaky_relu(h, 0.2f);

    h = kt_conv1d_nlc(h, c2_w, c2_b, 1, -1, 1);

    kt_tensor * shortcut = shortcut_in != NULL ? shortcut_in : x;
    kt_tensor * res;
    if (upsample) {
        kt_tensor * sup = kt_repeat_interleave_2x_nlc(shortcut);
        res = kt_conv1d_nlc(sup, sv_w, sv_b, 1, 0, 1);
    } else if (has_conv1x1) {
        res = kt_conv1d_nlc(shortcut, sv_w, sv_b, 1, 0, 1);
    } else {
        res = shortcut;
    }

    kt_tensor * out = kt_add(h, res);
    if (divide) { out = kt_scale(out, 1.0f / sqrtf(2.0f)); }
    return out;
}


// ===========================================================================
// AdaINResBlockHiFiGAN (3 dilations, snake activation, residual)
// ===========================================================================
//
// x:(C, L). Returns (C, L). The block has 3 sub-iterations with
// dilations (1, 3, 5); each sub-iteration is two AdaIN+snake+conv
// passes plus a residual add.
// Checkpoint helpers — copy tensor data to malloc'd host memory so we
// can reset the scratch arena between sub-stages without losing the
// state we need next. Without this, every intermediate tensor inside
// build_hifi_block (~21 per dilation × 3 dilations) accumulates in
// scratch and we OOM on iOS for long sentences.
typedef struct {
    int      ndim;
    int64_t  ne[4];
    float *  data;
} kt_savept;

static kt_savept kt_save(const kt_tensor * t) {
    kt_savept p;
    p.ndim = t->ndim;
    for (int i = 0; i < 4; i++) { p.ne[i] = t->ne[i]; }
    int64_t n = p.ne[0] * p.ne[1] * p.ne[2] * p.ne[3];
    p.data = (float *)malloc((size_t)n * sizeof(float));
    memcpy(p.data, t->data, (size_t)n * sizeof(float));
    return p;
}

static kt_tensor * kt_restore(kt_arena * a, const kt_savept * p) {
    kt_tensor * t;
    switch (p->ndim) {
        case 1:  t = kt_new_1d(a, p->ne[0]); break;
        case 2:  t = kt_new_2d(a, p->ne[0], p->ne[1]); break;
        case 3:  t = kt_new_3d(a, p->ne[0], p->ne[1], p->ne[2]); break;
        default: t = kt_new_4d(a, p->ne[0], p->ne[1],
                               p->ne[2], p->ne[3]); break;
    }
    int64_t n = p->ne[0] * p->ne[1] * p->ne[2] * p->ne[3];
    memcpy(t->data, p->data, (size_t)n * sizeof(float));
    return t;
}

static void kt_savept_free(kt_savept * p) {
    if (p && p->data) { free(p->data); p->data = NULL; }
}

static kt_tensor * build_hifi_block(kt_cpu_ctx * ctx, const char * prefix,
                                    kt_tensor * x, kt_tensor * style) {
    kt_arena * sa = ctx->scratch_arena;
    const int C = (int)x->ne[0];
    static const int dilations[3] = { 1, 3, 5 };
    kt_savept ps = kt_save(style);
    kt_tensor * out = x;
    for (int k = 0; k < 3; k++) {
        if (k > 0) {
            kt_savept po = kt_save(out);
            kt_arena_reset(sa);
            out   = kt_restore(sa, &po);  kt_savept_free(&po);
            style = kt_restore(sa, &ps);
        }
        const int d = dilations[k];
        kt_tensor * a1_fcW = kt_cpu_named_fmt(ctx, "%s.a1.%d.fcW", prefix, k);
        kt_tensor * a1_fcB = kt_cpu_named_fmt(ctx, "%s.a1.%d.fcB", prefix, k);
        kt_tensor * a1_nW  = kt_cpu_named_fmt(ctx, "%s.a1.%d.nW",  prefix, k);
        kt_tensor * a1_nB  = kt_cpu_named_fmt(ctx, "%s.a1.%d.nB",  prefix, k);
        kt_tensor * a2_fcW = kt_cpu_named_fmt(ctx, "%s.a2.%d.fcW", prefix, k);
        kt_tensor * a2_fcB = kt_cpu_named_fmt(ctx, "%s.a2.%d.fcB", prefix, k);
        kt_tensor * a2_nW  = kt_cpu_named_fmt(ctx, "%s.a2.%d.nW",  prefix, k);
        kt_tensor * a2_nB  = kt_cpu_named_fmt(ctx, "%s.a2.%d.nB",  prefix, k);
        kt_tensor * al1    = kt_cpu_named_fmt(ctx, "%s.al1.%d",    prefix, k);
        kt_tensor * al2    = kt_cpu_named_fmt(ctx, "%s.al2.%d",    prefix, k);
        kt_tensor * c1_w   = kt_cpu_named_fmt(ctx, "%s.c1.%d.weight", prefix, k);
        kt_tensor * c1_b   = kt_cpu_named_fmt(ctx, "%s.c1.%d.bias",   prefix, k);
        kt_tensor * c2_w   = kt_cpu_named_fmt(ctx, "%s.c2.%d.weight", prefix, k);
        kt_tensor * c2_b   = kt_cpu_named_fmt(ctx, "%s.c2.%d.bias",   prefix, k);

        const int K1 = (int)c1_w->ne[0];
        const int K2 = (int)c2_w->ne[0];

        kt_tensor * h = kt_ada_in_1d(out, style, a1_fcW, a1_fcB,
                                     a1_nW, a1_nB, C);
        h = kt_snake_1d(h, al1);
        h = kt_conv1d_nlc(h, c1_w, c1_b, 1, d * (K1 - 1) / 2, d);
        h = kt_ada_in_1d(h, style, a2_fcW, a2_fcB, a2_nW, a2_nB, C);
        h = kt_snake_1d(h, al2);
        h = kt_conv1d_nlc(h, c2_w, c2_b, 1, (K2 - 1) / 2, 1);
        out = kt_add(out, h);
    }
    kt_savept_free(&ps);
    return out;
}


// ===========================================================================
// BERT/Albert encoder
// ===========================================================================

static kt_tensor * build_albert(kt_cpu_ctx * ctx, int L,
                                const int32_t * ids,
                                const int32_t * pos,
                                const int32_t * type) {
    const kt_cpu_arch * a = &ctx->arch;
    const kt_cpu_weights * W = &ctx->W;
    kt_arena * sa = ctx->scratch_arena;

    kt_tensor * h = kt_get_rows(W->e_word, ids,  L);
    kt_tensor * p = kt_get_rows(W->e_pos,  pos,  L);
    kt_tensor * t = kt_get_rows(W->e_type, type, L);
    h = kt_add(h, p);
    h = kt_add(h, t);
    h = kt_layer_norm(h, W->e_ln_w, W->e_ln_b, a->ln_eps);
    h = kt_mul_mat(W->proj_w, h);
    h = kt_add(h, W->proj_b);

    const float kq_scale = 1.0f / sqrtf((float)a->head_dim);

    for (int il = 0; il < a->n_layers; il++) {
        kt_tensor * residual = h;

        kt_tensor * q = kt_add(kt_mul_mat(W->q_w, h), W->q_b);
        kt_tensor * k = kt_add(kt_mul_mat(W->k_w, h), W->k_b);
        kt_tensor * v = kt_add(kt_mul_mat(W->v_w, h), W->v_b);

        q = kt_reshape_4d(q, a->head_dim, a->n_heads, L, 1);
        k = kt_reshape_4d(k, a->head_dim, a->n_heads, L, 1);
        v = kt_reshape_4d(v, a->head_dim, a->n_heads, L, 1);

        q = kt_permute(q, 0, 2, 1, 3);   // (head_dim, L, n_heads, 1)
        k = kt_permute(k, 0, 2, 1, 3);
        v = kt_permute(v, 0, 2, 1, 3);
        v = kt_cont(kt_transpose(v));    // (L, head_dim, n_heads, 1)

        kt_tensor * kq = kt_mul_mat(k, q);          // (L, L, n_heads, 1)
        kq = kt_softmax(kq, 0, kq_scale);

        kt_tensor * kqv = kt_mul_mat(v, kq);        // (head_dim, L, n_heads, 1)
        kqv = kt_permute(kqv, 0, 2, 1, 3);          // (head_dim, n_heads, L, 1)
        kqv = kt_cont_2d(kqv, a->hidden, L);        // (hidden, L)

        kt_tensor * att_out = kt_add(kt_mul_mat(W->o_w, kqv), W->o_b);

        h = kt_add(att_out, residual);
        h = kt_layer_norm(h, W->attn_ln_w, W->attn_ln_b, a->ln_eps);
        kt_tensor * mid = h;

        kt_tensor * ffn = kt_add(kt_mul_mat(W->ffn_w, h), W->ffn_b);
        ffn = kt_gelu_erf(ffn);
        ffn = kt_add(kt_mul_mat(W->ffn_out_w, ffn), W->ffn_out_b);

        h = kt_add(ffn, mid);
        h = kt_layer_norm(h, W->full_ln_w, W->full_ln_b, a->ln_eps);
    }
    (void)sa;
    return h;
}


// ===========================================================================
// PredictorTextEncoder
// ===========================================================================

static kt_tensor * build_pred_text(kt_cpu_ctx * ctx,
                                   kt_tensor * bert_out, kt_tensor * style,
                                   kt_tensor * h0, kt_tensor * c0, int L) {
    const kt_cpu_weights * W = &ctx->W;
    kt_arena * sa = ctx->scratch_arena;
    const int C = 128;
    const int H = ctx->arch.lstm_hidden;

    kt_tensor * s_bcast = kt_style_bcast_CxL(sa, style, C, L);
    kt_tensor * x = kt_concat(bert_out, s_bcast, 0);  // (384, L)

    kt_tensor * y = kt_bidir_lstm(sa, x,
                                  W->pt_l0_fW, W->pt_l0_fR, W->pt_l0_fb,
                                  W->pt_l0_bW, W->pt_l0_bR, W->pt_l0_bb,
                                  h0, c0, H, L);
    kt_tensor * y1 = kt_ada_layer_norm(y, style, W->pt_fc1_w,
                                       W->pt_fc1_b, C);
    kt_tensor * x2 = kt_concat(y1, s_bcast, 0);       // (256, L)
    kt_tensor * y2 = kt_bidir_lstm(sa, x2,
                                   W->pt_l2_fW, W->pt_l2_fR, W->pt_l2_fb,
                                   W->pt_l2_bW, W->pt_l2_bR, W->pt_l2_bb,
                                   h0, c0, H, L);
    kt_tensor * y3 = kt_ada_layer_norm(y2, style, W->pt_fc3_w,
                                       W->pt_fc3_b, C);
    return y3;
}


// ===========================================================================
// AcousticTextEncoder
// ===========================================================================

static kt_tensor * build_acoustic(kt_cpu_ctx * ctx,
                                  const int32_t * ids,
                                  kt_tensor * h0, kt_tensor * c0, int L) {
    const kt_cpu_weights * W = &ctx->W;
    kt_arena * sa = ctx->scratch_arena;
    const int H = ctx->arch.lstm_hidden;

    kt_tensor * x = kt_get_rows(W->ac_embd, ids, L);  // (128, L) NLC

    for (int i = 0; i < 2; i++) {
        kt_tensor * cnnW = (i == 0) ? W->ac_c0_w : W->ac_c1_w;
        kt_tensor * cnnB = (i == 0) ? W->ac_c0_b : W->ac_c1_b;
        kt_tensor * lnG  = (i == 0) ? W->ac_ln0_g : W->ac_ln1_g;
        kt_tensor * lnB  = (i == 0) ? W->ac_ln0_b : W->ac_ln1_b;
        const int K = (int)cnnW->ne[0];
        const int pad = (K - 1) / 2;

        kt_tensor * x_ncl = kt_cont(kt_transpose(x));
        kt_tensor * x_ncl_3d = kt_reshape_3d(x_ncl,
                                             x_ncl->ne[0],
                                             x_ncl->ne[1], 1);
        kt_tensor * y_3d = kt_conv_1d(cnnW, x_ncl_3d, 1, pad, 1);
        kt_tensor * b3 = kt_reshape_3d(cnnB, 1, cnnB->ne[0], 1);
        y_3d = kt_add(y_3d, b3);

        kt_tensor * y_2d = kt_reshape_2d(y_3d, y_3d->ne[0], y_3d->ne[1]);
        x = kt_cont(kt_transpose(y_2d));
        x = kt_layer_norm(x, lnG, lnB, 1e-5f);
        x = kt_leaky_relu(x, 0.2f);
    }

    kt_tensor * y = kt_bidir_lstm(sa, x,
                                  W->ac_l_fW, W->ac_l_fR, W->ac_l_fb,
                                  W->ac_l_bW, W->ac_l_bR, W->ac_l_bb,
                                  h0, c0, H, L);
    return y;
}


// ===========================================================================
// TextStage orchestrator
// ===========================================================================

typedef struct {
    kt_tensor * prosody256;
    kt_tensor * text;
    kt_tensor * dur_sig;
} kt_textstage_outs;

static kt_textstage_outs build_textstage(kt_cpu_ctx * ctx, int L,
                                         const int32_t * ids,
                                         const int32_t * pos,
                                         const int32_t * type,
                                         kt_tensor * style_pr,
                                         kt_tensor * h0, kt_tensor * c0) {
    const kt_cpu_weights * W = &ctx->W;
    kt_arena * sa = ctx->scratch_arena;

    kt_tensor * bert = build_albert(ctx, L, ids, pos, type);
    kt_tensor * bert_proj = kt_add(kt_mul_mat(W->bert_enc_w, bert),
                                   W->bert_enc_b);   // (256, L)
    kt_tensor * prosody = build_pred_text(ctx, bert_proj, style_pr,
                                          h0, c0, L);   // (128, L)

    kt_tensor * s_bcast = kt_style_bcast_CxL(sa, style_pr,
                                             ctx->arch.style_dim, L);
    kt_tensor * prosody256 = kt_concat(prosody, s_bcast, 0);

    kt_tensor * dlstm = kt_bidir_lstm(sa, prosody256,
                                      W->dur_l_fW, W->dur_l_fR, W->dur_l_fb,
                                      W->dur_l_bW, W->dur_l_bR, W->dur_l_bb,
                                      h0, c0, ctx->arch.lstm_hidden, L);
    kt_tensor * dur_logits = kt_add(kt_mul_mat(W->dur_w, dlstm),
                                    W->dur_b);
    kt_tensor * dur_sig = kt_sigmoid(dur_logits);

    kt_tensor * text = build_acoustic(ctx, ids, h0, c0, L);

    kt_textstage_outs r = { prosody256, text, dur_sig };
    return r;
}


// ===========================================================================
// GenFront: shared LSTM + 6 AdaINResBlock1D + f0_proj / n_proj
// ===========================================================================

typedef struct {
    kt_tensor * f0_proj;   // (1, 2F)
    kt_tensor * n_proj;    // (1, 2F)
} kt_genfront_outs;

static kt_genfront_outs build_genfront(kt_cpu_ctx * ctx,
                                       kt_tensor * prosody_lr_nlc,
                                       kt_tensor * style,
                                       kt_tensor * h0, kt_tensor * c0,
                                       int F) {
    const kt_cpu_weights * W = &ctx->W;
    kt_arena * sa = ctx->scratch_arena;
    const int H = ctx->arch.lstm_hidden;

    kt_tensor * sh = kt_bidir_lstm(sa, prosody_lr_nlc,
                                   W->sh_fW, W->sh_fR, W->sh_fb,
                                   W->sh_bW, W->sh_bR, W->sh_bb,
                                   h0, c0, H, F);

    kt_tensor * f0 = build_ada_block_1d(ctx, "f0.0", sh, style, sh, 1);
    f0 = build_ada_block_1d(ctx, "f0.1", f0, style, f0, 1);
    f0 = build_ada_block_1d(ctx, "f0.2", f0, style, f0, 1);
    kt_tensor * f0_proj_w = kt_cpu_named_fmt(ctx, "f0_proj.weight");
    kt_tensor * f0_proj_b = kt_cpu_named_fmt(ctx, "f0_proj.bias");
    kt_tensor * f0p = kt_conv1d_nlc(f0, f0_proj_w, f0_proj_b, 1, -1, 1);

    kt_tensor * nx = build_ada_block_1d(ctx, "n.0", sh, style, sh, 1);
    nx = build_ada_block_1d(ctx, "n.1", nx, style, nx, 1);
    nx = build_ada_block_1d(ctx, "n.2", nx, style, nx, 1);
    kt_tensor * n_proj_w = kt_cpu_named_fmt(ctx, "n_proj.weight");
    kt_tensor * n_proj_b = kt_cpu_named_fmt(ctx, "n_proj.bias");
    kt_tensor * np = kt_conv1d_nlc(nx, n_proj_w, n_proj_b, 1, -1, 1);

    kt_genfront_outs r = { f0p, np };
    return r;
}


// ===========================================================================
// Decoder
// ===========================================================================

static kt_tensor * build_decoder(kt_cpu_ctx * ctx,
                                 kt_tensor * text_lr, kt_tensor * f0_proj,
                                 kt_tensor * n_proj, kt_tensor * style_aco) {
    kt_tensor * asrW = kt_cpu_named_fmt(ctx, "dec.asr.weight");
    kt_tensor * asrB = kt_cpu_named_fmt(ctx, "dec.asr.bias");
    kt_tensor * f0W  = kt_cpu_named_fmt(ctx, "dec.f0_conv.weight");
    kt_tensor * f0B  = kt_cpu_named_fmt(ctx, "dec.f0_conv.bias");
    kt_tensor * nW   = kt_cpu_named_fmt(ctx, "dec.n_conv.weight");
    kt_tensor * nB   = kt_cpu_named_fmt(ctx, "dec.n_conv.bias");

    kt_tensor * asr   = kt_conv1d_nlc(text_lr, asrW, asrB, 1, 0, 1);
    kt_tensor * f0_dn = kt_conv1d_nlc(f0_proj, f0W, f0B, 2, 1, 1);
    kt_tensor * n_dn  = kt_conv1d_nlc(n_proj,  nW,  nB,  2, 1, 1);

    kt_tensor * enc_in = kt_concat(text_lr, f0_dn, 0);
    enc_in = kt_concat(enc_in, n_dn, 0);

    kt_tensor * x = build_ada_block_1d(ctx, "dec.encode", enc_in,
                                       style_aco, enc_in, 1);
    for (int i = 0; i < 4; i++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "dec.decode.%d", i);
        kt_tensor * x_cat = kt_concat(x, asr, 0);
        x_cat = kt_concat(x_cat, f0_dn, 0);
        x_cat = kt_concat(x_cat, n_dn,  0);
        x = build_ada_block_1d(ctx, prefix, x_cat, style_aco, x_cat, 1);
    }
    return x;
}


// ===========================================================================
// Noise contributions (sine excitation + STFT analysis + noise_res blocks)
// ===========================================================================

// Conv1D on NLC tensor with F32 kernel (no F16 path). The stored STFT
// kernels are F32 and kt_conv1d_nlc would handle them fine, but the
// ggml original used a custom F32 im2col path; kt always runs F32 so
// this is just an alias.
static kt_tensor * kt_conv1d_nlc_f32k(kt_tensor * x, kt_tensor * kw,
                                      kt_tensor * kb,
                                      int stride, int pad, int dilation) {
    return kt_conv1d_nlc(x, kw, kb, stride, pad, dilation);
}

typedef struct { kt_tensor * nr0; kt_tensor * nr1; } kt_noise_outs;

static kt_noise_outs build_noise_contribs(kt_cpu_ctx * ctx,
                                          kt_tensor * f0_proj,
                                          kt_tensor * style_aco,
                                          kt_tensor * harmonics,
                                          kt_tensor * s_range,
                                          kt_tensor * eps_t,
                                          int F) {
    kt_arena * sa = ctx->scratch_arena;
    const int T_frames = 2 * F;
    const int hop      = 300;
    const int T_audio  = T_frames * hop;
    const float sr     = 24000.0f;
    const float two_pi = 2.0f * (float)M_PI;

    (void)sa;
    // f0_audio: nearest-neighbor upsample (1, 2F) -> (1, T_audio).
    kt_tensor * f0_3d  = kt_reshape_3d(f0_proj, 1, 1, T_frames);
    kt_tensor * f0_audio_3d = kt_repeat_to(f0_3d, 3,
                                           1, hop, T_frames, 1);
    kt_tensor * f0_audio = kt_reshape_2d(f0_audio_3d, 1, T_audio);

    kt_tensor * voiced = kt_step(f0_audio);

    // f0_per_frame[h, t] = f0_proj[t] * (h+1). Repeat (1, 2F) -> (9, 2F).
    kt_tensor * f0_repeated = kt_repeat_to(f0_proj, 2,
                                           9, T_frames, 1, 1);
    kt_tensor * harm_2d = kt_reshape_2d(harmonics, 9, 1);
    kt_tensor * f0_per_frame = kt_mul(f0_repeated, harm_2d);

    // step = f0_per_frame * (hop/sr); phase_start = (cumsum-step)*2π.
    kt_tensor * step_nlc = kt_scale(f0_per_frame, (float)hop / sr);
    kt_tensor * step_ncl = kt_cont(kt_transpose(step_nlc));    // (2F, 9)
    kt_tensor * cs       = kt_cumsum(step_ncl, 0);
    kt_tensor * ps_ncl   = kt_sub(cs, step_ncl);
    ps_ncl = kt_scale(ps_ncl, two_pi);
    kt_tensor * phase_start_nlc = kt_cont(kt_transpose(ps_ncl));

    // phase_within[h, t, s] = f0_per_frame[h, t] * s * (2π/sr)
    kt_tensor * fpf_3d = kt_reshape_3d(f0_per_frame, 9, T_frames, 1);
    kt_tensor * s_3d   = kt_reshape_3d(s_range, 1, 1, hop);
    kt_tensor * fpf_x  = kt_repeat_to(fpf_3d, 3, 9, T_frames, hop, 1);
    kt_tensor * s_x    = kt_repeat_to(s_3d,   3, 9, T_frames, hop, 1);
    kt_tensor * within = kt_mul(fpf_x, s_x);
    within = kt_scale(within, two_pi / sr);

    kt_tensor * ps_3d = kt_reshape_3d(phase_start_nlc, 9, T_frames, 1);
    kt_tensor * ps_expanded = kt_repeat_to(ps_3d, 3,
                                           9, T_frames, hop, 1);
    kt_tensor * phase = kt_add(ps_expanded, within);

    phase = kt_permute(phase, 0, 2, 1, 3);   // (9, hop, 2F)
    phase = kt_cont(phase);
    phase = kt_reshape_2d(phase, 9, T_audio);

    kt_tensor * sines = kt_scale(kt_sin(phase), 0.1f);
    kt_tensor * sin_gen = kt_mul(sines, voiced);

    kt_tensor * l_lin_w = kt_cpu_named_fmt(ctx, "l_lin.weight");
    kt_tensor * l_lin_b = kt_cpu_named_fmt(ctx, "l_lin.bias");
    kt_tensor * mixed = kt_add(kt_mul_mat(l_lin_w, sin_gen), l_lin_b);
    kt_tensor * excitation = kt_tanh(mixed);

    kt_tensor * stft_fr = kt_cpu_named_fmt(ctx, "stft_fwd.real");
    kt_tensor * stft_fi = kt_cpu_named_fmt(ctx, "stft_fwd.imag");
    kt_tensor * stft_real = kt_conv1d_nlc_f32k(excitation, stft_fr,
                                               NULL, 5, 10, 1);
    kt_tensor * stft_imag = kt_conv1d_nlc_f32k(excitation, stft_fi,
                                               NULL, 5, 10, 1);

    kt_tensor * re2 = kt_mul(stft_real, stft_real);
    kt_tensor * im2 = kt_mul(stft_imag, stft_imag);
    kt_tensor * mag2 = kt_add(re2, im2);
    mag2 = kt_add(mag2, eps_t);
    kt_tensor * mag = kt_sqrt(mag2);

    kt_tensor * phi = kt_atan2(stft_imag, stft_real);

    kt_tensor * stft_out = kt_concat(mag, phi, 0);

    kt_tensor * nc0_w = kt_cpu_named_fmt(ctx, "nc0.weight");
    kt_tensor * nc0_b = kt_cpu_named_fmt(ctx, "nc0.bias");
    kt_tensor * nc1_w = kt_cpu_named_fmt(ctx, "nc1.weight");
    kt_tensor * nc1_b = kt_cpu_named_fmt(ctx, "nc1.bias");
    kt_tensor * nc0 = kt_conv1d_nlc(stft_out, nc0_w, nc0_b, 6, 3, 1);
    kt_tensor * nc1 = kt_conv1d_nlc(stft_out, nc1_w, nc1_b, 1, 0, 1);

    // build_hifi_block resets the arena internally — save what we
    // need to survive the call. style_aco is needed by both hifi
    // calls; nc1 is needed by the second one; nr0 is needed after
    // both. Each restore creates a fresh tensor in arena from the
    // saved host buffer.
    kt_savept p_nc1 = kt_save(nc1);
    kt_savept p_sty = kt_save(style_aco);
    kt_tensor * nr0 = build_hifi_block(ctx, "nr0", nc0, style_aco);
    kt_savept p_nr0 = kt_save(nr0);
    nc1       = kt_restore(sa, &p_nc1);
    style_aco = kt_restore(sa, &p_sty);
    kt_tensor * nr1 = build_hifi_block(ctx, "nr1", nc1, style_aco);
    nr0 = kt_restore(sa, &p_nr0);
    kt_savept_free(&p_nc1);
    kt_savept_free(&p_sty);
    kt_savept_free(&p_nr0);

    kt_noise_outs r = { nr0, nr1 };
    return r;
}


// ===========================================================================
// Generator (HiFi-GAN-style upsamplers + ResBlocks + iSTFT head)
// ===========================================================================

static kt_tensor * build_generator(kt_cpu_ctx * ctx,
                                   kt_tensor * dec_out,
                                   kt_tensor * nr0, kt_tensor * nr1,
                                   kt_tensor * style_aco) {
    kt_tensor * u0_w = kt_cpu_named_fmt(ctx, "gen.u0.weight");
    kt_tensor * u0_b = kt_cpu_named_fmt(ctx, "gen.u0.bias");
    kt_tensor * u1_w = kt_cpu_named_fmt(ctx, "gen.u1.weight");
    kt_tensor * u1_b = kt_cpu_named_fmt(ctx, "gen.u1.bias");
    kt_tensor * cp_w = kt_cpu_named_fmt(ctx, "gen.cp.weight");
    kt_tensor * cp_b = kt_cpu_named_fmt(ctx, "gen.cp.bias");
    kt_tensor * sb_r = kt_cpu_named_fmt(ctx, "stft_bwd.real");
    kt_tensor * sb_i = kt_cpu_named_fmt(ctx, "stft_bwd.imag");

    kt_arena * sa = ctx->scratch_arena;

    // build_hifi_block resets the arena internally; every input that
    // the caller still needs after a hifi call (or another caller-
    // visible reset) must be snapshotted to host memory FIRST, then
    // restored. Snapshot style_aco and nr1 once at entry — both are
    // referenced multiple times after the first reset. nr0 is used
    // before any reset and doesn't need saving.
    kt_savept p_sty = kt_save(style_aco);
    kt_savept p_nr1 = kt_save(nr1);

    kt_tensor * x = kt_leaky_relu(dec_out, 0.1f);
    x = kt_conv_transpose_1d_nlc(x, u0_w, u0_b, 10, 5);
    x = kt_add(x, nr0);

    kt_savept px, pr;

    px = kt_save(x);
    kt_tensor * r0 = build_hifi_block(ctx, "gen.r0", x, style_aco);
    pr = kt_save(r0);
    x         = kt_restore(sa, &px);
    style_aco = kt_restore(sa, &p_sty);
    kt_tensor * r1 = build_hifi_block(ctx, "gen.r1", x, style_aco);
    r0 = kt_restore(sa, &pr);
    x = kt_scale(kt_add(r0, r1), 0.5f);
    x = kt_leaky_relu(x, 0.1f);
    kt_savept_free(&px); kt_savept_free(&pr);

    x = kt_conv_transpose_1d_nlc(x, u1_w, u1_b, 6, 3);
    x = kt_reflection_pad_left(x, 1);
    nr1 = kt_restore(sa, &p_nr1);
    style_aco = kt_restore(sa, &p_sty);
    x = kt_add(x, nr1);

    px = kt_save(x);
    kt_tensor * r2 = build_hifi_block(ctx, "gen.r2", x, style_aco);
    pr = kt_save(r2);
    x         = kt_restore(sa, &px);
    style_aco = kt_restore(sa, &p_sty);
    kt_tensor * r3 = build_hifi_block(ctx, "gen.r3", x, style_aco);
    r2 = kt_restore(sa, &pr);
    x = kt_scale(kt_add(r2, r3), 0.5f);
    x = kt_leaky_relu(x, 0.1f);
    kt_savept_free(&px); kt_savept_free(&pr);
    kt_savept_free(&p_nr1);
    kt_savept_free(&p_sty);

    x = kt_conv1d_nlc(x, cp_w, cp_b, 1, -1, 1);

    // iSTFT head: mag = exp(x[0:11, :]); inner = sin(x[11:22, :]);
    // real-and-imag conv-transposes; audio = audio_r - audio_i; trim.
    const int64_t L = x->ne[1];
    kt_tensor * mag_log = kt_view_2d(x, 11, L, (size_t)x->nb[1], 0);
    kt_tensor * phase   = kt_view_2d(x, 11, L,
                                     (size_t)x->nb[1],
                                     (size_t)11 * x->nb[0]);
    mag_log = kt_cont(mag_log);
    phase   = kt_cont(phase);
    kt_tensor * mag = kt_exp(mag_log);
    kt_tensor * inner = kt_sin(phase);
    kt_tensor * re = kt_mul(mag, kt_cos(inner));
    kt_tensor * im = kt_mul(mag, kt_sin(inner));

    kt_tensor * audio_r = kt_conv_transpose_1d_nlc(re, sb_r, NULL, 5, 0);
    kt_tensor * audio_i = kt_conv_transpose_1d_nlc(im, sb_i, NULL, 5, 0);

    kt_tensor * audio = kt_sub(audio_r, audio_i);    // (1, T_audio)

    const int trim = ctx->arch.istft_trim;
    const int64_t T = audio->ne[1];
    assert(T > 2 * trim);
    return kt_cont(kt_view_1d(audio, T - 2 * trim,
                              (size_t)trim * sizeof(float)));
}


// ===========================================================================
// Fade helpers (host-side)
// ===========================================================================

static void kt_fade_in(float * x, int n, int fade) {
    if (fade > 0 && fade <= n) {
        for (int i = 0; i < fade; i++) {
            const float t = (float)i / (float)(fade - 1 > 0 ? fade - 1 : 1);
            x[i] *= 0.5f - 0.5f * cosf((float)M_PI * t);
        }
    }
}

static void kt_fade_out(float * x, int n, int fade) {
    if (fade > 0 && fade <= n) {
        const int start = n - fade;
        for (int i = 0; i < fade; i++) {
            const float t = (float)i / (float)(fade - 1 > 0 ? fade - 1 : 1);
            x[start + i] *= 0.5f + 0.5f * cosf((float)M_PI * t);
        }
    }
}


// ===========================================================================
// Public kt_cpu_synthesize: full pipeline with length regulation
// ===========================================================================

kt_cpu_audio kt_cpu_synthesize(kt_cpu_ctx * ctx,
                               const int32_t * phonemes,
                               int n_phonemes,
                               const float * style256,
                               float speed) {
    kt_cpu_audio out = { NULL, 0 };
    if (ctx == NULL || phonemes == NULL || n_phonemes <= 0
        || style256 == NULL || speed <= 0.0f) {
        return out;
    }
    const int L = n_phonemes;
    const kt_cpu_arch * A = &ctx->arch;
    kt_arena * sa = ctx->scratch_arena;

    // CRITICAL: redirect ALL op outputs (kt_mul_mat, kt_add, kt_get_rows,
    // every conv, etc.) into the scratch arena. Without this, ops whose
    // first input is a model weight (q_w, ffn_w, embedding tables, ...)
    // allocate their output in weights_arena — which is never reset and
    // grows by ~100-200 MB per synthesize call until the process is OOM
    // killed. See cpu/kt_tensor.c::kt_arena_set_active.
    kt_arena_set_active(sa);

    // Host buffers we hold across arena resets.
    int   * durs = NULL;
    float * prosody_h = NULL, * text_h = NULL, * dur_h = NULL;
    float * prosody_lr_h = NULL, * text_lr_h = NULL;
    float * f0p_h = NULL, * np_h = NULL, * dec_h = NULL;
    float * audio_buf = NULL;

    // ---- Stage 1: TextStage ----
    kt_arena_reset(sa);
    {
        int32_t * pos_ids  = (int32_t *)malloc(sizeof(int32_t) * L);
        int32_t * type_ids = (int32_t *)malloc(sizeof(int32_t) * L);
        // Position embeddings only have max_pos rows. Long paragraphs
        // (heavy with "…", "—", etc.) can phonemize past that ceiling;
        // clamp to max_pos-1 so the model degrades gracefully rather
        // than asserting in kt_get_rows. Quality past max_pos drops
        // (every later token shares the last position embedding) but
        // playback continues. Swift-side chunking should prevent us
        // from getting here in the first place.
        const int max_p = A->max_pos > 0 ? A->max_pos - 1 : 0;
        for (int i = 0; i < L; i++) {
            pos_ids[i]  = i <= max_p ? i : max_p;
            type_ids[i] = 0;
        }

        kt_tensor * style_pr = kt_new_1d(sa, A->style_dim);
        memcpy(style_pr->data, style256 + A->style_dim,
               sizeof(float) * A->style_dim);
        kt_tensor * h0 = kt_new_1d(sa, A->lstm_hidden);
        kt_tensor * c0 = kt_new_1d(sa, A->lstm_hidden);
        memset(h0->data, 0, sizeof(float) * A->lstm_hidden);
        memset(c0->data, 0, sizeof(float) * A->lstm_hidden);

        kt_textstage_outs ts = build_textstage(ctx, L,
                                               phonemes, pos_ids, type_ids,
                                               style_pr, h0, c0);
        free(pos_ids); free(type_ids);

        prosody_h = (float *)malloc(sizeof(float) * 256 * L);
        text_h    = (float *)malloc(sizeof(float) * 128 * L);
        dur_h     = (float *)malloc(sizeof(float) *  50 * L);
        memcpy(prosody_h, ts.prosody256->data, sizeof(float) * 256 * L);
        memcpy(text_h,    ts.text->data,       sizeof(float) * 128 * L);
        memcpy(dur_h,     ts.dur_sig->data,    sizeof(float) *  50 * L);
    }

    // ---- Length regulation ----
    durs = (int *)malloc(sizeof(int) * L);
    int F = 0;
    for (int i = 0; i < L; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 50; j++) {
            sum += dur_h[i * 50 + j];
        }
        int d = (int)lrintf(sum / speed);
        if (d < 1) { d = 1; }
        durs[i] = d;
        F += d;
    }

    prosody_lr_h = (float *)calloc((size_t)256 * F, sizeof(float));
    text_lr_h    = (float *)calloc((size_t)128 * F, sizeof(float));
    {
        int t = 0;
        for (int l = 0; l < L; l++) {
            const int d = durs[l];
            for (int k = 0; k < d; k++, t++) {
                memcpy(prosody_lr_h + (size_t)t * 256,
                       prosody_h    + (size_t)l * 256,
                       sizeof(float) * 256);
                memcpy(text_lr_h    + (size_t)t * 128,
                       text_h       + (size_t)l * 128,
                       sizeof(float) * 128);
            }
        }
    }
    free(prosody_h); prosody_h = NULL;
    free(text_h);    text_h    = NULL;
    free(dur_h);     dur_h     = NULL;
    free(durs);      durs      = NULL;

    // ---- Stage 2: GenFront ----
    kt_arena_reset(sa);
    {
        kt_tensor * prosody_lr = kt_new_2d(sa, 256, F);
        memcpy(prosody_lr->data, prosody_lr_h,
               sizeof(float) * 256 * F);
        kt_tensor * style_pr = kt_new_1d(sa, A->style_dim);
        memcpy(style_pr->data, style256 + A->style_dim,
               sizeof(float) * A->style_dim);
        kt_tensor * h0 = kt_new_1d(sa, A->lstm_hidden);
        kt_tensor * c0 = kt_new_1d(sa, A->lstm_hidden);
        memset(h0->data, 0, sizeof(float) * A->lstm_hidden);
        memset(c0->data, 0, sizeof(float) * A->lstm_hidden);

        kt_genfront_outs g = build_genfront(ctx, prosody_lr, style_pr,
                                            h0, c0, F);
        f0p_h = (float *)malloc(sizeof(float) * 2 * F);
        np_h  = (float *)malloc(sizeof(float) * 2 * F);
        memcpy(f0p_h, g.f0_proj->data, sizeof(float) * 2 * F);
        memcpy(np_h,  g.n_proj->data,  sizeof(float) * 2 * F);
    }
    free(prosody_lr_h); prosody_lr_h = NULL;

    // ---- Stage 3: Decoder ----
    kt_arena_reset(sa);
    {
        kt_tensor * text_lr = kt_new_2d(sa, 128, F);
        kt_tensor * f0p_t   = kt_new_2d(sa, 1,   2 * F);
        kt_tensor * np_t    = kt_new_2d(sa, 1,   2 * F);
        kt_tensor * style_a = kt_new_1d(sa, 128);
        memcpy(text_lr->data, text_lr_h, sizeof(float) * 128 * F);
        memcpy(f0p_t->data,   f0p_h,     sizeof(float) * 2 * F);
        memcpy(np_t->data,    np_h,      sizeof(float) * 2 * F);
        memcpy(style_a->data, style256,  sizeof(float) * 128);

        kt_tensor * dec_out = build_decoder(ctx, text_lr, f0p_t, np_t,
                                            style_a);
        dec_h = (float *)malloc(sizeof(float) * 256 * 2 * F);
        memcpy(dec_h, dec_out->data, sizeof(float) * 256 * 2 * F);
    }
    free(text_lr_h); text_lr_h = NULL;
    free(np_h);      np_h      = NULL;

    // ---- Stage 4a: Noise contributions ----
    // Run noise_contribs to completion, copy nr0/nr1 to host buffers,
    // then reset the arena so all the noise intermediates (sine
    // generator, STFT, magnitude, phase, etc.) die before the
    // generator's HiFi blocks start. Cuts peak by ~150-200 MB at
    // long F.
    float * nr0_h = NULL, * nr1_h = NULL;
    int64_t nr0_ne[4] = {0,0,0,0}, nr1_ne[4] = {0,0,0,0};
    int     nr0_nd = 0, nr1_nd = 0;
    kt_arena_reset(sa);
    {
        kt_tensor * f0_t  = kt_new_2d(sa, 1,   2 * F);
        kt_tensor * sty_a = kt_new_1d(sa, 128);
        kt_tensor * harm  = kt_new_1d(sa, 9);
        kt_tensor * s_rng = kt_new_1d(sa, 300);
        kt_tensor * eps_t = kt_new_1d(sa, 1);
        memcpy(f0_t->data,  f0p_h,     sizeof(float) * 2 * F);
        memcpy(sty_a->data, style256,  sizeof(float) * 128);
        for (int i = 0; i < 9;   i++) { harm->data[i]  = (float)(i + 1); }
        for (int i = 0; i < 300; i++) { s_rng->data[i] = (float)i; }
        eps_t->data[0] = 1e-9f;

        kt_noise_outs nz = build_noise_contribs(ctx, f0_t, sty_a,
                                                harm, s_rng, eps_t, F);
        nr0_nd = nz.nr0->ndim;
        nr1_nd = nz.nr1->ndim;
        for (int i = 0; i < 4; i++) {
            nr0_ne[i] = nz.nr0->ne[i];
            nr1_ne[i] = nz.nr1->ne[i];
        }
        int64_t nr0_n = nr0_ne[0]*nr0_ne[1]*nr0_ne[2]*nr0_ne[3];
        int64_t nr1_n = nr1_ne[0]*nr1_ne[1]*nr1_ne[2]*nr1_ne[3];
        nr0_h = (float *)malloc((size_t)nr0_n * sizeof(float));
        nr1_h = (float *)malloc((size_t)nr1_n * sizeof(float));
        memcpy(nr0_h, nz.nr0->data, (size_t)nr0_n * sizeof(float));
        memcpy(nr1_h, nz.nr1->data, (size_t)nr1_n * sizeof(float));
    }

    // ---- Stage 4b: Generator + iSTFT ----
    int n = 0;
    kt_arena_reset(sa);
    {
        kt_tensor * dec_t = kt_new_2d(sa, 256, 2 * F);
        kt_tensor * sty_a = kt_new_1d(sa, 128);
        memcpy(dec_t->data, dec_h,     sizeof(float) * 256 * 2 * F);
        memcpy(sty_a->data, style256,  sizeof(float) * 128);

        // Recreate nr0 / nr1 as kt tensors in the (now-fresh) arena.
        kt_tensor * nr0;
        if (nr0_nd == 2) { nr0 = kt_new_2d(sa, nr0_ne[0], nr0_ne[1]); }
        else             { nr0 = kt_new_3d(sa, nr0_ne[0],
                                            nr0_ne[1], nr0_ne[2]); }
        memcpy(nr0->data, nr0_h, (size_t)(nr0_ne[0]*nr0_ne[1]
                                          *nr0_ne[2]*nr0_ne[3])
                                          * sizeof(float));
        kt_tensor * nr1;
        if (nr1_nd == 2) { nr1 = kt_new_2d(sa, nr1_ne[0], nr1_ne[1]); }
        else             { nr1 = kt_new_3d(sa, nr1_ne[0],
                                            nr1_ne[1], nr1_ne[2]); }
        memcpy(nr1->data, nr1_h, (size_t)(nr1_ne[0]*nr1_ne[1]
                                          *nr1_ne[2]*nr1_ne[3])
                                          * sizeof(float));

        kt_tensor * audio_t = build_generator(ctx, dec_t, nr0, nr1,
                                              sty_a);
        const int T_audio = (int)audio_t->ne[0];
        // Tail-drop 3 frames × 600 samples
        n = T_audio;
        const int tail_drop = 3 * 600;
        if (n > tail_drop) { n -= tail_drop; }
        audio_buf = (float *)malloc(sizeof(float) * (size_t)n);
        memcpy(audio_buf, audio_t->data, sizeof(float) * (size_t)n);
    }
    free(nr0_h); free(nr1_h);
    free(f0p_h); f0p_h = NULL;
    free(dec_h); dec_h = NULL;

    // Fade in 3 ms, fade out 40 ms.
    kt_fade_in (audio_buf, n,  72);
    kt_fade_out(audio_buf, n, 960);

    out.samples   = audio_buf;
    out.n_samples = (uint64_t)n;
    kt_arena_set_active(NULL);
    // Release the per-call peak. Without this, the doubling slab
    // chain (1+2+4+...+512+1024 MB for a long-sentence stage-4 peak)
    // stays resident until the next synthesize call's stage-1 reset,
    // which the OS reports as ~2-3 GB of RSS between sentences.
    kt_arena_reset(sa);
    return out;
}

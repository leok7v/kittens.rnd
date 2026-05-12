// test_gguf.c -- end-to-end verification that the kt_gguf reader can
// open kitten_full.gguf, read every KV that kittens-tts.c needs, find
// every named tensor it binds, and report consistent shapes.
//
// Also cross-checks tensor data against ggml's reader: loads the same
// tensor by name on both sides, compares bytes.

#include "kt_gguf.h"
#include "kt_tensor.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

static void fail(const char * msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    g_failures++;
}

static const char * arch_keys_u32[] = {
    "kittens-tts.vocab_size",
    "kittens-tts.max_position",
    "kittens-tts.token_types",
    "kittens-tts.embedding_dim",
    "kittens-tts.hidden_size",
    "kittens-tts.num_layers",
    "kittens-tts.num_heads",
    "kittens-tts.head_dim",
    "kittens-tts.ffn_dim",
    "kittens-tts.bert_enc_dim",
    "kittens-tts.style_dim",
    "kittens-tts.lstm_hidden",
    "kittens-tts.dur_logits",
    "kittens-tts.audio_per_frame",
    "kittens-tts.istft_hop",
    "kittens-tts.istft_trim",
    NULL,
};

static const char * top_level_tensor_names[] = {
    "embd.word.weight",
    "embd.pos.weight",
    "embd.type.weight",
    "embd.ln.weight",
    "embd.ln.bias",
    "embd_to_hidden.weight",
    "embd_to_hidden.bias",
    "layer.attn_q.weight",
    "layer.attn_q.bias",
    "layer.attn_k.weight",
    "layer.attn_k.bias",
    "layer.attn_v.weight",
    "layer.attn_v.bias",
    "layer.attn_out.weight",
    "layer.attn_out.bias",
    "layer.attn_ln.weight",
    "layer.attn_ln.bias",
    "layer.ffn.weight",
    "layer.ffn.bias",
    "layer.ffn_out.weight",
    "layer.ffn_out.bias",
    "layer.full_ln.weight",
    "layer.full_ln.bias",
    "bert_enc.weight",
    "bert_enc.bias",
    "pred_text.lstm0.fwd.W",
    "pred_text.lstm0.fwd.R",
    "pred_text.lstm0.fwd.b",
    "pred_text.fc1.weight",
    "dur.lstm.fwd.W",
    "dur_proj.weight",
    "acoustic.embd.weight",
    "acoustic.cnn0.weight",
    "acoustic.ln0.gamma",
    "shared.lstm.fwd.W",
    "f0.0.c1.weight",
    "f0.0.n1.fcW",
    "n.1.pool.weight",
    "n_proj.weight",
    "dec.encode.c1.weight",
    "dec.decode.3.pool.weight",
    "dec.asr.weight",
    "dec.f0_conv.weight",
    "dec.n_conv.weight",
    "gen.u0.weight",
    "gen.u1.weight",
    "gen.cp.weight",
    "gen.r0.c1.0.weight",
    "gen.r0.al1.0",
    "nc0.weight",
    "nc1.weight",
    "nr0.c1.0.weight",
    "nr1.c1.0.weight",
    "l_lin.weight",
    "l_lin.bias",
    "stft_fwd.real",
    "stft_fwd.imag",
    "stft_bwd.real",
    "stft_bwd.imag",
    NULL,
};

int main(int argc, char ** argv) {
    const char * path = argc > 1
        ? argv[1]
        : "app/Resources/nano/kitten_full.gguf";
    printf("opening %s\n", path);
    kt_gguf * g = kt_gguf_open(path);
    if (g == NULL) {
        fprintf(stderr, "kt_gguf_open failed: %s\n",
                kt_gguf_last_error());
        return 1;
    }
    int n_t = kt_gguf_n_tensors(g);
    printf("%d tensors total\n", n_t);
    // 1. Arch KV presence + value parity vs ggml.
    struct ggml_context * gctx = NULL;
    struct gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &gctx };
    struct gguf_context * ggu = gguf_init_from_file(path, gp);
    if (!ggu) { fail("ggml gguf_init_from_file failed"); return 1; }
    for (int i = 0; arch_keys_u32[i] != NULL; i++) {
        uint32_t kt_v = 0;
        int got = kt_gguf_get_u32(g, arch_keys_u32[i], &kt_v);
        int ggidx = (int)gguf_find_key(ggu, arch_keys_u32[i]);
        uint32_t gg_v = (ggidx >= 0)
            ? gguf_get_val_u32(ggu, ggidx) : 0;
        if (!got) {
            fprintf(stderr, "FAIL key missing: %s\n", arch_keys_u32[i]);
            g_failures++;
        } else if (kt_v != gg_v) {
            fprintf(stderr, "FAIL %s: kt=%u ggml=%u\n",
                    arch_keys_u32[i], kt_v, gg_v);
            g_failures++;
        } else {
            printf("  %-30s = %u\n", arch_keys_u32[i], kt_v);
        }
    }
    // 2. Tensor-name presence: every name kittens-tts.c binds via
    //    find(c, ...) must exist in our reader.
    int n_named = 0; int n_named_ok = 0;
    for (int i = 0; top_level_tensor_names[i] != NULL; i++) {
        n_named++;
        kt_arena * a = kt_arena_new(64);
        kt_tensor * t = kt_gguf_load_tensor(g, a, top_level_tensor_names[i]);
        if (t == NULL) {
            fprintf(stderr, "FAIL missing tensor: %s\n",
                    top_level_tensor_names[i]);
            g_failures++;
        } else {
            n_named_ok++;
        }
        kt_arena_free(a);
    }
    printf("named tensors: %d/%d present\n", n_named_ok, n_named);
    // 3. Tensor data parity: pick a handful and compare element-by-
    //    element vs the ggml reader. We focus on F32 tensors that we
    //    wrap-without-copy, and a F16 tensor we dequant.
    const char * spot_check[] = {
        "embd.word.weight",
        "embd_to_hidden.weight",
        "layer.attn_q.weight",    // F16 — exercises dequant.
        "pred_text.lstm0.fwd.W",
        NULL,
    };
    for (int i = 0; spot_check[i] != NULL; i++) {
        const char * nm = spot_check[i];
        kt_arena * a = kt_arena_new(256 * 1024 * 1024);
        kt_tensor * kt = kt_gguf_load_tensor(g, a, nm);
        if (kt == NULL) {
            fprintf(stderr, "FAIL spot %s: not found\n", nm);
            g_failures++;
        } else {
            struct ggml_tensor * gt = ggml_get_tensor(gctx, nm);
            if (gt == NULL) {
                fprintf(stderr, "FAIL ggml side missing %s\n", nm);
                g_failures++;
            } else {
                // Shapes — ggml stores ne similarly to us.
                int shape_ok = 1;
                for (int d = 0; d < kt->ndim; d++) {
                    if (kt->ne[d] != gt->ne[d]) { shape_ok = 0; }
                }
                if (!shape_ok) {
                    fprintf(stderr,
                        "FAIL shape %s: kt=[%lld,%lld,%lld,%lld] "
                        "ggml=[%lld,%lld,%lld,%lld]\n", nm,
                        (long long)kt->ne[0], (long long)kt->ne[1],
                        (long long)kt->ne[2], (long long)kt->ne[3],
                        (long long)gt->ne[0], (long long)gt->ne[1],
                        (long long)gt->ne[2], (long long)gt->ne[3]);
                    g_failures++;
                } else {
                    int64_t n = kt_nelements(kt);
                    float * gg_f32 = (float *)malloc(n * sizeof(float));
                    size_t off = gguf_get_data_offset(ggu)
                               + gguf_get_tensor_offset(ggu,
                                   (int)gguf_find_tensor(ggu, nm));
                    FILE * f = fopen(path, "rb");
                    fseek(f, (long)off, SEEK_SET);
                    if (gt->type == 0 /* F32 */) {
                        fread(gg_f32, sizeof(float), (size_t)n, f);
                    } else {
                        // F16 — read raw u16, convert same as kt does.
                        uint16_t * src16 = (uint16_t *)malloc(n * 2);
                        fread(src16, 2, (size_t)n, f);
                        for (int64_t k = 0; k < n; k++) {
                            _Float16 h;
                            memcpy(&h, &src16[k], 2);
                            gg_f32[k] = (float)h;
                        }
                        free(src16);
                    }
                    fclose(f);
                    double sqd = 0.0;
                    float maxd = 0.0f;
                    for (int64_t k = 0; k < n; k++) {
                        float d = kt->data[k] - gg_f32[k];
                        sqd += (double)d * (double)d;
                        if (fabsf(d) > maxd) { maxd = fabsf(d); }
                    }
                    float rmse = (float)sqrt(sqd / (double)n);
                    if (rmse > 1e-6f) {
                        printf("FAIL %s: RMSE %.3e max %.3e (n=%lld)\n",
                               nm, rmse, maxd, (long long)n);
                        g_failures++;
                    } else {
                        printf("PASS %-32s  RMSE %.3e (n=%lld, dtype=%d)\n",
                               nm, rmse, (long long)n, gt->type);
                    }
                    free(gg_f32);
                }
            }
        }
        kt_arena_free(a);
    }
    gguf_free(ggu);
    ggml_free(gctx);
    kt_gguf_close(g);
    if (g_failures == 0) {
        printf("\ngguf: ALL PASS\n");
    } else {
        printf("\ngguf: %d failures\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}

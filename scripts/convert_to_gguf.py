#!/usr/bin/env python3
"""
Comprehensive safetensors -> GGUF for the full KittenTTS-nano-v0.8 model
(BERT + TextStage heads + GeneratorStage). One file per arch ("kittens-tts").

The naming scheme mirrors torch_kitten.py's class hierarchy for traceability:

    embd.{word,pos,type}.weight          Albert input embeddings (fp32)
    embd.ln.{weight,bias}                Albert input LN (fp32)
    embd_to_hidden.{weight,bias}         128 -> 768 mapping
    layer.attn_{q,k,v,out}.{weight,bias} shared Albert attention
    layer.attn_ln.{weight,bias}
    layer.ffn{,_out}.{weight,bias}
    layer.full_ln.{weight,bias}

    bert_enc.{weight,bias}               768 -> 256 (post-BERT projection)
    pred_text.lstm0.{fwd,bwd}.{W,R,b}    PredictorTextEncoder
    pred_text.fc1.{weight,bias}          AdaLayerNorm projection 1
    pred_text.lstm2.{fwd,bwd}.{W,R,b}
    pred_text.fc3.{weight,bias}
    dur_proj.{weight,bias}               duration projection (128 -> 50)

    acoustic.embd.weight                 phoneme embedding (178, 128)
    acoustic.cnn{0,1}.{weight,bias}      2x conv1d
    acoustic.ln{0,1}.{gamma,beta}        2x LN
    acoustic.lstm.{fwd,bwd}.{W,R,b}      bidir LSTM

    shared.lstm.{fwd,bwd}.{W,R,b}        GeneratorStage shared LSTM

    For AdaINResBlock1D blocks, per block:
        <prefix>.n1.{fcW,fcB,nW,nB}      norm1 AdaFC (nW/nB optional)
        <prefix>.n2.{fcW,fcB,nW,nB}
        <prefix>.pool.{weight,bias}      only if upsample
        <prefix>.c1.{weight,bias}
        <prefix>.c2.{weight,bias}
        <prefix>.sv.{weight,bias}        only if has_conv1x1
        <prefix>.has_conv1x1             u8 flag (0/1)
        <prefix>.upsample                u8 flag

    Used prefixes:
        f0.{0,1,2}, n.{0,1,2}            predictor F0/N
        f0_proj.{weight,bias}, n_proj.{weight,bias}
        dec.encode.<...>                 decoder.encode (AdaINResBlock1D)
        dec.decode.{0,1,2,3}.<...>       decoder.decode[i]
        dec.asr.{weight,bias}            decoder.asr_res.0
        dec.f0_conv.{weight,bias}, dec.n_conv.{weight,bias}

    For AdaINResBlockHiFiGAN (3 dilations: 1,3,5):
        <prefix>.{a1,a2}.{k}.{fcW,fcB,nW,nB}  AdaFC × 3
        <prefix>.al{1,2}.{k}                  alpha snake parameters
        <prefix>.c{1,2}.{k}.{weight,bias}     dilation-k conv1d

    Used prefixes: nr0, nr1, gen.r{0..3}

    Generator:
        gen.u0.{weight,bias}             ConvTranspose1d (ne[0]=stride=10)
        gen.u1.{weight,bias}             ConvTranspose1d (ne[0]=stride=6)
        gen.cp.{weight,bias}             conv_post
        nc0.{weight,bias}, nc1.{weight,bias}   noise_convs

    Sine-generator + STFT:
        l_lin.{weight,bias}              sine->excitation linear (9 -> 1)
        stft_fwd.{real,imag}             STFT analysis kernels (conv1d)
        stft_bwd.{real,imag}             iSTFT synth kernels (conv_transpose_1d)

Convention: matmul weights stored numpy-(out,in) so ggml interprets them as
(in,out) — same as M1.  Conv1d / conv-transpose-1d weights are written as-is
in PyTorch (Cout, Cin, K) layout — ggml's conv ops expect (K, Cin, Cout) so
reshape is done at graph-build time via reinterpretation. (See the C reader
for details.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "vendors" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
from torch_kitten import (  # noqa: E402
    WeightBag, _permute_iofc_to_ifgo,
)

ARCH = "kittens-tts"


# ---------------------------------------------------------------------------

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def _np_native(t: torch.Tensor) -> np.ndarray:
    """Keep fp16 if source is fp16 — only used for the BERT matmuls."""
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------

class GG:
    """Thin wrapper that records tensor names and writes a GGUF in one shot."""

    def __init__(self, out_path: str, arch: str = ARCH):
        self.w = gguf.GGUFWriter(out_path, arch)
        self.names: set[str] = set()
        self.kvs: list[tuple[str, object]] = []

    def kv_u32(self, k: str, v: int) -> None:
        self.w.add_uint32(k, int(v))

    def kv_i32(self, k: str, v: int) -> None:
        self.w.add_int32(k, int(v))

    def kv_f32(self, k: str, v: float) -> None:
        self.w.add_float32(k, float(v))

    def kv_bool(self, k: str, v: bool) -> None:
        self.w.add_bool(k, bool(v))

    def t(self, name: str, arr: np.ndarray, dtype: str = "f32") -> None:
        if name in self.names:
            raise RuntimeError(f"duplicate tensor: {name}")
        self.names.add(name)
        if dtype == "f16" and arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        elif dtype == "f32" and arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        self.w.add_tensor(name, np.ascontiguousarray(arr))

    def close(self) -> None:
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()


# ---------------------------------------------------------------------------
# BERT (same as M1)
# ---------------------------------------------------------------------------

def write_bert(g: GG, raw: dict[str, torch.Tensor]) -> None:
    def t32(k: str) -> np.ndarray:
        return raw[k].float().numpy()

    def native(k: str) -> np.ndarray:
        return raw[k].numpy()

    g.t("embd.word.weight", t32("kmodel.bert.embeddings.word_embeddings.weight"))
    g.t("embd.pos.weight",  t32("kmodel.bert.embeddings.position_embeddings.weight"))
    g.t("embd.type.weight", t32("kmodel.bert.embeddings.token_type_embeddings.weight"))
    g.t("embd.ln.weight",   t32("kmodel.bert.embeddings.LayerNorm.weight"))
    g.t("embd.ln.bias",     t32("kmodel.bert.embeddings.LayerNorm.bias"))

    g.t("embd_to_hidden.weight",
        t32("onnx::MatMul_5661").T)  # (768, 128) -> ggml (128, 768)
    g.t("embd_to_hidden.bias",
        t32("kmodel.bert.encoder.embedding_hidden_mapping_in.bias"))

    base = "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0"
    g.t("layer.attn_q.weight", native("onnx::MatMul_5662").T, dtype="f16")
    g.t("layer.attn_q.bias",   t32(f"{base}.attention.query.bias"))
    g.t("layer.attn_k.weight", native("onnx::MatMul_5665").T, dtype="f16")
    g.t("layer.attn_k.bias",   t32(f"{base}.attention.key.bias"))
    g.t("layer.attn_v.weight", native("onnx::MatMul_5668").T, dtype="f16")
    g.t("layer.attn_v.bias",   t32(f"{base}.attention.value.bias"))
    g.t("layer.attn_out.weight", t32("onnx::MatMul_5672").T)
    g.t("layer.attn_out.bias", t32(f"{base}.attention.dense.bias"))
    g.t("layer.attn_ln.weight", t32(f"{base}.attention.LayerNorm.weight"))
    g.t("layer.attn_ln.bias",   t32(f"{base}.attention.LayerNorm.bias"))

    g.t("layer.ffn.weight",     native("onnx::MatMul_5673").T, dtype="f16")
    g.t("layer.ffn.bias",       t32(f"{base}.ffn.bias"))
    g.t("layer.ffn_out.weight", native("onnx::MatMul_5674").T, dtype="f16")
    g.t("layer.ffn_out.bias",   t32(f"{base}.ffn_output.bias"))
    g.t("layer.full_ln.weight", t32(f"{base}.full_layer_layer_norm.weight"))
    g.t("layer.full_ln.bias",   t32(f"{base}.full_layer_layer_norm.bias"))


# ---------------------------------------------------------------------------
# LSTM helpers
# ---------------------------------------------------------------------------

def write_lstm(g: GG, prefix: str, w: WeightBag,
               W_key: str, R_key: str, B_key: str) -> None:
    """Write a bidir LSTM as <prefix>.{fwd,bwd}.{W,R,b}.

    Source layout (after dequant_lstm): (num_dir=2, 4H, in) in iofc order.
    Permute each 4H block to ifgo. Then save with numpy (4H, in_or_H) so
    ggml interprets as (in_or_H, 4H) — matches the M1 matmul convention.
    """
    W = w.dequant_lstm(W_key)
    R = w.dequant_lstm(R_key)
    B = w.f32(B_key)
    assert W is not None and R is not None and B is not None
    assert W.shape[0] == 2, f"expected bidir, got {W.shape}"
    H = W.shape[1] // 4

    for d_idx, side in enumerate(("fwd", "bwd")):
        Wd = _permute_iofc_to_ifgo(W[d_idx])               # (4H, in)
        Rd = _permute_iofc_to_ifgo(R[d_idx])               # (4H, H)
        bd_W = _permute_iofc_to_ifgo(B[d_idx][: 4 * H])     # (4H,)
        bd_R = _permute_iofc_to_ifgo(B[d_idx][4 * H:])      # (4H,)
        # combined bias since both add identically into the gate pre-activation
        bd = bd_W + bd_R
        g.t(f"{prefix}.{side}.W", _np(Wd))   # numpy (4H, in)  -> ggml (in, 4H)
        g.t(f"{prefix}.{side}.R", _np(Rd))
        g.t(f"{prefix}.{side}.b", _np(bd))


# ---------------------------------------------------------------------------
# Conv1D helpers (kept in PyTorch native layout (Cout, Cin, K))
# ---------------------------------------------------------------------------

def _conv_w(w: WeightBag, base: str) -> np.ndarray:
    t = w.dequant(base)
    assert t is not None, f"missing conv weight {base}"
    return _np(t)  # (Cout, Cin, K)


def _conv_b(w: WeightBag, base: str, default_size: int) -> np.ndarray:
    b = w.bias(base)
    return _np(b) if b is not None else np.zeros(default_size, dtype=np.float32)


def write_conv1d(g: GG, prefix: str, w: WeightBag, base: str) -> None:
    """Conv1d kernel is stored as F16 because ggml_conv_1d requires F16
    src0 internally (the f16 im2col path). Bias stays F32."""
    cw = _conv_w(w, base)
    g.t(f"{prefix}.weight", cw, dtype="f16")
    g.t(f"{prefix}.bias",   _conv_b(w, base, cw.shape[0]))


# ---------------------------------------------------------------------------
# AdaFC (style -> gamma|beta projection used by AdaIN / AdaLayerNorm)
# ---------------------------------------------------------------------------

def write_ada_fc(g: GG, prefix: str, w: WeightBag, base: str, which: str) -> None:
    """Writes <prefix>.fcW (style->2C), .fcB, optional .nW/.nB for the inner LN.

    fcW layout in ONNX: (styleDim, 2C). Saved transposed so ggml (2C, style)
    matmul takes style on the right with style.ne[0] = styleDim.
    """
    fc_base = f"{base}.{which}.fc"
    fcW = w.dequant(fc_base)
    if fcW is None:
        fcW = w.f32(f"kmodel.{fc_base}.weight")
    assert fcW is not None, f"missing fc {fc_base}"
    g.t(f"{prefix}.fcW", _np(fcW).T)             # numpy (2C, style)
    g.t(f"{prefix}.fcB", _np(w.bias(fc_base)))   # (2C,)

    nw = w.raw.get(f"kmodel.{base}.{which}.norm.weight")
    nb = w.raw.get(f"kmodel.{base}.{which}.norm.bias")
    if nw is not None and nb is not None:
        g.t(f"{prefix}.nW", _np(nw))
        g.t(f"{prefix}.nB", _np(nb))
        g.kv_bool(f"{prefix}.has_norm", True)
    else:
        g.kv_bool(f"{prefix}.has_norm", False)


# ---------------------------------------------------------------------------
# AdaINResBlock1D weights
# ---------------------------------------------------------------------------

def write_ada_block_1d(g: GG, prefix: str, w: WeightBag, base: str,
                       upsample: bool, has_conv1x1: Optional[bool] = None
                       ) -> None:
    write_ada_fc(g, f"{prefix}.n1", w, base, "norm1")
    write_ada_fc(g, f"{prefix}.n2", w, base, "norm2")
    g.kv_bool(f"{prefix}.upsample", upsample)
    if upsample:
        pW = w.f32(f"kmodel.{base}.pool.weight")        # (C, 1, K)
        pB = w.f32(f"kmodel.{base}.pool.bias")
        assert pW is not None
        # Depthwise conv-transpose-1d doesn't exist in ggml. We replace it
        # with: insert zeros between input samples + ggml_conv_1d_dw using a
        # K-axis-flipped kernel. (See dump_lstm_fixture.py adjacent test.)
        # Pre-flip the K axis here once at conversion time. F16 because
        # ggml_conv_1d_dw goes through the f16 im2col path.
        pW_flipped = np.ascontiguousarray(_np(pW)[..., ::-1])
        g.t(f"{prefix}.pool.weight", pW_flipped, dtype="f16")
        g.t(f"{prefix}.pool.bias",   _np(pB) if pB is not None
            else np.zeros(pW.shape[0], dtype=np.float32))

    write_conv1d(g, f"{prefix}.c1", w, f"{base}.conv1")
    write_conv1d(g, f"{prefix}.c2", w, f"{base}.conv2")

    has_cv = (has_conv1x1 if has_conv1x1 is not None
              else (upsample
                    or w.has(f"kmodel.{base}.conv1x1.weight")
                    or w.has(f"kmodel.{base}.conv1x1.weight_quantized")))
    g.kv_bool(f"{prefix}.has_conv1x1", has_cv)
    if has_cv:
        write_conv1d(g, f"{prefix}.sv", w, f"{base}.conv1x1")


# ---------------------------------------------------------------------------
# AdaINResBlockHiFiGAN weights
# ---------------------------------------------------------------------------

def write_hifi_block(g: GG, prefix: str, w: WeightBag, base: str,
                     dilations=(1, 3, 5)) -> None:
    g.kv_u32(f"{prefix}.dilation_count", len(dilations))
    for k, d in enumerate(dilations):
        g.kv_u32(f"{prefix}.d{k}", d)
        write_ada_fc(g, f"{prefix}.a1.{k}", w, base, f"adain1.{k}")
        write_ada_fc(g, f"{prefix}.a2.{k}", w, base, f"adain2.{k}")
        al1 = w.f32(f"kmodel.{base}.alpha1.{k}")
        al2 = w.f32(f"kmodel.{base}.alpha2.{k}")
        # Store as (C,) — ggml broadcasts a 1D tensor over both ne[1] and ne[2]
        # of the (C, L) NLC activation.  Source shape is (1, C, 1).
        g.t(f"{prefix}.al1.{k}", _np(al1).reshape(-1))
        g.t(f"{prefix}.al2.{k}", _np(al2).reshape(-1))
        write_conv1d(g, f"{prefix}.c1.{k}", w, f"{base}.convs1.{k}")
        write_conv1d(g, f"{prefix}.c2.{k}", w, f"{base}.convs2.{k}")


# ---------------------------------------------------------------------------
# TextStage heads (post-BERT)
# ---------------------------------------------------------------------------

def write_textstage_heads(g: GG, w: WeightBag) -> None:
    # 768 -> 256 projection after BERT
    g.t("bert_enc.weight", _np(w.f32("onnx::MatMul_5818")).T)  # numpy (256, 768)
    g.t("bert_enc.bias",   _np(w.bias("bert_encoder")))

    # PredictorTextEncoder
    write_lstm(g, "pred_text.lstm0", w,
               "onnx::LSTM_5872", "onnx::LSTM_5873", "onnx::LSTM_5871")
    g.t("pred_text.fc1.weight",
        _np(w.f32("kmodel.predictor.text_encoder.lstms.1.fc.weight")).T)  # ONNX (style, 2C) -> save (2C, style)
    g.t("pred_text.fc1.bias",
        _np(w.f32("kmodel.predictor.text_encoder.lstms.1.fc.bias")))
    write_lstm(g, "pred_text.lstm2", w,
               "onnx::LSTM_5922", "onnx::LSTM_5923", "onnx::LSTM_5921")
    g.t("pred_text.fc3.weight",
        _np(w.f32("kmodel.predictor.text_encoder.lstms.3.fc.weight")).T)
    g.t("pred_text.fc3.bias",
        _np(w.f32("kmodel.predictor.text_encoder.lstms.3.fc.bias")))

    # duration LSTM (separate from PredictorTextEncoder's LSTMs)
    write_lstm(g, "dur.lstm", w,
               "onnx::LSTM_5971", "onnx::LSTM_5972", "onnx::LSTM_5970")

    # Duration projection (128 -> 50)
    g.t("dur_proj.weight", _np(w.f32("onnx::MatMul_5973")).T)
    g.t("dur_proj.bias",   _np(w.bias("predictor.duration_proj.linear_layer")))

    # AcousticTextEncoder
    g.t("acoustic.embd.weight", _np(w.f32("kmodel.text_encoder.embedding.weight")))
    for i in range(2):
        write_conv1d(g, f"acoustic.cnn{i}", w, f"text_encoder.cnn.{i}.0")
        g.t(f"acoustic.ln{i}.gamma", _np(w.f32(f"kmodel.text_encoder.cnn.{i}.1.gamma")))
        g.t(f"acoustic.ln{i}.beta",  _np(w.f32(f"kmodel.text_encoder.cnn.{i}.1.beta")))
    write_lstm(g, "acoustic.lstm", w,
               "onnx::LSTM_5652", "onnx::LSTM_5653", "onnx::LSTM_5651")


# ---------------------------------------------------------------------------
# GeneratorStage front (LR inputs -> F0/N projections)
# ---------------------------------------------------------------------------

def write_generator_front(g: GG, w: WeightBag) -> None:
    write_lstm(g, "shared.lstm", w,
               "onnx::LSTM_6020", "onnx::LSTM_6021", "onnx::LSTM_6019")

    write_ada_block_1d(g, "f0.0", w, "predictor.F0.0", upsample=False)
    write_ada_block_1d(g, "f0.1", w, "predictor.F0.1", upsample=True)
    write_ada_block_1d(g, "f0.2", w, "predictor.F0.2", upsample=False)
    write_ada_block_1d(g, "n.0",  w, "predictor.N.0",  upsample=False)
    write_ada_block_1d(g, "n.1",  w, "predictor.N.1",  upsample=True)
    write_ada_block_1d(g, "n.2",  w, "predictor.N.2",  upsample=False)

    write_conv1d(g, "f0_proj", w, "predictor.F0_proj")
    write_conv1d(g, "n_proj",  w, "predictor.N_proj")


# ---------------------------------------------------------------------------
# Decoder pipeline
# ---------------------------------------------------------------------------

def write_decoder(g: GG, w: WeightBag) -> None:
    write_conv1d(g, "dec.asr",     w, "decoder.asr_res.0")
    write_conv1d(g, "dec.f0_conv", w, "decoder.F0_conv")
    write_conv1d(g, "dec.n_conv",  w, "decoder.N_conv")
    write_ada_block_1d(g, "dec.encode", w, "decoder.encode",
                       upsample=False, has_conv1x1=True)
    for i in range(4):
        write_ada_block_1d(g, f"dec.decode.{i}", w, f"decoder.decode.{i}",
                           upsample=(i == 3), has_conv1x1=True)


# ---------------------------------------------------------------------------
# Generator + iSTFT + noise path
# ---------------------------------------------------------------------------

def write_generator_back(g: GG, w: WeightBag) -> None:
    # Upsamplers: ConvTranspose1d, weights stored (Cin, Cout, K) per PyTorch
    g.t("gen.u0.weight", _np(w.f32("kmodel.decoder.generator.ups.0.weight")))
    g.t("gen.u0.bias",   _np(w.f32("kmodel.decoder.generator.ups.0.bias")))
    g.t("gen.u1.weight", _np(w.f32("kmodel.decoder.generator.ups.1.weight")))
    g.t("gen.u1.bias",   _np(w.f32("kmodel.decoder.generator.ups.1.bias")))

    write_hifi_block(g, "gen.r0", w, "decoder.generator.resblocks.0")
    write_hifi_block(g, "gen.r1", w, "decoder.generator.resblocks.1")
    write_hifi_block(g, "gen.r2", w, "decoder.generator.resblocks.2")
    write_hifi_block(g, "gen.r3", w, "decoder.generator.resblocks.3")

    write_conv1d(g, "gen.cp", w, "decoder.generator.conv_post")

    # Noise path
    write_conv1d(g, "nc0", w, "decoder.generator.noise_convs.0")
    write_conv1d(g, "nc1", w, "decoder.generator.noise_convs.1")
    write_hifi_block(g, "nr0", w, "decoder.generator.noise_res.0")
    write_hifi_block(g, "nr1", w, "decoder.generator.noise_res.1")

    # sine->excitation linear (9 -> 1)
    l_lin = w.f32("onnx::MatMul_6116")
    if l_lin is None:
        l_lin = w.f32("onnx::MatMul_6388")
    assert l_lin is not None, "missing l_linear matmul"
    g.t("l_lin.weight", _np(l_lin).T)
    g.t("l_lin.bias",   _np(w.bias("decoder.generator.m_source.l_linear")))

    # STFT analysis kernels — kept F32 (used via custom im2col+mul_mat path
    # for higher precision than F16; sin/cos kernel values lose noticeable
    # precision in F16).
    g.t("stft_fwd.real",
        _np(w.f32("kmodel.decoder.generator.stft.weight_forward_real")))
    g.t("stft_fwd.imag",
        _np(w.f32("kmodel.decoder.generator.stft.weight_forward_imag")))

    # iSTFT synthesis kernels (conv_transpose_1d) — PyTorch (Cin, Cout, K)
    g.t("stft_bwd.real",
        _np(w.f32("kmodel.decoder.generator.stft.weight_backward_real")))
    g.t("stft_bwd.imag",
        _np(w.f32("kmodel.decoder.generator.stft.weight_backward_imag")))


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path,
                    default=REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    ap.add_argument("--out", dest="dst", type=Path,
                    default=REPO_ROOT / "tmp/kitten_full.gguf")
    args = ap.parse_args()

    print(f"reading {args.src}")
    raw: dict[str, torch.Tensor] = {}
    with safe_open(str(args.src), framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    w = WeightBag(raw)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    g = GG(str(args.dst), ARCH)

    # ---- arch metadata (BERT shared with M1) ----
    g.kv_u32("kittens-tts.vocab_size", 178)
    g.kv_u32("kittens-tts.max_position", 512)
    g.kv_u32("kittens-tts.token_types", 2)
    g.kv_u32("kittens-tts.embedding_dim", 128)
    g.kv_u32("kittens-tts.hidden_size", 768)
    g.kv_u32("kittens-tts.num_layers", 12)
    g.kv_u32("kittens-tts.num_heads", 12)
    g.kv_u32("kittens-tts.head_dim", 64)
    g.kv_u32("kittens-tts.ffn_dim", 2048)
    g.kv_f32("kittens-tts.layer_norm_eps", 1e-12)

    # ---- TextStage / GeneratorStage shapes ----
    g.kv_u32("kittens-tts.bert_enc_dim", 256)
    g.kv_u32("kittens-tts.style_dim", 128)
    g.kv_u32("kittens-tts.lstm_hidden", 64)        # H per direction (so 2H=128)
    g.kv_u32("kittens-tts.dur_logits", 50)
    g.kv_u32("kittens-tts.audio_per_frame", 600)   # frame upsamples to 2*hop=600 samples (24kHz, 80fps)
    g.kv_u32("kittens-tts.istft_hop", 5)
    g.kv_u32("kittens-tts.istft_trim", 10)

    # ---- tensors ----
    print("[bert]")
    write_bert(g, raw)
    print("[textstage heads]")
    write_textstage_heads(g, w)
    print("[generator front]")
    write_generator_front(g, w)
    print("[decoder]")
    write_decoder(g, w)
    print("[generator back + iSTFT]")
    write_generator_back(g, w)

    g.close()
    sz = args.dst.stat().st_size / (1024 * 1024)
    print(f"done. {args.dst}  {sz:.1f} MB  ({len(g.names)} tensors)")


if __name__ == "__main__":
    main()

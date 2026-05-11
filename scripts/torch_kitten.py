#!/usr/bin/env python3
"""PyTorch port of Sources/KittenApp/TTS.swift's kittenForward.

Serves two purposes:
  1. coremltools.convert needs a torch.nn.Module; this file provides
     TextStage and GeneratorStage modules that trace cleanly.
  2. Python-side numerical validation against the reference ONNX runtime.

Naming, ordering, and math match TTS.swift line-for-line. The MLX version
was validated against the upstream ONNX model, so this port just needs to
reproduce the MLX code to reproduce the model.

Weight layout notes:
  * Linear/Conv weights may be stored as fp32, fp16, or INT8-quantized
    (weight_quantized + weight_scale + weight_zero_point). WeightBag.dequant
    handles all three.
  * AdaIN fc weights live in ONNX MatMul layout (in, out); PyTorch Linear
    wants (out, in) so we transpose on load.
  * LSTM weights are stored in ONNX layout (num_dir, input_size, 4H) with
    gate order iofc. PyTorch nn.LSTM uses (4H, input_size) per direction
    with gate order ifgo — we permute on load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# =============================================================================
# Weight bag
# =============================================================================


class WeightBag:
    """Holds every raw tensor from the safetensors file.

    Mirrors the `Weights` struct in TTS.swift. All returned tensors are fp32.
    """

    def __init__(self, raw: dict[str, torch.Tensor]):
        self.raw = raw

    @classmethod
    def load(cls, path: str | Path) -> "WeightBag":
        raw: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
        return cls(raw)

    def has(self, key: str) -> bool:
        return key in self.raw

    def f32(self, key: str) -> Optional[torch.Tensor]:
        t = self.raw.get(key)
        return None if t is None else t.to(torch.float32)

    def dequant(self, base: str) -> Optional[torch.Tensor]:
        """Dequantize `kmodel.{base}.weight` if an INT8 triple exists."""
        prefix = f"kmodel.{base}"
        wq = self.raw.get(f"{prefix}.weight_quantized")
        if wq is not None:
            s = self.raw[f"{prefix}.weight_scale"].to(torch.float32)
            z = self.raw[f"{prefix}.weight_zero_point"].to(torch.float32)
            return (wq.to(torch.float32) - z) * s
        w = self.raw.get(f"{prefix}.weight")
        return None if w is None else w.to(torch.float32)

    def dequant_raw(self, base: str) -> Optional[torch.Tensor]:
        """Dequantize `{base}_quantized/_scale/_zero_point`."""
        wq = self.raw.get(f"{base}_quantized")
        if wq is not None:
            s = self.raw[f"{base}_scale"].to(torch.float32)
            z = self.raw[f"{base}_zero_point"].to(torch.float32)
            return (wq.to(torch.float32) - z) * s
        t = self.raw.get(base)
        return None if t is None else t.to(torch.float32)

    def bias(self, base: str) -> Optional[torch.Tensor]:
        b = self.raw.get(f"kmodel.{base}.bias")
        return None if b is None else b.to(torch.float32)

    def dequant_lstm(self, base: str) -> Optional[torch.Tensor]:
        """Dequantize a DynamicQuantizeLSTM matrix.

        Input  shape: (num_dir, input_size, 4H) INT8 with per-direction scalars.
        Output shape: (num_dir, 4H, input_size) — ONNX LSTM spec layout.
        """
        wq = self.raw.get(f"{base}_quantized")
        if wq is None:
            return None
        s = self.raw[f"{base}_scale"].to(torch.float32).reshape(-1, 1, 1)
        z = self.raw[f"{base}_zero_point"].to(torch.float32).reshape(-1, 1, 1)
        dq = (wq.to(torch.float32) - z) * s
        return dq.transpose(1, 2).contiguous()


# =============================================================================
# Activations & helpers (mirror TTS.swift MARK: - Activations / padding)
# =============================================================================


def snake_1d(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake: x + (1/alpha) * sin(alpha * x)^2. alpha is (C,) on NCL."""
    a = alpha.reshape(1, -1, 1)
    inner = torch.sin(a * x)
    return x + (1.0 / a) * inner * inner


def reflection_pad_left(x: torch.Tensor, n: int) -> torch.Tensor:
    """Reflect-pad on the LEFT only (KittenTTS generator quirk)."""
    if n <= 0:
        return x
    # indices [1..n] reversed — skip the boundary sample.
    reflected = x[..., 1 : n + 1].flip(-1)
    return torch.cat([reflected, x], dim=-1)


# =============================================================================
# LayerNorm / InstanceNorm
# =============================================================================


def layer_norm_last(x: torch.Tensor, weight: Optional[torch.Tensor],
                    bias: Optional[torch.Tensor], eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    # Match MLX variance (population, unbiased=False).
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    h = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        h = h * weight
    if bias is not None:
        h = h + bias
    return h


def instance_norm_1d_ncl(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps)


# =============================================================================
# AdaIN / AdaLayerNorm
# =============================================================================


@dataclass
class AdaFC:
    fcW: torch.Tensor   # (styleDim, 2C)  — ONNX MatMul layout
    fcB: torch.Tensor   # (2C,)
    normW: Optional[torch.Tensor]  # (C,) or None
    normB: Optional[torch.Tensor]  # (C,) or None


def load_ada_fc(w: WeightBag, base: str, which: str) -> AdaFC:
    fc_base = f"{base}.{which}.fc"
    fcW = w.dequant(fc_base)
    if fcW is None:
        fcW = w.f32(f"kmodel.{fc_base}.weight")
    assert fcW is not None, f"missing fc weight for {fc_base}"
    fcB = w.bias(fc_base)
    assert fcB is not None, f"missing fc bias for {fc_base}"
    nw = w.raw.get(f"kmodel.{base}.{which}.norm.weight")
    nb = w.raw.get(f"kmodel.{base}.{which}.norm.bias")
    return AdaFC(
        fcW=fcW,
        fcB=fcB,
        normW=nw.to(torch.float32) if nw is not None else None,
        normB=nb.to(torch.float32) if nb is not None else None,
    )


def ada_in_1d(x: torch.Tensor, style: torch.Tensor, fc: AdaFC) -> torch.Tensor:
    """AdaIN1d on NCL. style (B, styleDim). fcW (styleDim, 2C)."""
    h = style @ fc.fcW + fc.fcB  # (B, 2C)
    C = h.shape[-1] // 2
    gamma = h[:, :C].reshape(-1, C, 1)
    beta = h[:, C:].reshape(-1, C, 1)
    normed = instance_norm_1d_ncl(x)
    if fc.normW is not None:
        normed = normed * fc.normW.reshape(1, -1, 1)
    if fc.normB is not None:
        normed = normed + fc.normB.reshape(1, -1, 1)
    return normed * (1.0 + gamma) + beta


def ada_layer_norm(x: torch.Tensor, style: torch.Tensor,
                   fcW: torch.Tensor, fcB: torch.Tensor) -> torch.Tensor:
    """AdaLayerNorm on NLC. fcW (styleDim, 2C). Normalizes last dim."""
    h = style @ fcW + fcB  # (B, 2C)
    C = h.shape[-1] // 2
    gamma = h[:, :C].reshape(-1, 1, C)
    beta = h[:, C:].reshape(-1, 1, C)
    normed = layer_norm_last(x, None, None)
    return normed * (1.0 + gamma) + beta


# =============================================================================
# Conv1d loaders (NCL, ONNX weight layout = (Cout, Cin, K))
# =============================================================================


def load_conv1d(w: WeightBag, base: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (weight (Cout, Cin, K), bias (Cout,) or None)."""
    weight = w.dequant(base)
    assert weight is not None, f"missing conv weight for {base}"
    return weight, w.bias(base)


def conv1d_ncl(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
               stride: int = 1, padding: int = 0, dilation: int = 1,
               groups: int = 1) -> torch.Tensor:
    return F.conv1d(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)


def conv_transpose_1d_ncl(x: torch.Tensor, weight: torch.Tensor,
                          bias: Optional[torch.Tensor], stride: int = 1,
                          padding: int = 0, output_padding: int = 0,
                          groups: int = 1) -> torch.Tensor:
    return F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding,
                              output_padding=output_padding, groups=groups)


def conv_transpose_1d_depthwise_ncl(x: torch.Tensor, weight: torch.Tensor,
                                    bias: Optional[torch.Tensor],
                                    stride: int, padding: int,
                                    output_padding: int) -> torch.Tensor:
    """Depthwise ConvTranspose1d. Weight shape (C, 1, K); use groups=C."""
    C = weight.shape[0]
    return F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding,
                              output_padding=output_padding, groups=C)


# =============================================================================
# ONNX LSTM (wraps nn.LSTM with weights re-permuted from iofc -> ifgo)
# =============================================================================
#
# ONNX gate order is i,o,f,c (cell-hat). PyTorch nn.LSTM expects i,f,g,o.
# Row permutation of the 4H-sized weight axis: [0..H, 2H..3H, 3H..4H, H..2H].


def _permute_iofc_to_ifgo(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Permute the 4H axis of `t` from iofc to ifgo ordering."""
    H4 = t.shape[dim]
    assert H4 % 4 == 0, f"expected 4H axis, got {t.shape}"
    H = H4 // 4
    i = t.narrow(dim, 0, H)
    o = t.narrow(dim, H, H)
    f = t.narrow(dim, 2 * H, H)
    g = t.narrow(dim, 3 * H, H)
    return torch.cat([i, f, g, o], dim=dim).contiguous()


class ONNXBidirLSTM(nn.Module):
    """Bidirectional LSTM that wraps torch.nn.LSTM with ONNX -> PyTorch gate
    permutation on load. Traces to a single CoreML `lstm` primitive — ~10×
    faster than a Python-unrolled loop for L=128.

    Weights loaded from ONNX layout:
        W:  (num_dir, 4H, in)   (after WeightBag.dequant_lstm)
        R:  (num_dir, 4H, H)
        B:  (num_dir, 8H)  = [Wb | Rb] per direction
    All in iofc gate order; permuted to ifgo (PyTorch's order) on load.

    NOTE: this version does NOT mask state at pad timesteps. When the caller
    pads the input sequence the backward direction's state will drift slightly
    through the pad prefix. Callers should zero pad inputs (x * mask) to
    minimise drive drift, and pick the smallest bucket that fits their real
    length to minimise pad count. See scripts/diagnose_lstm_leak.py for
    measured drift vs. pad count.

    Output shape: (seq, 2, batch, H), to match the rest of the port.
    """

    def __init__(self, W: torch.Tensor, R: torch.Tensor, B: torch.Tensor):
        super().__init__()
        num_dir, H4, in_size = W.shape
        H = H4 // 4
        assert num_dir == 2, "bidirectional only"
        self.hidden_size = H
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=H,
                            num_layers=1, bidirectional=True,
                            batch_first=False)
        with torch.no_grad():
            for d_idx, suffix in enumerate(("l0", "l0_reverse")):
                wih = _permute_iofc_to_ifgo(W[d_idx])
                whh = _permute_iofc_to_ifgo(R[d_idx])
                wb = _permute_iofc_to_ifgo(B[d_idx][: 4 * H])
                rb = _permute_iofc_to_ifgo(B[d_idx][4 * H :])
                getattr(self.lstm, f"weight_ih_{suffix}").copy_(wih)
                getattr(self.lstm, f"weight_hh_{suffix}").copy_(whh)
                getattr(self.lstm, f"bias_ih_{suffix}").copy_(wb)
                getattr(self.lstm, f"bias_hh_{suffix}").copy_(rb)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (seq, batch, in). mask is accepted for API compatibility but
        currently ignored — pad drift is handled by (a) x*mask at call sites
        and (b) smallest-bucket-that-fits at runtime."""
        _ = mask
        out, _ = self.lstm(x)           # (seq, batch, 2H)
        seq, batch, two_H = out.shape
        H = two_H // 2
        return out.view(seq, batch, 2, H).permute(0, 2, 1, 3).contiguous()


def load_onnx_bidir_lstm(w: WeightBag, W_key: str, R_key: str, B_key: str
                         ) -> ONNXBidirLSTM:
    W = w.dequant_lstm(W_key)
    R = w.dequant_lstm(R_key)
    B = w.f32(B_key)
    assert W is not None and R is not None and B is not None
    return ONNXBidirLSTM(W, R, B)


def load_dq_lstm(w: WeightBag, W_key: str, R_key: str, B_key: str
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy — returns raw (W, R, B). Kept for a few remaining call sites."""
    W = w.dequant_lstm(W_key)
    R = w.dequant_lstm(R_key)
    B = w.f32(B_key)
    assert W is not None and R is not None and B is not None
    return W, R, B


# =============================================================================
# BERT stack (Albert-style shared 12-layer)
# =============================================================================


class BertStack(nn.Module):
    """12 shared Albert layers + embeddings. Tensors are registered as buffers."""

    def __init__(self, w: WeightBag):
        super().__init__()

        def reg(name: str, t: torch.Tensor):
            self.register_buffer(name, t.contiguous())

        reg("we", w.dequant("bert.embeddings.word_embeddings"))
        reg("pe", w.dequant("bert.embeddings.position_embeddings"))
        reg("te", w.dequant("bert.embeddings.token_type_embeddings"))
        reg("lnW", w.dequant("bert.embeddings.LayerNorm"))
        reg("lnB", w.bias("bert.embeddings.LayerNorm"))

        # 128 -> 768 mapping at the top of the Albert encoder.
        reg("mIn", w.f32("onnx::MatMul_5661"))
        reg("mInB", w.bias("bert.encoder.embedding_hidden_mapping_in"))

        # Shared Albert block (applied 12x).
        base = "bert.encoder.albert_layer_groups.0.albert_layers.0"
        reg("qW", w.f32("onnx::MatMul_5662"))
        reg("kW", w.f32("onnx::MatMul_5665"))
        reg("vW", w.f32("onnx::MatMul_5668"))
        reg("dW", w.f32("onnx::MatMul_5672"))
        reg("ffnW", w.f32("onnx::MatMul_5673"))
        reg("ffnOutW", w.f32("onnx::MatMul_5674"))
        reg("qB", w.bias(f"{base}.attention.query"))
        reg("kB", w.bias(f"{base}.attention.key"))
        reg("vB", w.bias(f"{base}.attention.value"))
        reg("dB", w.bias(f"{base}.attention.dense"))
        reg("ffnB", w.bias(f"{base}.ffn"))
        reg("ffnOutB", w.bias(f"{base}.ffn_output"))
        reg("attLnW", w.dequant(f"{base}.attention.LayerNorm"))
        reg("attLnB", w.bias(f"{base}.attention.LayerNorm"))
        reg("fullLnW", w.dequant(f"{base}.full_layer_layer_norm"))
        reg("fullLnB", w.bias(f"{base}.full_layer_layer_norm"))

        self.n_layers = 12
        self.n_head = 12
        self.d_head = 64
        self.d_model = 768

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Batch is always 1; avoid reading input_ids.shape[0] (triggers
        # aten::Int nodes that break coremltools tracing).
        L = input_ids.size(1)
        pos = torch.arange(L, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        word_emb = F.embedding(input_ids, self.we)
        pos_emb = F.embedding(pos, self.pe)
        type_emb = F.embedding(torch.zeros_like(input_ids), self.te)
        h = word_emb + pos_emb + type_emb
        h = layer_norm_last(h, self.lnW, self.lnB, eps=1e-12)
        h = h @ self.mIn + self.mInB

        # Build additive attention bias once: 0 for real, -1e9 for pad.
        # Broadcasts against scores shape (1, n_head, L, L): need (1, 1, 1, L).
        if attention_mask is not None:
            att_bias = (1.0 - attention_mask) * -1e9
            att_bias = att_bias.reshape(1, 1, 1, -1)
        else:
            att_bias = None

        nh = self.n_head
        dh = self.d_head
        D = self.d_model
        for _ in range(self.n_layers):
            q = (h @ self.qW + self.qB).reshape(1, -1, nh, dh).transpose(1, 2)
            k = (h @ self.kW + self.kB).reshape(1, -1, nh, dh).transpose(1, 2)
            v = (h @ self.vW + self.vB).reshape(1, -1, nh, dh).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            if att_bias is not None:
                scores = scores + att_bias
            attn = torch.softmax(scores, dim=-1)
            ctx = (attn @ v).transpose(1, 2).reshape(1, -1, D)
            att_out = ctx @ self.dW + self.dB
            h_mid = layer_norm_last(att_out + h, self.attLnW, self.attLnB, eps=1e-12)
            ffn_h = h_mid @ self.ffnW + self.ffnB
            ffn_act = F.gelu(ffn_h)
            ffn_res = ffn_act @ self.ffnOutW + self.ffnOutB
            h = layer_norm_last(ffn_res + h_mid, self.fullLnW, self.fullLnB, eps=1e-12)
        return h  # (1, L, 768)


# =============================================================================
# Predictor / acoustic text encoders
# =============================================================================


class PredictorTextEncoder(nn.Module):
    def __init__(self, w: WeightBag):
        super().__init__()
        self.lstm0 = load_onnx_bidir_lstm(w, "onnx::LSTM_5872", "onnx::LSTM_5873", "onnx::LSTM_5871")
        self.register_buffer("fc1W", w.f32("kmodel.predictor.text_encoder.lstms.1.fc.weight"))
        self.register_buffer("fc1B", w.f32("kmodel.predictor.text_encoder.lstms.1.fc.bias"))
        self.lstm2 = load_onnx_bidir_lstm(w, "onnx::LSTM_5922", "onnx::LSTM_5923", "onnx::LSTM_5921")
        self.register_buffer("fc3W", w.f32("kmodel.predictor.text_encoder.lstms.3.fc.weight"))
        self.register_buffer("fc3B", w.f32("kmodel.predictor.text_encoder.lstms.3.fc.bias"))

    def forward(self, bert_out_nlc: torch.Tensor, prosody_style: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s_bcast = prosody_style.unsqueeze(1) + torch.zeros_like(bert_out_nlc[..., :128])

        # mask for per-timestep LSTM state freezing.  shape (L, 1) to match
        # the (L, 1, in) LSTM input layout.
        lstm_mask = attention_mask.transpose(0, 1) if attention_mask is not None else None

        x = torch.cat([bert_out_nlc, s_bcast], dim=-1)
        xT = x.transpose(0, 1).contiguous()  # (L, 1, 256)
        y = self.lstm0(xT, mask=lstm_mask)
        y0 = y.permute(2, 0, 1, 3).reshape(1, -1, 128)
        y1 = ada_layer_norm(y0, prosody_style, self.fc1W, self.fc1B)
        if attention_mask is not None:
            y1 = y1 * attention_mask.unsqueeze(-1)

        x = torch.cat([y1, s_bcast], dim=-1)
        xT = x.transpose(0, 1).contiguous()
        y = self.lstm2(xT, mask=lstm_mask)
        y2 = y.permute(2, 0, 1, 3).reshape(1, -1, 128)
        return ada_layer_norm(y2, prosody_style, self.fc3W, self.fc3B)


class AcousticTextEncoder(nn.Module):
    def __init__(self, w: WeightBag):
        super().__init__()
        self.register_buffer("w_emb", w.f32("kmodel.text_encoder.embedding.weight"))
        for i in range(2):
            cw, cb = load_conv1d(w, f"text_encoder.cnn.{i}.0")
            self.register_buffer(f"cnn{i}W", cw)
            self.register_buffer(f"cnn{i}B", cb if cb is not None else torch.zeros(cw.shape[0]))
            self.register_buffer(f"ln{i}g", w.f32(f"kmodel.text_encoder.cnn.{i}.1.gamma"))
            self.register_buffer(f"ln{i}b", w.f32(f"kmodel.text_encoder.cnn.{i}.1.beta"))
        self.lstm = load_onnx_bidir_lstm(w, "onnx::LSTM_5652", "onnx::LSTM_5653", "onnx::LSTM_5651")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_nlc = F.embedding(input_ids[0], self.w_emb).unsqueeze(0)  # (1, L, 128)
        if attention_mask is not None:
            x_nlc = x_nlc * attention_mask.unsqueeze(-1)
        for i in range(2):
            cnnW = getattr(self, f"cnn{i}W")
            cnnB = getattr(self, f"cnn{i}B")
            K = cnnW.shape[-1]
            x_ncl = x_nlc.transpose(1, 2)
            x_ncl = conv1d_ncl(x_ncl, cnnW, cnnB, padding=(K - 1) // 2)
            x_nlc = x_ncl.transpose(1, 2)
            g = getattr(self, f"ln{i}g")
            b = getattr(self, f"ln{i}b")
            x_nlc = layer_norm_last(x_nlc, g, b)
            x_nlc = F.leaky_relu(x_nlc, 0.2)
            if attention_mask is not None:
                x_nlc = x_nlc * attention_mask.unsqueeze(-1)

        xT = x_nlc.transpose(0, 1).contiguous()
        lstm_mask = attention_mask.transpose(0, 1) if attention_mask is not None else None
        y = self.lstm(xT, mask=lstm_mask)
        y_nlc = y.permute(2, 0, 1, 3).reshape(1, -1, 128)
        return y_nlc.transpose(1, 2)  # (1, 128, L)


# =============================================================================
# AdaIN ResBlock 1D (Predictor F0/N, decoder encode/decode)
# =============================================================================


class AdaINResBlock1D(nn.Module):
    def __init__(self, w: WeightBag, base: str, upsample: bool, divide: bool,
                 has_conv1x1: Optional[bool] = None):
        super().__init__()
        self.base = base
        self.upsample = upsample
        self.divide = divide

        def reg_ada(prefix: str, which: str):
            fc = load_ada_fc(w, base, which)
            self.register_buffer(f"{prefix}_fcW", fc.fcW)
            self.register_buffer(f"{prefix}_fcB", fc.fcB)
            if fc.normW is not None:
                self.register_buffer(f"{prefix}_nW", fc.normW)
                self.register_buffer(f"{prefix}_nB", fc.normB)
            else:
                self.register_parameter(f"{prefix}_nW", None)
                self.register_parameter(f"{prefix}_nB", None)

        reg_ada("n1", "norm1")
        reg_ada("n2", "norm2")

        if upsample:
            pW = w.f32(f"kmodel.{base}.pool.weight")
            pB = w.f32(f"kmodel.{base}.pool.bias")
            self.register_buffer("poolW", pW)
            if pB is not None:
                self.register_buffer("poolB", pB)
            else:
                self.register_parameter("poolB", None)

        c1w, c1b = load_conv1d(w, f"{base}.conv1")
        self.register_buffer("c1W", c1w)
        if c1b is not None:
            self.register_buffer("c1B", c1b)
        else:
            self.register_parameter("c1B", None)
        c2w, c2b = load_conv1d(w, f"{base}.conv2")
        self.register_buffer("c2W", c2w)
        if c2b is not None:
            self.register_buffer("c2B", c2b)
        else:
            self.register_parameter("c2B", None)

        has_cv = (has_conv1x1 if has_conv1x1 is not None
                  else (upsample
                        or w.has(f"kmodel.{base}.conv1x1.weight")
                        or w.has(f"kmodel.{base}.conv1x1.weight_quantized")))
        self.has_conv1x1 = has_cv
        if has_cv:
            sw, sb = load_conv1d(w, f"{base}.conv1x1")
            self.register_buffer("svW", sw)
            if sb is not None:
                self.register_buffer("svB", sb)
            else:
                self.register_parameter("svB", None)

    def _ada(self, prefix: str, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        fc = AdaFC(
            fcW=getattr(self, f"{prefix}_fcW"),
            fcB=getattr(self, f"{prefix}_fcB"),
            normW=getattr(self, f"{prefix}_nW", None),
            normB=getattr(self, f"{prefix}_nB", None),
        )
        return ada_in_1d(x, style, fc)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                shortcut_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        shortcut = shortcut_input if shortcut_input is not None else x
        h = self._ada("n1", x, style)
        h = F.leaky_relu(h, 0.2)
        if self.upsample:
            h = conv_transpose_1d_depthwise_ncl(
                h, self.poolW, getattr(self, "poolB", None),
                stride=2, padding=1, output_padding=1)
        K1 = self.c1W.shape[-1]
        h = conv1d_ncl(h, self.c1W, getattr(self, "c1B", None), padding=(K1 - 1) // 2)
        h = self._ada("n2", h, style)
        h = F.leaky_relu(h, 0.2)
        K2 = self.c2W.shape[-1]
        h = conv1d_ncl(h, self.c2W, getattr(self, "c2B", None), padding=(K2 - 1) // 2)

        if self.upsample:
            repeated = shortcut.repeat_interleave(2, dim=-1)
            res = conv1d_ncl(repeated, self.svW, getattr(self, "svB", None))
        elif self.has_conv1x1:
            res = conv1d_ncl(shortcut, self.svW, getattr(self, "svB", None))
        else:
            res = shortcut

        out = h + res
        if self.divide:
            out = out / math.sqrt(2.0)
        return out


# =============================================================================
# AdaIN ResBlock HiFi-GAN (Snake + multi-receptive-field)
# =============================================================================


class AdaINResBlockHiFiGAN(nn.Module):
    def __init__(self, w: WeightBag, base: str, dilations=(1, 3, 5)):
        super().__init__()
        self.dilations = list(dilations)
        for k, _d in enumerate(self.dilations):
            a1 = load_ada_fc(w, base, f"adain1.{k}")
            a2 = load_ada_fc(w, base, f"adain2.{k}")
            self.register_buffer(f"a1_{k}_fcW", a1.fcW)
            self.register_buffer(f"a1_{k}_fcB", a1.fcB)
            if a1.normW is not None:
                self.register_buffer(f"a1_{k}_nW", a1.normW)
                self.register_buffer(f"a1_{k}_nB", a1.normB)
            self.register_buffer(f"a2_{k}_fcW", a2.fcW)
            self.register_buffer(f"a2_{k}_fcB", a2.fcB)
            if a2.normW is not None:
                self.register_buffer(f"a2_{k}_nW", a2.normW)
                self.register_buffer(f"a2_{k}_nB", a2.normB)
            self.register_buffer(f"al1_{k}", w.f32(f"kmodel.{base}.alpha1.{k}"))
            self.register_buffer(f"al2_{k}", w.f32(f"kmodel.{base}.alpha2.{k}"))
            c1w, c1b = load_conv1d(w, f"{base}.convs1.{k}")
            c2w, c2b = load_conv1d(w, f"{base}.convs2.{k}")
            self.register_buffer(f"c1W_{k}", c1w)
            self.register_buffer(f"c1B_{k}", c1b if c1b is not None else torch.zeros(c1w.shape[0]))
            self.register_buffer(f"c2W_{k}", c2w)
            self.register_buffer(f"c2B_{k}", c2b if c2b is not None else torch.zeros(c2w.shape[0]))

    def _ada(self, group: str, k: int, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        fc = AdaFC(
            fcW=getattr(self, f"{group}_{k}_fcW"),
            fcB=getattr(self, f"{group}_{k}_fcB"),
            normW=getattr(self, f"{group}_{k}_nW", None),
            normB=getattr(self, f"{group}_{k}_nB", None),
        )
        return ada_in_1d(x, style, fc)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        out = x
        for k, d in enumerate(self.dilations):
            h = out
            h = self._ada("a1", k, h, style)
            h = snake_1d(h, getattr(self, f"al1_{k}"))
            c1W = getattr(self, f"c1W_{k}")
            K = c1W.shape[-1]
            h = conv1d_ncl(h, c1W, getattr(self, f"c1B_{k}"),
                           padding=d * (K - 1) // 2, dilation=d)
            h = self._ada("a2", k, h, style)
            h = snake_1d(h, getattr(self, f"al2_{k}"))
            c2W = getattr(self, f"c2W_{k}")
            K2 = c2W.shape[-1]
            h = conv1d_ncl(h, c2W, getattr(self, f"c2B_{k}"),
                           padding=(K2 - 1) // 2)
            out = out + h
        return out


# =============================================================================
# iSTFT head
# =============================================================================


def istft_head(conv_post_out: torch.Tensor, w: WeightBag,
               trim: int = 10) -> torch.Tensor:
    """iSTFT head matching the ONNX graph exactly.

    ONNX nests sin() on phase (Sin -> Sin/Cos) — NOT an export bug, it's what
    the model actually computes. Output trim is 10 samples from each side
    (Slice[10:-10]), giving (T-20) samples where T = 5*T_conv_post + 15.
    """
    mag_logits = conv_post_out[:, 0:11, :]
    phase = conv_post_out[:, 11:22, :]
    mag = torch.exp(mag_logits)
    inner = torch.sin(phase)
    real = mag * torch.cos(inner)
    imag = mag * torch.sin(inner)
    wReal = w.f32("kmodel.decoder.generator.stft.weight_backward_real")
    wImag = w.f32("kmodel.decoder.generator.stft.weight_backward_imag")
    audio_real = F.conv_transpose1d(real, wReal, stride=5, padding=0)
    audio_imag = F.conv_transpose1d(imag, wImag, stride=5, padding=0)
    audio = audio_real - audio_imag  # (1, 1, T)
    T = audio.shape[-1]
    return audio[:, 0, trim : T - trim].reshape(-1)


# =============================================================================
# Noise path (F0 -> SineGen -> STFT -> noise_convs -> noise_res)
# =============================================================================


def compute_noise_contribs(f0_proj: torch.Tensor, n_frames: int,
                           acoustic_style: torch.Tensor, w: WeightBag,
                           nr0: "AdaINResBlockHiFiGAN", nr1: "AdaINResBlockHiFiGAN",
                           seed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    T_frames = n_frames * 2
    hop = 300
    T_audio = T_frames * hop
    sr = 24000.0
    sine_amp = 0.1
    noise_std = 0.003

    f0_expand = f0_proj.reshape(1, 1, T_frames, 1)
    f0_bcast = f0_expand.expand(1, 1, T_frames, hop)
    f0_audio = f0_bcast.reshape(1, 1, T_audio)
    voiced = (f0_audio > 0).to(torch.float32)

    harmonics = torch.arange(1, 10, dtype=torch.float32, device=f0_proj.device).reshape(1, 9, 1)

    # Per-frame phase accumulation.
    #
    # Original formulation summed phase_inc (length T_audio = n_frames*2*hop,
    # up to ~160k samples) via a single long cumsum. In fp32 this accumulates
    # ~10^-3 abs error over the full run; coremltools' cumsum rewrite drifts
    # even more, which causes +30 dB spurious peaks at voice harmonics in
    # the exported model. Since f0 is piecewise-constant per frame, we can
    # (a) cumsum over just T_frames frame-sized steps (one per frame) and
    # (b) add a within-frame linear phase for samples inside each frame.
    # That shrinks the long-axis cumsum by `hop`= 300× and eliminates the
    # drift both in torch fp32 and in the CoreML export.
    f0_per_frame = f0_proj.reshape(1, 1, T_frames) * harmonics      # (1, 9, T_frames)
    step = f0_per_frame * (hop / sr)                                # cycles per frame
    phase_start = (step.cumsum(dim=-1) - step) * (2.0 * math.pi)    # (1, 9, T_frames)

    t_in_frame = torch.arange(hop, dtype=torch.float32,
                              device=f0_proj.device).reshape(1, 1, 1, hop)
    phase_within = (f0_per_frame.unsqueeze(-1) * t_in_frame) / sr * (2.0 * math.pi)
    phase = (phase_start.unsqueeze(-1) + phase_within).reshape(1, 9, T_audio)

    # The original upstream model added random phase_jitter per harmonic
    # and unvoiced gaussian noise to the sine generator as a dither pass.
    # coremltools maps torch.rand / torch.randn to CoreML's own RNG, which
    # draws a completely independent stream from torch's — so the exported
    # model produced audibly different audio each run and diverged from
    # the MLX / PyTorch references. The dither is at sine_amp * O(1) rad
    # for phase and noise_std = 0.003 for uv_noise — both imperceptible
    # in speech, so zeroing them keeps the CoreML export bit-reproducible
    # and keeps torch / coreml / mlx on the same output (up to ~2 % RMS).
    _ = seed; _ = noise_std
    phase_jitter = torch.zeros(1, 9, 1, device=f0_proj.device)
    uv_noise = torch.zeros(1, 9, T_audio, device=f0_proj.device)
    sines = torch.sin(phase + phase_jitter) * sine_amp
    sin_gen = sines * voiced + uv_noise * (1.0 - voiced)

    l_linW = w.f32("onnx::MatMul_6116")
    if l_linW is None:
        l_linW = w.f32("onnx::MatMul_6388")
    l_linB = w.bias("decoder.generator.m_source.l_linear")
    mixed = sin_gen.transpose(1, 2) @ l_linW + l_linB  # (1, T, 1)
    excitation = torch.tanh(mixed.transpose(1, 2))     # (1, 1, T_audio)

    stftR = w.f32("kmodel.decoder.generator.stft.weight_forward_real")
    stftI = w.f32("kmodel.decoder.generator.stft.weight_forward_imag")
    stft_real = conv1d_ncl(excitation, stftR, None, stride=5, padding=10)
    stft_imag = conv1d_ncl(excitation, stftI, None, stride=5, padding=10)
    mag = torch.sqrt(stft_real * stft_real + stft_imag * stft_imag + 1e-9)
    phi = torch.atan2(stft_imag, stft_real)
    stft_out = torch.cat([mag, phi], dim=1)  # (1, 22, T_stft)

    nc0W, nc0B = load_conv1d(w, "decoder.generator.noise_convs.0")
    nc0 = conv1d_ncl(stft_out, nc0W, nc0B, stride=6, padding=3)
    nc1W, nc1B = load_conv1d(w, "decoder.generator.noise_convs.1")
    nc1 = conv1d_ncl(stft_out, nc1W, nc1B)

    return nr0(nc0, acoustic_style), nr1(nc1, acoustic_style)


# =============================================================================
# Decoder + Generator pipelines
# =============================================================================


class DecoderPipeline(nn.Module):
    def __init__(self, w: WeightBag):
        super().__init__()
        asrW, asrB = load_conv1d(w, "decoder.asr_res.0")
        self.register_buffer("asrW", asrW)
        self.register_buffer("asrB", asrB if asrB is not None else torch.zeros(asrW.shape[0]))
        f0W, f0B = load_conv1d(w, "decoder.F0_conv")
        self.register_buffer("f0W", f0W)
        self.register_buffer("f0B", f0B if f0B is not None else torch.zeros(f0W.shape[0]))
        nW, nB = load_conv1d(w, "decoder.N_conv")
        self.register_buffer("nW", nW)
        self.register_buffer("nB", nB if nB is not None else torch.zeros(nW.shape[0]))
        self.encode = AdaINResBlock1D(w, "decoder.encode", upsample=False, divide=True, has_conv1x1=True)
        self.decode = nn.ModuleList([
            AdaINResBlock1D(w, f"decoder.decode.{i}", upsample=(i == 3), divide=True, has_conv1x1=True)
            for i in range(4)
        ])

    def forward(self, text_features_ncl: torch.Tensor, f0_ncl: torch.Tensor,
                n_ncl: torch.Tensor, acoustic_style: torch.Tensor) -> torch.Tensor:
        asr = conv1d_ncl(text_features_ncl, self.asrW, self.asrB, padding=0)
        f0_dn = conv1d_ncl(f0_ncl, self.f0W, self.f0B, stride=2, padding=1)
        n_dn = conv1d_ncl(n_ncl, self.nW, self.nB, stride=2, padding=1)
        enc_in = torch.cat([text_features_ncl, f0_dn, n_dn], dim=1)
        x = self.encode(enc_in, acoustic_style, shortcut_input=enc_in)
        for blk in self.decode:
            x_cat = torch.cat([x, asr, f0_dn, n_dn], dim=1)
            x = blk(x_cat, acoustic_style, shortcut_input=x_cat)
        return x


class GeneratorPipeline(nn.Module):
    def __init__(self, w: WeightBag):
        super().__init__()
        self.register_buffer("u0W", w.f32("kmodel.decoder.generator.ups.0.weight"))
        self.register_buffer("u0B", w.f32("kmodel.decoder.generator.ups.0.bias"))
        self.r0 = AdaINResBlockHiFiGAN(w, "decoder.generator.resblocks.0")
        self.r1 = AdaINResBlockHiFiGAN(w, "decoder.generator.resblocks.1")
        self.register_buffer("u1W", w.f32("kmodel.decoder.generator.ups.1.weight"))
        self.register_buffer("u1B", w.f32("kmodel.decoder.generator.ups.1.bias"))
        self.r2 = AdaINResBlockHiFiGAN(w, "decoder.generator.resblocks.2")
        self.r3 = AdaINResBlockHiFiGAN(w, "decoder.generator.resblocks.3")
        cpW, cpB = load_conv1d(w, "decoder.generator.conv_post")
        self.register_buffer("cpW", cpW)
        self.register_buffer("cpB", cpB if cpB is not None else torch.zeros(cpW.shape[0]))
        self._w = w

    def forward(self, decoder_out_ncl: torch.Tensor, acoustic_style: torch.Tensor,
                noise_res0: torch.Tensor, noise_res1: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(decoder_out_ncl, 0.1)
        x = conv_transpose_1d_ncl(x, self.u0W, self.u0B, stride=10, padding=5)
        x = x + noise_res0
        r0 = self.r0(x, acoustic_style)
        r1 = self.r1(x, acoustic_style)
        x = (r0 + r1) / 2.0
        x = F.leaky_relu(x, 0.1)
        x = conv_transpose_1d_ncl(x, self.u1W, self.u1B, stride=6, padding=3)
        x = reflection_pad_left(x, 1)
        x = x + noise_res1
        r2 = self.r2(x, acoustic_style)
        r3 = self.r3(x, acoustic_style)
        x = (r2 + r3) / 2.0
        x = F.leaky_relu(x, 0.1)
        x = conv1d_ncl(x, self.cpW, self.cpB, padding=3)
        return istft_head(x, self._w)


# =============================================================================
# Top-level end-to-end
# =============================================================================


class KittenTTS(nn.Module):
    """End-to-end (monolithic) port matching TTS.swift kittenForward.

    Dynamic shapes (L depends on phoneme count, nFrames on durations) make
    this unsuitable for direct CoreML conversion — use TextStage +
    GeneratorStage for that. This class is the numerical reference.
    """

    def __init__(self, w: WeightBag):
        super().__init__()
        self._w = w
        self.bert = BertStack(w)
        self.register_buffer("beW", w.f32("onnx::MatMul_5818"))
        self.register_buffer("beB", w.bias("bert_encoder"))
        self.pred_text = PredictorTextEncoder(w)

        # predictor.lstm (shared duration/prosody LSTM).
        self.duration_lstm = load_onnx_bidir_lstm(w, "onnx::LSTM_5971", "onnx::LSTM_5972", "onnx::LSTM_5970")
        self.register_buffer("dpW", w.f32("onnx::MatMul_5973"))
        self.register_buffer("dpB", w.bias("predictor.duration_proj.linear_layer"))

        self.shared_lstm = load_onnx_bidir_lstm(w, "onnx::LSTM_6020", "onnx::LSTM_6021", "onnx::LSTM_6019")

        # F0 / N stacks on the generator side.
        self.f0_0 = AdaINResBlock1D(w, "predictor.F0.0", upsample=False, divide=True)
        self.f0_1 = AdaINResBlock1D(w, "predictor.F0.1", upsample=True,  divide=True)
        self.f0_2 = AdaINResBlock1D(w, "predictor.F0.2", upsample=False, divide=True)
        self.N_0 = AdaINResBlock1D(w, "predictor.N.0", upsample=False, divide=True)
        self.N_1 = AdaINResBlock1D(w, "predictor.N.1", upsample=True,  divide=True)
        self.N_2 = AdaINResBlock1D(w, "predictor.N.2", upsample=False, divide=True)
        f0pW, f0pB = load_conv1d(w, "predictor.F0_proj")
        self.register_buffer("f0pW", f0pW)
        self.register_buffer("f0pB", f0pB if f0pB is not None else torch.zeros(f0pW.shape[0]))
        npW, npB = load_conv1d(w, "predictor.N_proj")
        self.register_buffer("npW", npW)
        self.register_buffer("npB", npB if npB is not None else torch.zeros(npW.shape[0]))

        self.acoustic = AcousticTextEncoder(w)
        self.decoder = DecoderPipeline(w)

        self.noise_res0 = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.0")
        self.noise_res1 = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.1")
        self.generator = GeneratorPipeline(w)

    def forward(self, input_ids: torch.Tensor, style256: torch.Tensor,
                speed: float = 1.0, noise_seed: Optional[int] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        acoustic_style = style256[:, :128].to(torch.float32)
        prosodic_style = style256[:, 128:256].to(torch.float32)

        bert_out = self.bert(input_ids)
        prosody_in = bert_out @ self.beW + self.beB
        prosody = self.pred_text(prosody_in, prosodic_style)
        s_bcast = prosodic_style.reshape(B, 1, -1).expand(B, L, 128)
        prosody256 = torch.cat([prosody, s_bcast], dim=-1)
        prosody_ncl = prosody256.transpose(1, 2).contiguous()  # (1, 256, L)

        lstm_in = prosody_ncl.permute(2, 0, 1).contiguous().to(torch.float32)
        dy = self.duration_lstm(lstm_in)
        lstm_out = dy.permute(2, 0, 1, 3).reshape(1, L, 128)

        dur_logits = lstm_out @ self.dpW + self.dpB
        dur_sig = torch.sigmoid(dur_logits)
        dur_sum = dur_sig.sum(dim=-1)  # (1, L)
        dur_scaled = dur_sum[0] / speed
        durs = torch.maximum(torch.tensor(1, dtype=torch.int32),
                             torch.round(dur_scaled).to(torch.int32))
        n_frames = int(durs.sum().item())

        # Length regulation (alignment matrix).
        align = torch.zeros(1, L, n_frames, dtype=torch.float32, device=input_ids.device)
        j = 0
        for i, d in enumerate(durs.tolist()):
            align[0, i, j : j + int(d)] = 1.0
            j += int(d)

        prosody_lr_ncl = prosody_ncl @ align  # (1, 256, nFrames)

        shared_in = prosody_lr_ncl.permute(2, 0, 1).contiguous().to(torch.float32)
        sy = self.shared_lstm(shared_in)
        fn_lstm_nlc = sy.permute(2, 0, 1, 3).reshape(1, n_frames, 128)
        fn_in_ncl = fn_lstm_nlc.transpose(1, 2).contiguous()

        f0 = self.f0_0(fn_in_ncl, prosodic_style, shortcut_input=fn_in_ncl)
        f0 = self.f0_1(f0, prosodic_style, shortcut_input=f0)
        f0 = self.f0_2(f0, prosodic_style, shortcut_input=f0)
        f0_proj = conv1d_ncl(f0, self.f0pW, self.f0pB)

        n = self.N_0(fn_in_ncl, prosodic_style, shortcut_input=fn_in_ncl)
        n = self.N_1(n, prosodic_style, shortcut_input=n)
        n = self.N_2(n, prosodic_style, shortcut_input=n)
        n_proj = conv1d_ncl(n, self.npW, self.npB)

        text_features_ncl = self.acoustic(input_ids)
        text_lr_ncl = text_features_ncl @ align

        dec_out = self.decoder(text_lr_ncl, f0_proj, n_proj, acoustic_style)

        nr0, nr1 = compute_noise_contribs(
            f0_proj, n_frames, acoustic_style, self._w,
            self.noise_res0, self.noise_res1, seed=noise_seed)

        wav = self.generator(dec_out, acoustic_style, nr0, nr1)
        return wav, durs


# =============================================================================
# Stage modules for CoreML conversion
# =============================================================================
#
# Splitting kittenForward into two CoreML models sidesteps the dynamic-shape
# pain caused by length regulation. Swift computes durations, builds the
# alignment matrix, and expands prosody_ncl + text_features_ncl between the
# stages. Each stage has a single dynamic axis:
#     TextStage:      L (phoneme count)
#     GeneratorStage: nFrames (after length regulation)


class TextStage(nn.Module):
    """Static-on-L half of kittenForward.

    Outputs:
        prosody_ncl      (1, 256, L)  — feed to length regulation
        text_features    (1, 128, L)  — feed to length regulation
        dur_sig          (1, L, 50)   — Swift sums+rounds to durations
    """

    def __init__(self, w: WeightBag):
        super().__init__()
        self.bert = BertStack(w)
        self.register_buffer("beW", w.f32("onnx::MatMul_5818"))
        self.register_buffer("beB", w.bias("bert_encoder"))
        self.pred_text = PredictorTextEncoder(w)
        self.duration_lstm = load_onnx_bidir_lstm(
            w, "onnx::LSTM_5971", "onnx::LSTM_5972", "onnx::LSTM_5970")
        self.register_buffer("dpW", w.f32("onnx::MatMul_5973"))
        self.register_buffer("dpB", w.bias("predictor.duration_proj.linear_layer"))
        self.acoustic = AcousticTextEncoder(w)

    def forward(self, input_ids: torch.Tensor, style256: torch.Tensor,
                attention_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prosodic_style = style256[:, 128:256]

        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        prosody_in = bert_out @ self.beW + self.beB
        prosody = self.pred_text(prosody_in, prosodic_style,
                                 attention_mask=attention_mask)
        s_bcast = prosodic_style.unsqueeze(1) + torch.zeros_like(prosody)
        prosody256 = torch.cat([prosody, s_bcast], dim=-1)
        prosody_ncl = prosody256.transpose(1, 2).contiguous()

        lstm_in = prosody_ncl.permute(2, 0, 1).contiguous()
        lstm_mask = attention_mask.transpose(0, 1)
        dy = self.duration_lstm(lstm_in, mask=lstm_mask)
        lstm_out = dy.permute(2, 0, 1, 3).reshape(1, -1, 128)
        dur_logits = lstm_out @ self.dpW + self.dpB
        dur_sig = torch.sigmoid(dur_logits)

        text_features_ncl = self.acoustic(input_ids,
                                          attention_mask=attention_mask)
        return prosody_ncl, text_features_ncl, dur_sig


class GeneratorStage(nn.Module):
    """Static-on-nFrames half of kittenForward.

    Inputs are the length-regulated prosody + text features (Swift has
    already applied the alignment matrix). Noise is generated inside — the
    first CoreML pass will be non-deterministic, which matches the ONNX
    model's behaviour.
    """

    def __init__(self, w: WeightBag):
        super().__init__()
        self._w = w
        self.shared_lstm = load_onnx_bidir_lstm(
            w, "onnx::LSTM_6020", "onnx::LSTM_6021", "onnx::LSTM_6019")
        self.f0_0 = AdaINResBlock1D(w, "predictor.F0.0", upsample=False, divide=True)
        self.f0_1 = AdaINResBlock1D(w, "predictor.F0.1", upsample=True,  divide=True)
        self.f0_2 = AdaINResBlock1D(w, "predictor.F0.2", upsample=False, divide=True)
        self.N_0 = AdaINResBlock1D(w, "predictor.N.0", upsample=False, divide=True)
        self.N_1 = AdaINResBlock1D(w, "predictor.N.1", upsample=True,  divide=True)
        self.N_2 = AdaINResBlock1D(w, "predictor.N.2", upsample=False, divide=True)
        f0pW, f0pB = load_conv1d(w, "predictor.F0_proj")
        self.register_buffer("f0pW", f0pW)
        self.register_buffer("f0pB", f0pB if f0pB is not None else torch.zeros(f0pW.shape[0]))
        npW, npB = load_conv1d(w, "predictor.N_proj")
        self.register_buffer("npW", npW)
        self.register_buffer("npB", npB if npB is not None else torch.zeros(npW.shape[0]))
        self.decoder = DecoderPipeline(w)
        self.noise_res0 = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.0")
        self.noise_res1 = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.1")
        self.generator = GeneratorPipeline(w)

    def forward(self, prosody_lr_ncl: torch.Tensor, text_lr_ncl: torch.Tensor,
                style256: torch.Tensor) -> torch.Tensor:
        acoustic_style = style256[:, :128]
        prosodic_style = style256[:, 128:256]

        shared_in = prosody_lr_ncl.permute(2, 0, 1).contiguous()
        sy = self.shared_lstm(shared_in)
        n_frames = shared_in.shape[0]
        fn_lstm_nlc = sy.permute(2, 0, 1, 3).reshape(1, n_frames, 128)
        fn_in_ncl = fn_lstm_nlc.transpose(1, 2).contiguous()

        f0 = self.f0_0(fn_in_ncl, prosodic_style, shortcut_input=fn_in_ncl)
        f0 = self.f0_1(f0, prosodic_style, shortcut_input=f0)
        f0 = self.f0_2(f0, prosodic_style, shortcut_input=f0)
        f0_proj = conv1d_ncl(f0, self.f0pW, self.f0pB)

        n = self.N_0(fn_in_ncl, prosodic_style, shortcut_input=fn_in_ncl)
        n = self.N_1(n, prosodic_style, shortcut_input=n)
        n = self.N_2(n, prosodic_style, shortcut_input=n)
        n_proj = conv1d_ncl(n, self.npW, self.npB)

        dec_out = self.decoder(text_lr_ncl, f0_proj, n_proj, acoustic_style)

        nr0, nr1 = compute_noise_contribs(
            f0_proj, n_frames, acoustic_style, self._w,
            self.noise_res0, self.noise_res1, seed=None)

        return self.generator(dec_out, acoustic_style, nr0, nr1)

import Foundation
import MLX
import MLXNN
// CEPhonemizer's C API is exposed via the bridging header — no import.

public final nonisolated class KittenTTS: @unchecked Sendable {
    static let weightsFilename = "kitten_tts_nano_v0_8.safetensors"

    public nonisolated struct Config: Sendable {
        public var speed:   Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed   = speed
            self.voiceID = voiceID
        }
    }

    private var cachedWeights: Weights?
    private var cachedVoices: [String: MLXArray]?

    public static let voiceAliases: [String: String] = [
        "Bella":  "expr-voice-2-f", "Jasper": "expr-voice-2-m",
        "Luna":   "expr-voice-3-f", "Bruno":  "expr-voice-3-m",
        "Rosie":  "expr-voice-4-f", "Hugo":   "expr-voice-4-m",
        "Kiki":   "expr-voice-5-f", "Leo":    "expr-voice-5-m",
    ]
    /// Per-voice speed multipliers from upstream KittenTTS.
    static let speedPriors: [String: Float] = [
        "expr-voice-2-f": 0.8, "expr-voice-2-m": 0.8,
        "expr-voice-3-m": 0.8, "expr-voice-3-f": 0.8,
        "expr-voice-4-m": 0.9, "expr-voice-4-f": 0.8,
        "expr-voice-5-m": 0.8, "expr-voice-5-f": 0.8,
    ]
    // Ordered by subjective cleanness (informal listening): Hugo and
    // Luna ship the cleanest prosody on the current models, so they
    // lead.
    public static let voiceDisplayOrder: [String] = [
        "Hugo",  "Luna",   "Kiki",  "Leo",
        "Bella", "Jasper", "Bruno", "Rosie",
    ]

    public init() {
        Self.installMetalLib()
    }

    public typealias SpeakCallback =
        (UnsafePointer<Int16>, Int) -> Void

    /// Per-chunk runtime metrics. Same shape idea as
    /// KittenTTSCoreML.ChunkMetrics but without bucket fields (MLX
    /// runs at the actual L / nFrames).
    public struct ChunkMetrics: Sendable {
        public let phonemes:  Int
        public let elapsedMs: Double
        public let samples:   Int
    }
    public var onChunkMetrics: ((ChunkMetrics) -> Void)?

    /// Copy bundled mlx.metallib next to the running binary so MLX's
    /// colocated-search finds it without any env vars or path hacks.
    ///
    /// Needed only for the bare-executable SwiftPM CLI layout
    /// (`.build/…/debug/KittenApp` — no surrounding `.app`). Inside a
    /// macOS `.app`, mlx-swift's own `mlx-swift_Cmlx.bundle` already
    /// ships `default.metallib` where MLX expects it, and an extra
    /// copy next to the executable just creates an unsigned file that
    /// trips the incremental CodeSign phase on subsequent builds.
    // Called at-most-once from the single-threaded init() path;
    // unsafe is fine.
    private nonisolated(unsafe) static var metalLibInstalled = false
    /// Internal so CoreMLBackend can invoke the install side effect
    /// directly instead of constructing a throwaway `KittenTTS()` just
    /// to trigger it.
    static func installMetalLib() {
        if !metalLibInstalled {
            metalLibInstalled = true
            if Bundle.main.bundleURL.pathExtension != "app" {
                if let src = Bundle.main.url(
                        forResource: "mlx",
                        withExtension: "metallib") {
                    let argv0 = CommandLine.arguments[0]
                    let binDir = URL(fileURLWithPath: argv0)
                                     .deletingLastPathComponent()
                    let dst = binDir.appendingPathComponent(
                        "mlx.metallib")
                    if !FileManager.default.fileExists(
                            atPath: dst.path) {
                        try? FileManager.default.copyItem(
                            at: src, to: dst)
                    }
                }
            }
        }
    }

    /// Pre-load model weights so the first `speak` call starts
    /// instantly.
    public func preload(modelPath: String? = nil) async throws {
        try Device.withDefaultDevice(.gpu) {
            _ = try loadWeightsIfNeeded(modelPath: modelPath)
        }
    }

    /// Release cached safetensors + voice arrays and flush the MLX
    /// cache. Call this when switching to a different backend to free
    /// RAM.
    public func unload() {
        cachedWeights = nil
        cachedVoices = nil
        MLX.Memory.clearCache()
    }

    private func loadWeightsIfNeeded(modelPath: String? = nil)
            throws -> (Weights, [String: MLXArray]) {
        let result: (Weights, [String: MLXArray])
        if let w = cachedWeights, let v = cachedVoices {
            result = (w, v)
        } else {
            let modelDir: URL
            if let path = modelPath {
                modelDir = URL(fileURLWithPath: path)
            } else if let bundledDir = ModelLoader.bundledModelDir() {
                modelDir = bundledDir
            } else {
                throw NSError(
                    domain: "KittenTTS", code: 2,
                    userInfo: [NSLocalizedDescriptionKey:
                        "No model found. Call preload() or " +
                        "provide modelPath."])
            }
            let rawWeights = try loadArrays(
                url: modelDir.appendingPathComponent(
                    Self.weightsFilename))
            let voices = try loadArrays(
                url: modelDir.appendingPathComponent(
                    "voices.safetensors"))
            let weights = Weights(rawWeights)
            cachedWeights = weights
            cachedVoices = voices
            result = (weights, voices)
        }
        return result
    }

    public func speak(
        text:      String,
        modelPath: String? = nil,
        config:    Config = Config(),
        callback:  SpeakCallback? = nil
    ) async throws -> [Float] {
        // Wrap the MLX evaluation in `withError` so MLX's internal
        // fatalError path (e.g. "[metal::Device] Unable to load
        // kernel steel_gemm_…" when the Metal compiler XPC service
        // crashes or gets jetsammed on memory-constrained iOS
        // devices) surfaces as a thrown Swift error instead of
        // trapping the whole app. The caller's catch can then fall
        // back to a working backend.
        return try withError {
            try Device.withDefaultDevice(.gpu) {
                let (weights, voices) = try loadWeightsIfNeeded(
                    modelPath: modelPath)
                let voiceID = KittenTTS
                                  .voiceAliases[config.voiceID]
                                  ?? config.voiceID
                let voiceEmbeds: MLXArray
                if let v = voices[voiceID]
                        ?? voices["expr-voice-5-m"] {
                    voiceEmbeds = v
                } else {
                    throw NSError(
                        domain: "KittenTTS", code: 1,
                        userInfo: [NSLocalizedDescriptionKey:
                            "Voice '\(voiceID)' not found"])
                }
                let effectiveSpeed = config.speed
                    * (KittenTTS.speedPriors[voiceID] ?? 1.0)
                let normalised = TextPreprocessor.process(text)
                // MLX uses dynamic shapes — no L=400 bucket like
                // CoreML. Pass `.max` so the chunker only breaks on
                // `.!?` and never on commas; whole sentences stay
                // together regardless of length. Eliminates the
                // mid-sentence comma-split pause.
                let chunks = TextChunker.chunk(normalised,
                                               maxLen: .max)
                // 120 ms inter-sentence silence — only between true
                // sentence breaks (prev chunk ends in `.!?`).
                // Sub-clause boundaries (`,;:` left mid-sentence by
                // the maxLen overflow split, or a soft-comma
                // terminator) get no gap so the listener doesn't
                // hear an unexpected pause inside one thought.
                let gap = [Float](repeating: 0,
                                  count: Int(0.12 * 24000))
                func gapAfter(_ prev: String) -> [Float] {
                    var g: [Float] = []
                    if let last = prev.last,
                       last == "." || last == "!" || last == "?" {
                        g = gap
                    }
                    return g
                }
                var allAudio: [Float] = []
                for (idx, chunk) in chunks.enumerated() {
                    let phonemes = try Phonemizer.phonemize(chunk)
                    let inputIds = MLXArray(
                        phonemes.map { p in Int32(p) })
                            .reshaped([1, -1])
                    let refId = min(chunk.count,
                                    voiceEmbeds.dim(0) - 1)
                    let style = voiceEmbeds[refId...refId]
                                    .reshaped([1, -1])
                    let tStart = Date()
                    let audio = kittenForward(
                        weights:  weights,
                        inputIds: inputIds,
                        style256: style,
                        speed:    effectiveSpeed)
                    let samples: [Float] = audio.asArray(Float.self)
                    let elapsedMs = Date().timeIntervalSince(tStart)
                                        * 1000.0
                    onChunkMetrics?(ChunkMetrics(
                        phonemes:  phonemes.count,
                        elapsedMs: elapsedMs,
                        samples:   samples.count))
                    let emit: [Float] = idx == 0
                        ? samples
                        : gapAfter(chunks[idx - 1]) + samples
                    if let cb = callback {
                        let int16Samples = emit.map { f -> Int16 in
                            // NaN/Inf guard before Int(...) which
                            // traps on non-finite.
                            let safe: Float = f.isFinite ? f : 0
                            return Int16(clamping:
                                Int(round(safe * 32767.0)))
                        }
                        int16Samples.withUnsafeBufferPointer { buf in
                            if let base = buf.baseAddress {
                                cb(base, buf.count)
                            }
                        }
                    }
                    allAudio.append(contentsOf: emit)
                }
                return allAudio
            }
        }
    }
}

// All KittenTTS weights live under kmodel.* prefixes inside the single
// safetensors file. Most parameters exist as a (quantized, scale,
// zero_point) triple; a handful live as plain float tensors.
// `Weights` hides that plumbing and exposes name-based fetches in the
// ONNX layout. Layout conversion is done inside the NCL Conv helpers
// below so lookups stay verbatim with the Python reference in
// `kitten_mlx.py`.

struct Weights {
    let raw: [String: MLXArray]

    init(_ raw: [String: MLXArray]) {
        self.raw = raw
    }

    subscript(key: String) -> MLXArray? { raw[key] }

    func f32(_ key: String) -> MLXArray? {
        raw[key].map { a in a.asType(.float32) }
    }

    /// Dequantize `kmodel.{base}.weight` using
    /// `weight_quantized/scale/zero_point` if present, otherwise
    /// returns the plain float `weight`.
    func dequant(_ base: String) -> MLXArray? {
        let prefix = "kmodel.\(base)"
        let result: MLXArray?
        if let wq = raw["\(prefix).weight_quantized"],
           let s = raw["\(prefix).weight_scale"],
           let z = raw["\(prefix).weight_zero_point"] {
            result = (wq.asType(.float32) - z.asType(.float32))
                   * s.asType(.float32)
        } else {
            result = raw["\(prefix).weight"]?.asType(.float32)
        }
        return result
    }

    func bias(_ base: String) -> MLXArray? {
        raw["kmodel.\(base).bias"]?.asType(.float32)
    }

    /// Dequantize an arbitrary `{base}_quantized/_scale/_zero_point`
    /// triple.
    func dequantRaw(_ base: String) -> MLXArray? {
        let result: MLXArray?
        if let wq = raw["\(base)_quantized"],
           let s = raw["\(base)_scale"],
           let z = raw["\(base)_zero_point"] {
            result = (wq.asType(.float32) - z.asType(.float32))
                   * s.asType(.float32)
        } else {
            result = raw[base]?.asType(.float32)
        }
        return result
    }

    /// Dequantize a DynamicQuantizeLSTM matrix stored as
    /// `(num_dir, in, 4H)` int8 with per-direction scalar scale/zp.
    /// Returned shape `(num_dir, 4H, in)` — ONNX LSTM spec layout.
    func dequantLSTM(_ base: String) -> MLXArray? {
        let result: MLXArray?
        if let wq = raw["\(base)_quantized"],
           let s = raw["\(base)_scale"],
           let z = raw["\(base)_zero_point"] {
            let sr = s.asType(.float32).reshaped([-1, 1, 1])
            let zr = z.asType(.float32).reshaped([-1, 1, 1])
            let dq = (wq.asType(.float32) - zr) * sr
            result = dq.transposed(0, 2, 1)
        } else {
            result = nil
        }
        return result
    }
}

private func leakyReLU(_ x: MLXArray, slope: Float) -> MLXArray {
    return MLX.where(x .> 0, x, x * slope)
}

/// Snake activation: `x + (1/alpha) * sin(alpha*x)^2`. `alpha` is
/// (C,) and broadcasts on NCL.
private func snake1D(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let a = alpha.reshaped([1, -1, 1])
    let inner = MLX.sin(a * x)
    return x + (1.0 / a) * inner * inner
}

/// Reflection pad on the LEFT by `n` samples along the last axis.
/// Matches the KittenTTS generator reflection pad between ups.1 and
/// the MRF stack.
private func reflectionPadLeft(_ x: MLXArray, _ n: Int) -> MLXArray {
    let result: MLXArray
    if n > 0 {
        // Reflect indices: for n=1, pad element is x[..,1] (skip the
        // boundary).
        let reflected = x[.ellipsis, 1...n]
        // Python slicing `x[..,1:n+1]` then reverse along last axis ->
        // left pad.
        let reversed = reflected[.ellipsis, .stride(by: -1)]
        result = MLX.concatenated([reversed, x], axis: -1)
    } else {
        result = x
    }
    return result
}

// ONNX stores Conv1d weights as (Cout, Cin, K) and MLX Conv1d
// operates on NLC tensors with weight (Cout, K, Cin). Since we're
// operating on many tensors whose layout is NCL (matching the Python
// port), and we need dilation support, we implement Conv1d /
// ConvTranspose1d directly on MLX arrays with numpy-style windowed
// matmuls.

/// Standard Conv1d on NCL tensor. Weight shape (Cout, Cin, K), bias
/// (Cout,).
private func conv1dNCL(_ x: MLXArray, weight: MLXArray,
                       bias: MLXArray?,
                       stride: Int = 1, padding: Int = 0,
                       dilation: Int = 1) -> MLXArray {
    var xp = x
    if padding > 0 {
        xp = MLX.padded(xp, widths: [IntOrPair(0),
                                     IntOrPair(0),
                                     IntOrPair(padding)])
    }
    let Cout = weight.dim(0)
    let K = weight.dim(-1)
    let B = xp.dim(0); let _ = xp.dim(1); let Lp = xp.dim(2)
    let Lout = (Lp - dilation * (K - 1) - 1) / stride + 1
    var acc: MLXArray? = nil
    for i in 0..<K {
        // Slice xp along last axis with step=stride, offset=i*dil,
        // length=Lout.
        let startIdx = i * dilation
        let endIdx = startIdx + Lout * stride
        let xs = xp[0..., 0...,
                    .stride(from: startIdx, to: endIdx,
                            by: stride)]
        // xs: (B, Cin, Lout). weight[:, :, i]: (Cout, Cin).
        let w_i = weight[0..., 0..., i]  // (Cout, Cin)
        // contribution: einsum('bcl,oc->bol', xs, w_i)
        // = matmul(xs.transposed(0,2,1), w_i.T).transposed(0,2,1)
        let contrib = MLX.matmul(xs.transposed(0, 2, 1),
                                 w_i.transposed(1, 0))
                          .transposed(0, 2, 1)
        acc = acc.map { a in a + contrib } ?? contrib
    }
    var out = acc ?? MLX.zeros([B, Cout, Lout])
    if let b = bias {
        out = out + b.reshaped([1, Cout, 1])
    }
    return out
}

/// Depthwise Conv1d on NCL. Weight (C, 1, K) — one filter per
/// channel.
private func conv1dDepthwiseNCL(_ x: MLXArray, weight: MLXArray,
                                bias: MLXArray?,
                                stride: Int = 1,
                                padding: Int = 0) -> MLXArray {
    var xp = x
    if padding > 0 {
        xp = MLX.padded(xp, widths: [IntOrPair(0),
                                     IntOrPair(0),
                                     IntOrPair(padding)])
    }
    let C = weight.dim(0); let K = weight.dim(-1)
    let Lp = xp.dim(2)
    let Lout = (Lp - K) / stride + 1
    var acc: MLXArray? = nil
    for i in 0..<K {
        let s = i; let e = i + Lout * stride
        let xs = xp[0..., 0...,
                    .stride(from: s, to: e, by: stride)]
        let w_i = weight[0..., 0, i].reshaped([1, C, 1])
        let contrib = xs * w_i
        acc = acc.map { a in a + contrib } ?? contrib
    }
    var out = acc ?? MLX.zeros([xp.dim(0), C, Lout])
    if let b = bias { out = out + b.reshaped([1, C, 1]) }
    return out
}

/// Standard (group=1) ConvTranspose1d on NCL. Weight ONNX layout
/// (Cin, Cout, K).
private func convTranspose1dNCL(_ x: MLXArray, weight: MLXArray,
                                bias: MLXArray?,
                                stride: Int = 1, padding: Int = 0,
                                outPadding: Int = 0) -> MLXArray {
    let B = x.dim(0); let _ = x.dim(1); let Lin = x.dim(2)
    let Cout = weight.dim(1); let K = weight.dim(2)
    let Lfull = (Lin - 1) * stride + K
    // Accumulate kernel-offset contributions into a zero-initialized
    // buffer.
    var buf = MLX.zeros([B, Cout, Lfull])
    for k in 0..<K {
        // contrib: (B, Cout, Lin) = einsum('bcl,co->bol', x, w[:,:,k])
        let w_k = weight[0..., 0..., k]  // (Cin, Cout)
        let contrib = MLX.matmul(x.transposed(0, 2, 1), w_k)
                          .transposed(0, 2, 1)
        // Place contribution into buf at positions
        // k, k+stride, k+2*stride, ...
        // Equivalent to: buf[:, :, k::stride][:, :, :Lin] += contrib
        // MLX has no scatter-add, so we build a one-hot mask and add.
        // For small K (≤20) and moderate Lfull, build-then-pad is
        // fine:
        let before = k
        let after = Lfull - k - (Lin - 1) * stride - 1
        if after >= 0 {
            let spread = zeroInterleaveWithStride(
                contrib, stride: stride,
                zerosBefore: before, zerosAfter: after)
            buf = buf + spread
        } else {
            // Shouldn't happen given Lfull definition.
            let spread = zeroInterleaveWithStride(
                contrib, stride: stride,
                zerosBefore: before, zerosAfter: 0)
            let trimmed = spread[0..., 0..., 0..<Lfull]
            buf = buf + trimmed
        }
    }
    let Lout = Lfull - 2 * padding + outPadding
    var res = buf[0..., 0..., padding..<(padding + Lout)]
    if res.dim(-1) < Lout {
        let pad = Lout - res.dim(-1)
        res = MLX.padded(res, widths: [IntOrPair(0),
                                       IntOrPair(0),
                                       IntOrPair((0, pad))])
    }
    if let b = bias { res = res + b.reshaped([1, Cout, 1]) }
    return res
}

/// Depthwise ConvTranspose1d on NCL (groups=C, weight (C, 1, K)).
private func convTranspose1dDepthwiseNCL(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray?,
        stride: Int = 2, padding: Int = 1, outPadding: Int = 1)
        -> MLXArray {
    let B = x.dim(0); let C = x.dim(1); let Lin = x.dim(2)
    let K = weight.dim(-1)
    let Lfull = (Lin - 1) * stride + K
    var buf = MLX.zeros([B, C, Lfull])
    for k in 0..<K {
        // (1, C, 1).
        let w_k = weight[0..., 0, k].reshaped([1, C, 1])
        let contrib = x * w_k  // (B, C, Lin)
        let before = k
        let after = Lfull - k - (Lin - 1) * stride - 1
        let spread = zeroInterleaveWithStride(
            contrib, stride: stride,
            zerosBefore: before, zerosAfter: max(0, after))
        let trimmed = spread[0..., 0..., 0..<Lfull]
        buf = buf + trimmed
    }
    let Lout = Lfull - 2 * padding + outPadding
    var res = buf[0..., 0..., padding..<(padding + Lout)]
    if res.dim(-1) < Lout {
        let pad = Lout - res.dim(-1)
        res = MLX.padded(res, widths: [IntOrPair(0),
                                       IntOrPair(0),
                                       IntOrPair((0, pad))])
    }
    if let b = bias { res = res + b.reshaped([1, C, 1]) }
    return res
}

/// Place `contrib` (B, C, Lin) at indices [zerosBefore,
/// zerosBefore+stride, ...] on a zero-filled track of length
/// zerosBefore + (Lin-1)*stride + 1 + zerosAfter. Implemented by
/// interleaving zeros then left-/right-padding.
private func zeroInterleaveWithStride(_ x: MLXArray, stride: Int,
                                      zerosBefore: Int,
                                      zerosAfter: Int) -> MLXArray {
    var y = x
    if stride > 1 {
        // Interleave (stride-1) zero samples between each input
        // sample along last axis.
        let B = y.dim(0); let C = y.dim(1); let Lin = y.dim(2)
        // Build (B, C, Lin, stride) where only index 0 holds x, rest
        // are zero.
        let zeros = MLX.zeros([B, C, Lin, stride - 1])
        let expanded = y.reshaped([B, C, Lin, 1])
        let stacked = MLX.concatenated([expanded, zeros], axis: -1)
        y = stacked.reshaped([B, C, Lin * stride])
        // Trim trailing stride-1 zeros so len == (Lin-1)*stride + 1.
        y = y[0..., 0..., 0..<((Lin - 1) * stride + 1)]
    }
    if zerosBefore > 0 || zerosAfter > 0 {
        y = MLX.padded(y, widths: [
            IntOrPair(0), IntOrPair(0),
            IntOrPair((zerosBefore, zerosAfter))])
    }
    return y
}

private func layerNormLast(_ x: MLXArray, weight: MLXArray?,
                           bias: MLXArray?,
                           eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let variance = x.variance(axis: -1, keepDims: true)
    var h = (x - mean) * MLX.rsqrt(variance + eps)
    if let w = weight { h = h * w }
    if let b = bias { h = h + b }
    return h
}

/// Channel-wise InstanceNorm1d on NCL (per-sample, per-channel over L
/// axis). No affine.
private func instanceNorm1DNCL(_ x: MLXArray,
                               eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let variance = x.variance(axis: -1, keepDims: true)
    return (x - mean) * MLX.rsqrt(variance + eps)
}

/// AdaIN1d. `style` (B, styleDim). `fc` projects to (B, 2C): split
/// into gamma/beta. Optional `normW`/`normB` apply affine after
/// InstanceNorm (for blocks whose PyTorch InstanceNorm was
/// constructed with affine=True).
private func adaIN1D(_ x: MLXArray, style: MLXArray,
                     fcW: MLXArray, fcB: MLXArray,
                     normW: MLXArray?,
                     normB: MLXArray?) -> MLXArray {
    // fcW stored as ONNX MatMul layout (styleDim, 2C);
    // style (B, styleDim).
    let h = MLX.matmul(style, fcW) + fcB  // (B, 2C)
    let C = h.dim(-1) / 2
    let gamma = h[0..., 0..<C].reshaped([-1, C, 1])
    let beta = h[0..., C..<(2 * C)].reshaped([-1, C, 1])
    var normed = instanceNorm1DNCL(x)
    if let nw = normW { normed = normed * nw.reshaped([1, -1, 1]) }
    if let nb = normB { normed = normed + nb.reshaped([1, -1, 1]) }
    return normed * (1.0 + gamma) + beta
}

/// AdaLayerNorm on NLC: LayerNorm on last dim then (1+gamma)*x + beta
/// with gamma,beta from fc(style).
private func adaLayerNorm(_ x: MLXArray, style: MLXArray,
                          fcW: MLXArray, fcB: MLXArray) -> MLXArray {
    let h = MLX.matmul(style, fcW) + fcB  // (B, 2C)
    let C = h.dim(-1) / 2
    let gamma = h[0..., 0..<C].reshaped([-1, 1, C])
    let beta = h[0..., C..<(2 * C)].reshaped([-1, 1, C])
    let normed = layerNormLast(x, weight: nil, bias: nil)
    return normed * (1.0 + gamma) + beta
}

private struct AdaFC {
    let fcW:   MLXArray
    let fcB:   MLXArray
    let normW: MLXArray?
    let normB: MLXArray?
}

private func loadAdaFC(_ w: Weights, base: String,
                       which: String) -> AdaFC {
    let fcBase = "\(base).\(which).fc"
    let fcW = (w.dequant(fcBase) ?? w.f32("kmodel.\(fcBase).weight"))!
    let fcB = w.bias(fcBase)!
    let nw = w.raw["kmodel.\(base).\(which).norm.weight"]?
                  .asType(.float32)
    let nb = w.raw["kmodel.\(base).\(which).norm.bias"]?
                  .asType(.float32)
    return AdaFC(fcW: fcW, fcB: fcB, normW: nw, normB: nb)
}

/// AdainResBlock1d: norm1 -> LeakyReLU -> [pool upsample] -> conv1 ->
/// norm2 -> LeakyReLU -> conv2, with a (1+residual)/sqrt(2) shortcut.
/// The shortcut can take a distinct tensor (the upstream block's
/// /sqrt(2)-normalized output).
private func adainResBlock1D(_ x: MLXArray, style: MLXArray,
                             weights w: Weights, base: String,
                             upsample: Bool = false,
                             divide: Bool = true,
                             shortcutInput: MLXArray? = nil,
                             hasConv1x1: Bool? = nil) -> MLXArray {
    let shortcut = shortcutInput ?? x
    let a1 = loadAdaFC(w, base: base, which: "norm1")
    let a2 = loadAdaFC(w, base: base, which: "norm2")
    var h = adaIN1D(x, style: style,
                    fcW: a1.fcW, fcB: a1.fcB,
                    normW: a1.normW, normB: a1.normB)
    h = leakyReLU(h, slope: 0.2)
    if upsample {
        let poolW = w.f32("kmodel.\(base).pool.weight")!
        let poolB = w.f32("kmodel.\(base).pool.bias")
        h = convTranspose1dDepthwiseNCL(
            h, weight: poolW, bias: poolB,
            stride: 2, padding: 1, outPadding: 1)
    }
    let (c1W, c1B) = loadConv1dNCL(w, base: "\(base).conv1")
    let K1 = c1W.dim(-1)
    h = conv1dNCL(h, weight: c1W, bias: c1B,
                  padding: (K1 - 1) / 2)
    h = adaIN1D(h, style: style,
                fcW: a2.fcW, fcB: a2.fcB,
                normW: a2.normW, normB: a2.normB)
    h = leakyReLU(h, slope: 0.2)
    let (c2W, c2B) = loadConv1dNCL(w, base: "\(base).conv2")
    let K2 = c2W.dim(-1)
    h = conv1dNCL(h, weight: c2W, bias: c2B,
                  padding: (K2 - 1) / 2)
    let shortcutActive: Bool = hasConv1x1 ?? (
        upsample
            || w.raw["kmodel.\(base).conv1x1.weight"] != nil
            || w.raw["kmodel.\(base).conv1x1.weight_quantized"]
                   != nil
    )
    let res: MLXArray
    if upsample {
        // Nearest-neighbor 2x on shortcut, then conv1x1.
        let repeated = MLX.repeated(shortcut, count: 2, axis: -1)
        let (sw, sb) = loadConv1dNCL(w, base: "\(base).conv1x1")
        res = conv1dNCL(repeated, weight: sw, bias: sb)
    } else if shortcutActive {
        let (sw, sb) = loadConv1dNCL(w, base: "\(base).conv1x1")
        res = conv1dNCL(shortcut, weight: sw, bias: sb)
    } else {
        res = shortcut
    }
    var out = h + res
    if divide { out = out / sqrt(Float(2.0)) }
    return out
}

private func loadConv1dNCL(_ w: Weights, base: String)
        -> (MLXArray, MLXArray?) {
    let weight = w.dequant(base)!
    let bias = w.bias(base)
    return (weight, bias)
}

private func adainResBlockHiFiGAN(_ x: MLXArray, style: MLXArray,
                                  weights w: Weights, base: String,
                                  dilations: [Int] = [1, 3, 5])
        -> MLXArray {
    var out = x
    for (k, d) in dilations.enumerated() {
        var h = out
        let a1 = loadAdaFC(w, base: base, which: "adain1.\(k)")
        h = adaIN1D(h, style: style,
                    fcW: a1.fcW, fcB: a1.fcB,
                    normW: a1.normW, normB: a1.normB)
        let alpha1 = w.f32("kmodel.\(base).alpha1.\(k)")!
        h = snake1D(h, alpha: alpha1)
        let (c1W, c1B) = loadConv1dNCL(
            w, base: "\(base).convs1.\(k)")
        let K = c1W.dim(-1)
        h = conv1dNCL(h, weight: c1W, bias: c1B,
                      padding: d * (K - 1) / 2, dilation: d)
        let a2 = loadAdaFC(w, base: base, which: "adain2.\(k)")
        h = adaIN1D(h, style: style,
                    fcW: a2.fcW, fcB: a2.fcB,
                    normW: a2.normW, normB: a2.normB)
        let alpha2 = w.f32("kmodel.\(base).alpha2.\(k)")!
        h = snake1D(h, alpha: alpha2)
        let (c2W, c2B) = loadConv1dNCL(
            w, base: "\(base).convs2.\(k)")
        let K2 = c2W.dim(-1)
        h = conv1dNCL(h, weight: c2W, bias: c2B,
                      padding: (K2 - 1) / 2)
        out = out + h
    }
    return out
}

/// ONNX LSTM forward. `x` (seq, batch, in). `W` (num_dir, 4H, in).
/// `R` (num_dir, 4H, H). `B` (num_dir, 8H) = [Wb | Rb] (summed
/// inside). Returns (seq, num_dir, batch, H).
private func onnxLSTMForward(_ x: MLXArray, W: MLXArray,
                             R: MLXArray, B: MLXArray,
                             bidirectional: Bool) -> MLXArray {
    let seqLen = x.dim(0); let batch = x.dim(1); let _ = x.dim(2)
    let H = W.dim(1) / 4
    func oneDir(_ d: Int, reverse: Bool) -> MLXArray {
        let Wd = W[d]  // (4H, in)
        let Rd = R[d]  // (4H, H)
        let Bd = B[d]  // (8H,)
        let Wb = Bd[0..<(4 * H)]
        let Rb = Bd[(4 * H)..<(8 * H)]
        let bias = (Wb + Rb).reshaped([1, -1])
        var h = MLX.zeros([batch, H])
        var c = MLX.zeros([batch, H])
        var outSteps: [MLXArray] = Array(
            repeating: MLX.zeros([batch, H]), count: seqLen)
        let range: StrideThrough<Int> = reverse
            ? stride(from: seqLen - 1, through: 0, by: -1)
            : stride(from: 0, through: seqLen - 1, by: 1)
        for t in range {
            let xt = x[t]  // (batch, in)
            let gates = MLX.matmul(xt, Wd.transposed(1, 0))
                      + MLX.matmul(h,  Rd.transposed(1, 0))
                      + bias
            let i = MLX.sigmoid(gates[0..., 0..<H])
            let o = MLX.sigmoid(gates[0..., H..<(2 * H)])
            let f = MLX.sigmoid(gates[0..., (2 * H)..<(3 * H)])
            let cHat = MLX.tanh(gates[0..., (3 * H)..<(4 * H)])
            c = f * c + i * cHat
            h = o * MLX.tanh(c)
            outSteps[t] = h
        }
        return MLX.stacked(outSteps, axis: 0)  // (seq, batch, H)
    }
    let result: MLXArray
    if !bidirectional {
        let yF = oneDir(0, reverse: false)  // (seq, batch, H)
        result = yF.reshaped([seqLen, 1, batch, H])
    } else {
        let yF = oneDir(0, reverse: false)
        let yR = oneDir(1, reverse: true)
        // (seq, 2, batch, H).
        result = MLX.stacked([yF, yR], axis: 1)
    }
    return result
}

private func loadDQLSTM(_ w: Weights, W: String, R: String,
                        B: String) -> (MLXArray, MLXArray, MLXArray) {
    let Wd = w.dequantLSTM(W)!
    let Rd = w.dequantLSTM(R)!
    let Bd = w.f32(B)!
    return (Wd, Rd, Bd)
}

/// iSTFT decoder head. `convPostOut` is (1, 22, T); first 11ch =
/// log-mag, last 11ch = phase. Returns waveform (numSamples,) after
/// trimming the tail by `tailTrim` samples to match upstream
/// kittentts (`audio[..., :-5000]`).
private func istftHead(_ convPostOut: MLXArray, weights w: Weights,
                       tailTrim: Int = 5000) -> MLXArray {
    let magLogits = convPostOut[0..., 0..<11, 0...]
    let phase = convPostOut[0..., 11..<22, 0...]
    let mag = MLX.exp(magLogits)
    let inner = MLX.sin(phase)  // KittenTTS export quirk
    let real = mag * MLX.cos(inner)
    let imag = mag * MLX.sin(inner)
    let wReal = w.f32(
        "kmodel.decoder.generator.stft.weight_backward_real")!
    let wImag = w.f32(
        "kmodel.decoder.generator.stft.weight_backward_imag")!
    // ConvTranspose1d (Cin=11 -> Cout=1, K=20, stride=5, pad=0).
    let audioReal = convTranspose1dNCL(real, weight: wReal,
                                       bias: nil,
                                       stride: 5, padding: 0)
    let audioImag = convTranspose1dNCL(imag, weight: wImag,
                                       bias: nil,
                                       stride: 5, padding: 0)
    let audio = audioReal - audioImag  // (1, 1, T)
    let T = audio.dim(-1)
    let end = max(0, T - tailTrim)
    let crop = audio[0..., 0, 0..<end]
    return crop.reshaped([-1])
}

private func bertForward(_ inputIds: MLXArray,
                         weights w: Weights) -> MLXArray {
    // Embeddings.
    let we = w.dequant("bert.embeddings.word_embeddings")!
    let pe = w.dequant("bert.embeddings.position_embeddings")!
    let te = w.dequant("bert.embeddings.token_type_embeddings")!
    let lnW = w.dequant("bert.embeddings.LayerNorm")!
    let lnB = w.bias("bert.embeddings.LayerNorm")!
    let L = inputIds.dim(1)
    let pos = MLXArray(Int32(0)..<Int32(L)).reshaped([1, -1])
    let wordEmb = we[inputIds]
    let posEmb = pe[pos]
    let typeEmb = te[MLX.zeros(like: inputIds)]
    var h = wordEmb + posEmb + typeEmb
    h = layerNormLast(h, weight: lnW, bias: lnB, eps: 1e-12)
    // mapping_in (128 -> 768). ONNX initializer:
    // onnx::MatMul_5661 shape (128,768).
    let mIn = w.f32("onnx::MatMul_5661")!
    let mInB = w.bias("bert.encoder.embedding_hidden_mapping_in")!
    h = MLX.matmul(h, mIn) + mInB
    // 12 iterations of shared albert layer. ONNX MatMul initializer
    // names resolved from verify_out/kitten_exposed.onnx.
    let base = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    let qW = w.f32("onnx::MatMul_5662")!      // (768, 768)
    let kW = w.f32("onnx::MatMul_5665")!      // (768, 768)
    let vW = w.f32("onnx::MatMul_5668")!      // (768, 768)
    let dW = w.f32("onnx::MatMul_5672")!      // (768, 768)
    let ffnW = w.f32("onnx::MatMul_5673")!    // (768, 2048)
    let ffnOutW = w.f32("onnx::MatMul_5674")! // (2048, 768)
    let qB = w.bias("\(base).attention.query")!
    let kB = w.bias("\(base).attention.key")!
    let vB = w.bias("\(base).attention.value")!
    let dB = w.bias("\(base).attention.dense")!
    let ffnB = w.bias("\(base).ffn")!
    let ffnOutB = w.bias("\(base).ffn_output")!
    let attLnW = w.dequant("\(base).attention.LayerNorm")!
    let attLnB = w.bias("\(base).attention.LayerNorm")!
    let fullLnW = w.dequant("\(base).full_layer_layer_norm")!
    let fullLnB = w.bias("\(base).full_layer_layer_norm")!
    for _ in 0..<12 {
        let B = h.dim(0); let Lh = h.dim(1)
        let D = 768; let nHead = 12; let dHead = 64
        let q = (MLX.matmul(h, qW) + qB)
                    .reshaped([B, Lh, nHead, dHead])
                    .transposed(0, 2, 1, 3)
        let k = (MLX.matmul(h, kW) + kB)
                    .reshaped([B, Lh, nHead, dHead])
                    .transposed(0, 2, 1, 3)
        let v = (MLX.matmul(h, vW) + vB)
                    .reshaped([B, Lh, nHead, dHead])
                    .transposed(0, 2, 1, 3)
        let scores = MLX.matmul(q, k.transposed(0, 1, 3, 2))
                   / sqrt(Float(dHead))
        let attn = MLX.softmax(scores, axis: -1)
        let ctx = MLX.matmul(attn, v)
                      .transposed(0, 2, 1, 3)
                      .reshaped([B, Lh, D])
        let attOut = MLX.matmul(ctx, dW) + dB
        let hMid = layerNormLast(attOut + h,
                                 weight: attLnW, bias: attLnB,
                                 eps: 1e-12)
        let ffnH = MLX.matmul(hMid, ffnW) + ffnB
        let ffnAct = MLXNN.gelu(ffnH)
        let ffnRes = MLX.matmul(ffnAct, ffnOutW) + ffnOutB
        h = layerNormLast(ffnRes + hMid,
                          weight: fullLnW, bias: fullLnB,
                          eps: 1e-12)
    }
    return h
}

private func predictorTextEncoder(_ bertOutNLC: MLXArray,
                                  prosodyStyle: MLXArray,
                                  weights w: Weights) -> MLXArray {
    let B = bertOutNLC.dim(0); let L = bertOutNLC.dim(1)
    let sBcast = MLX.broadcast(
        prosodyStyle.reshaped([B, 1, -1]),
        to: [B, L, prosodyStyle.dim(-1)])
    // lstms.0: concat(bert, style) (1, L, 256) -> bidir LSTM
    // (hidden 64) -> (1, L, 128)
    var x = MLX.concatenated([bertOutNLC, sBcast], axis: -1)
    var xT = x.transposed(1, 0, 2).asType(.float32)  // (L, 1, 256)
    var (W, R, Bv) = loadDQLSTM(w,
                                W: "onnx::LSTM_5872",
                                R: "onnx::LSTM_5873",
                                B: "onnx::LSTM_5871")
    // (L, 2, 1, 64).
    var y = onnxLSTMForward(xT, W: W, R: R, B: Bv,
                            bidirectional: true)
    let y0 = y.transposed(2, 0, 1, 3).reshaped([B, L, 128])
    // lstms.1: AdaLN
    let fc1W = w.f32(
        "kmodel.predictor.text_encoder.lstms.1.fc.weight")!
    let fc1B = w.f32(
        "kmodel.predictor.text_encoder.lstms.1.fc.bias")!
    let y1 = adaLayerNorm(y0, style: prosodyStyle,
                          fcW: fc1W, fcB: fc1B)
    // lstms.2: concat(y1, style) -> bidir LSTM -> (1, L, 128)
    x = MLX.concatenated([y1, sBcast], axis: -1)
    xT = x.transposed(1, 0, 2).asType(.float32)
    (W, R, Bv) = loadDQLSTM(w,
                            W: "onnx::LSTM_5922",
                            R: "onnx::LSTM_5923",
                            B: "onnx::LSTM_5921")
    y = onnxLSTMForward(xT, W: W, R: R, B: Bv,
                        bidirectional: true)
    let y2 = y.transposed(2, 0, 1, 3).reshaped([B, L, 128])
    // lstms.3: AdaLN
    let fc3W = w.f32(
        "kmodel.predictor.text_encoder.lstms.3.fc.weight")!
    let fc3B = w.f32(
        "kmodel.predictor.text_encoder.lstms.3.fc.bias")!
    return adaLayerNorm(y2, style: prosodyStyle,
                        fcW: fc3W, fcB: fc3B)
}

private func acousticTextEncoder(_ inputIds: MLXArray,
                                 weights w: Weights) -> MLXArray {
    let wEmb = w.f32("kmodel.text_encoder.embedding.weight")!
    var xNLC = wEmb[inputIds[0]]  // (L, 128)
    xNLC = xNLC.reshaped([1, -1, 128])
    for i in 0..<2 {
        let (conv, bias) = loadConv1dNCL(
            w, base: "text_encoder.cnn.\(i).0")
        let K = conv.dim(-1)
        var xNCL = xNLC.transposed(0, 2, 1)
        xNCL = conv1dNCL(xNCL, weight: conv, bias: bias,
                         padding: (K - 1) / 2)
        xNLC = xNCL.transposed(0, 2, 1)
        let gamma = w.f32(
            "kmodel.text_encoder.cnn.\(i).1.gamma")!
        let beta = w.f32(
            "kmodel.text_encoder.cnn.\(i).1.beta")!
        xNLC = layerNormLast(xNLC, weight: gamma, bias: beta)
        xNLC = leakyReLU(xNLC, slope: 0.2)
    }
    // text_encoder.lstm: bidir LSTM (in 128 -> hidden 64*2).
    let xT = xNLC.transposed(1, 0, 2).asType(.float32)
    let (W, R, Bv) = loadDQLSTM(w,
                                W: "onnx::LSTM_5652",
                                R: "onnx::LSTM_5653",
                                B: "onnx::LSTM_5651")
    // (L, 2, 1, 64).
    let y = onnxLSTMForward(xT, W: W, R: R, B: Bv,
                            bidirectional: true)
    let L = xT.dim(0)
    let yNLC = y.transposed(2, 0, 1, 3).reshaped([1, L, 128])
    return yNLC.transposed(0, 2, 1)  // (1, 128, L)
}

/// Build an alignment matrix (1, nPhonemes, nFrames):
/// A[0,i,j]=1 iff phoneme i spans frame j (block-diagonal one-hot).
/// Used as a matmul to duration-expand both prosody and acoustic
/// features.
private func buildAlignment(durations: [Int32],
                            nPhonemes: Int,
                            nFrames: Int) -> MLXArray {
    var flat = [Float](repeating: 0.0,
                       count: nPhonemes * nFrames)
    var t = 0
    for (i, dv) in durations.enumerated() {
        let d = Int(dv)
        for j in t..<(t + d) where j < nFrames {
            flat[i * nFrames + j] = 1.0
        }
        t += d
    }
    return MLXArray(flat).reshaped([1, nPhonemes, nFrames])
}

private func decoderPipeline(textFeaturesNCL: MLXArray,
                             f0NCL: MLXArray, nNCL: MLXArray,
                             acousticStyle: MLXArray,
                             weights w: Weights) -> MLXArray {
    // asr_res on text features (128 -> 64, k=1).
    let (asrW, asrB) = loadConv1dNCL(w, base: "decoder.asr_res.0")
    let asr = conv1dNCL(textFeaturesNCL,
                        weight: asrW, bias: asrB, padding: 0)
    // F0_conv / N_conv (stride 2, k=3, pad=1) downsample F0/N by 2.
    let (f0W, f0B) = loadConv1dNCL(w, base: "decoder.F0_conv")
    let f0Dn = conv1dNCL(f0NCL, weight: f0W, bias: f0B,
                         stride: 2, padding: 1)
    let (nW, nB) = loadConv1dNCL(w, base: "decoder.N_conv")
    let nDn = conv1dNCL(nNCL, weight: nW, bias: nB,
                        stride: 2, padding: 1)
    // encode input: concat(text (128), F0 (1), N (1)) = 130 channels.
    let encIn = MLX.concatenated([textFeaturesNCL, f0Dn, nDn],
                                 axis: 1)
    var x = adainResBlock1D(encIn, style: acousticStyle,
                            weights: w, base: "decoder.encode",
                            upsample: false, divide: true,
                            shortcutInput: encIn,
                            hasConv1x1: true)
    // decode.0..3: concat(prev, asr_res (64), F0 (1), N (1))
    // = 322 channels. decode.3 upsamples.
    for i in 0..<4 {
        let xCat = MLX.concatenated([x, asr, f0Dn, nDn], axis: 1)
        let isUp = (i == 3)
        x = adainResBlock1D(xCat, style: acousticStyle,
                            weights: w,
                            base: "decoder.decode.\(i)",
                            upsample: isUp, divide: true,
                            shortcutInput: xCat,
                            hasConv1x1: true)
    }
    // (1, 256 or 64 after decode.3, nFrames*2).
    return x
}

private func generatorPipeline(_ decoderOutNCL: MLXArray,
                               acousticStyle: MLXArray,
                               noiseRes0: MLXArray,
                               noiseRes1: MLXArray,
                               weights w: Weights) -> MLXArray {
    // LeakyReLU -> ups.0 (256->128, stride=10, k=20, pad=5).
    var x = leakyReLU(decoderOutNCL, slope: 0.1)
    let u0W = w.f32("kmodel.decoder.generator.ups.0.weight")!
    let u0B = w.f32("kmodel.decoder.generator.ups.0.bias")!
    x = convTranspose1dNCL(x, weight: u0W, bias: u0B,
                           stride: 10, padding: 5)
    x = x + noiseRes0
    // MRF: avg(resblocks.0, resblocks.1).
    let r0 = adainResBlockHiFiGAN(
        x, style: acousticStyle, weights: w,
        base: "decoder.generator.resblocks.0")
    let r1 = adainResBlockHiFiGAN(
        x, style: acousticStyle, weights: w,
        base: "decoder.generator.resblocks.1")
    x = (r0 + r1) / 2.0
    // LeakyReLU -> ups.1 (128->64, stride=6, k=12, pad=3).
    x = leakyReLU(x, slope: 0.1)
    let u1W = w.f32("kmodel.decoder.generator.ups.1.weight")!
    let u1B = w.f32("kmodel.decoder.generator.ups.1.bias")!
    x = convTranspose1dNCL(x, weight: u1W, bias: u1B,
                           stride: 6, padding: 3)
    // Reflection pad on the LEFT by 1 (KittenTTS quirk — NOT right).
    x = reflectionPadLeft(x, 1)
    x = x + noiseRes1
    let r2 = adainResBlockHiFiGAN(
        x, style: acousticStyle, weights: w,
        base: "decoder.generator.resblocks.2")
    let r3 = adainResBlockHiFiGAN(
        x, style: acousticStyle, weights: w,
        base: "decoder.generator.resblocks.3")
    x = (r2 + r3) / 2.0
    x = leakyReLU(x, slope: 0.1)
    let (cpW, cpB) = loadConv1dNCL(
        w, base: "decoder.generator.conv_post")
    // (1, 22, T).
    x = conv1dNCL(x, weight: cpW, bias: cpB, padding: 3)
    return istftHead(x, weights: w)
}

// The real noise path: SineGen generates pitch-synchronised harmonic
// excitation from F0, a learned linear layer mixes the 9 harmonics
// down to one channel, a learned forward STFT (Conv1d real+imag)
// extracts magnitude+phase, and two strided noise_convs produce
// features that feed into the noise_res AdaIN ResBlocks. The outputs
// are added into the HiFi-GAN upsampling stages.

private func computeNoiseContribs(f0Proj: MLXArray, nFrames: Int,
                                  acousticStyle: MLXArray,
                                  weights w: Weights)
        -> (MLXArray, MLXArray) {
    // ── 1. SineGen: F0 -> 9 harmonic sinusoids at audio rate ──
    let T_frames = nFrames * 2
    let hopSize = 300          // 24000 Hz / 80 Hz frame rate
    let T_audio = T_frames * hopSize
    let sampleRate: Float = 24000.0
    let sineAmp: Float = 0.1
    let noiseStd: Float = 0.003
    // Nearest-neighbour upsample F0 by 300x:
    // (1,1,T_frames) -> (1,1,T_audio)
    let f0Expand = f0Proj.reshaped([1, 1, T_frames, 1])
    let f0Bcast = MLX.broadcast(f0Expand,
                                to: [1, 1, T_frames, hopSize])
    let f0Audio = f0Bcast.reshaped([1, 1, T_audio])
    // Voiced mask (F0 > 0)
    // (1, 1, T_audio).
    let voiced = (f0Audio .> 0).asType(.float32)
    // Harmonic frequencies: f0 * [1,2,...,9] — (1, 9, T_frames).
    let harmonics = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8, 9])
                        .reshaped([1, 9, 1])
    // (1, 9, T_frames).
    let f0PerFrame = f0Proj.reshaped([1, 1, T_frames]) * harmonics
    // Per-frame phase accumulation + within-frame linear
    // extrapolation. The original formulation cumsum'd phase_inc at
    // audio rate over ~144 k samples, which accumulates fp32 rounding
    // error enough to phase-shift voice harmonics and manifest as
    // spurious tones in the CoreML export. Reducing the long-axis
    // cumsum to one-per-frame (T_frames ≈ 480 for a 6 s utterance)
    // eliminates that drift in both backends.
    // (1, 9, T_frames).
    let step = f0PerFrame
             * Float(Double(hopSize) / Double(sampleRate))
    // (1, 9, T_frames).
    let phaseStart = (step.cumsum(axis: -1) - step)
                   * Float(2.0 * .pi)
    let tInFrame = MLXArray(
        stride(from: Float(0), to: Float(hopSize), by: 1)
            .map { v in v })
        .reshaped([1, 1, 1, hopSize])
    let phaseWithin = f0PerFrame.expandedDimensions(axis: -1)
                          * tInFrame
                          / sampleRate * Float(2.0 * .pi)
    let phase = (phaseStart.expandedDimensions(axis: -1)
                 + phaseWithin)
                .reshaped([1, 9, T_audio])
    // Upstream KittenTTS randomizes phase_jitter per harmonic and
    // adds a tiny unvoiced gaussian dither (noise_std = 0.003). Both
    // are drawn from an RNG that differs between backends (MLX,
    // CoreML, torch), producing audibly different output each run
    // and diverging from the reference. Since both terms are
    // perceptually negligible in speech, zeroing them yields
    // deterministic output and lets this backend match the CoreML
    // backend bit-for-bit on the deterministic path.
    _ = noiseStd
    let phaseJitter = MLXArray.zeros([1, 9, 1], type: Float.self)
    // (1, 9, T_audio).
    let sines = MLX.sin(phase + phaseJitter) * sineAmp
    let uvNoise = MLXArray.zeros([1, 9, T_audio], type: Float.self)
    // Blend: sine where voiced, noise where unvoiced.
    // (1, 9, T_audio).
    let sinGen = sines * voiced + uvNoise * (1.0 - voiced)
    // ── 2. l_linear: 9 harmonics -> 1 channel + tanh ──
    // Nano key: onnx::MatMul_6116 (mini: onnx::MatMul_6388).
    // (9, 1).
    let lLinW = (w.f32("onnx::MatMul_6116")
                 ?? w.f32("onnx::MatMul_6388"))!
    // (1,).
    let lLinB = w.bias("decoder.generator.m_source.l_linear")!
    // (1, T, 1).
    let mixed = MLX.matmul(sinGen.transposed(0, 2, 1), lLinW)
              + lLinB
    // (1, 1, T_audio).
    let excitation = MLX.tanh(mixed.transposed(0, 2, 1))
    // ── 3. Forward STFT (learned Conv1d, not FFT) ──
    // (11, 1, 20).
    let stftR = w.f32(
        "kmodel.decoder.generator.stft.weight_forward_real")!
    let stftI = w.f32(
        "kmodel.decoder.generator.stft.weight_forward_imag")!
    let stftReal = conv1dNCL(excitation, weight: stftR, bias: nil,
                             stride: 5, padding: 10)
    let stftImag = conv1dNCL(excitation, weight: stftI, bias: nil,
                             stride: 5, padding: 10)
    let magSq = stftReal * stftReal + stftImag * stftImag
    // (1, 11, T_stft).
    let mag   = MLX.sqrt(magSq + Float(1e-9))
    // (1, 11, T_stft).
    let phi   = atan2(stftImag, stftReal)
    // (1, 22, T_stft).
    let stftOut = MLX.concatenated([mag, phi], axis: 1)
    // ── 4. noise_convs ──
    let (nc0W, nc0B) = loadConv1dNCL(
        w, base: "decoder.generator.noise_convs.0")
    let nc0 = conv1dNCL(stftOut, weight: nc0W, bias: nc0B,
                        stride: 6, padding: 3)
    let (nc1W, nc1B) = loadConv1dNCL(
        w, base: "decoder.generator.noise_convs.1")
    let nc1 = conv1dNCL(stftOut, weight: nc1W, bias: nc1B)
    // ── 5. noise_res (AdaIN ResBlocks with Snake activation) ──
    let nr0 = adainResBlockHiFiGAN(
        nc0, style: acousticStyle, weights: w,
        base: "decoder.generator.noise_res.0")
    let nr1 = adainResBlockHiFiGAN(
        nc1, style: acousticStyle, weights: w,
        base: "decoder.generator.noise_res.1")
    return (nr0, nr1)
}

/// Tokens -> waveform. `style256` first half is acoustic, second half
/// prosodic.
func kittenForward(weights w: Weights, inputIds: MLXArray,
                   style256: MLXArray,
                   speed: Float = 1.0) -> MLXArray {
    let B = inputIds.dim(0); let L = inputIds.dim(1)
    // (1, 128).
    let acousticStyle = style256[0..., 0..<128].asType(.float32)
    // (1, 128).
    let prosodicStyle = style256[0..., 128..<256].asType(.float32)
    // BERT + bert_encoder (768 -> 128).
    // (1, L, 768).
    let bertOut = bertForward(inputIds, weights: w)
    let beW = w.f32("onnx::MatMul_5818")!  // (768, 128)
    let beB = w.bias("bert_encoder")!
    // (1, L, 128).
    let prosodyIn = MLX.matmul(bertOut, beW) + beB
    // predictor.text_encoder (4-layer LSTM/AdaLN stack).
    // (1, L, 128).
    let prosody = predictorTextEncoder(prosodyIn,
                                       prosodyStyle: prosodicStyle,
                                       weights: w)
    let sBcast = MLX.broadcast(
        prosodicStyle.reshaped([B, 1, -1]),
        to: [B, L, 128])
    let prosody256 = MLX.concatenated([prosody, sBcast], axis: -1)
    // (1, 256, L).
    let prosodyNCL = prosody256.transposed(0, 2, 1)
    // predictor.lstm (shared duration/prosody LSTM).
    // (L, 1, 256).
    let lstmIn = prosodyNCL.transposed(2, 0, 1).asType(.float32)
    let (dW, dR, dB) = loadDQLSTM(w,
                                  W: "onnx::LSTM_5971",
                                  R: "onnx::LSTM_5972",
                                  B: "onnx::LSTM_5970")
    let dy = onnxLSTMForward(lstmIn, W: dW, R: dR, B: dB,
                             bidirectional: true)
    let lstmOut = dy.transposed(2, 0, 1, 3).reshaped([1, L, 128])
    // duration_proj: (128 -> 50) per frame, sigmoid, sum ->
    // duration-per-phoneme.
    let dpW = w.f32("onnx::MatMul_5973")!  // (128, 50)
    let dpB = w.bias("predictor.duration_proj.linear_layer")!
    // (1, L, 50).
    let durLogits = MLX.matmul(lstmOut, dpW) + dpB
    let durSig = MLX.sigmoid(durLogits)
    let durSum = durSig.sum(axis: -1)  // (1, L)
    let durScaled = durSum[0] / MLXArray(speed)
    let durs = MLX.maximum(MLXArray([1]),
                           MLX.round(durScaled).asType(.int32))
    let durArr: [Int32] = durs.asArray(Int32.self)
    let nFrames = Int(durArr.reduce(0, +))
    // Alignment matrix + length regulation.
    let align = buildAlignment(durations: durArr,
                               nPhonemes: L, nFrames: nFrames)
    // (1, 256, nFrames).
    let prosodyLRNCL = MLX.matmul(prosodyNCL, align)
    // shared LSTM -> (1, 128, nFrames).
    let sharedIn = prosodyLRNCL.transposed(2, 0, 1).asType(.float32)
    let (sW, sR, sB) = loadDQLSTM(w,
                                  W: "onnx::LSTM_6020",
                                  R: "onnx::LSTM_6021",
                                  B: "onnx::LSTM_6019")
    let sy = onnxLSTMForward(sharedIn, W: sW, R: sR, B: sB,
                             bidirectional: true)
    let fnLSTMNLC = sy.transposed(2, 0, 1, 3)
                       .reshaped([1, nFrames, 128])
    // (1, 128, nFrames).
    let fnInNCL = fnLSTMNLC.transposed(0, 2, 1)
    // F0 stack.
    var f0 = adainResBlock1D(fnInNCL, style: prosodicStyle,
                             weights: w, base: "predictor.F0.0",
                             upsample: false, divide: true,
                             shortcutInput: fnInNCL)
    f0 = adainResBlock1D(f0, style: prosodicStyle,
                         weights: w, base: "predictor.F0.1",
                         upsample: true, divide: true,
                         shortcutInput: f0)
    f0 = adainResBlock1D(f0, style: prosodicStyle,
                         weights: w, base: "predictor.F0.2",
                         upsample: false, divide: true,
                         shortcutInput: f0)
    let (f0pW, f0pB) = loadConv1dNCL(w, base: "predictor.F0_proj")
    // (1, 1, nFrames*2).
    let f0Proj = conv1dNCL(f0, weight: f0pW, bias: f0pB)
    // N stack.
    var n = adainResBlock1D(fnInNCL, style: prosodicStyle,
                            weights: w, base: "predictor.N.0",
                            upsample: false, divide: true,
                            shortcutInput: fnInNCL)
    n = adainResBlock1D(n, style: prosodicStyle,
                        weights: w, base: "predictor.N.1",
                        upsample: true, divide: true,
                        shortcutInput: n)
    n = adainResBlock1D(n, style: prosodicStyle,
                        weights: w, base: "predictor.N.2",
                        upsample: false, divide: true,
                        shortcutInput: n)
    let (npW, npB) = loadConv1dNCL(w, base: "predictor.N_proj")
    let nProj = conv1dNCL(n, weight: npW, bias: npB)
    // Acoustic text encoder + length regulation.
    // (1, 128, L).
    let textFeaturesNCL = acousticTextEncoder(inputIds, weights: w)
    // (1, 128, nFrames).
    let textLRNCL = MLX.matmul(textFeaturesNCL, align)
    // Decoder.
    let decOut = decoderPipeline(textFeaturesNCL: textLRNCL,
                                 f0NCL: f0Proj, nNCL: nProj,
                                 acousticStyle: acousticStyle,
                                 weights: w)
    // Noise path: F0 -> SineGen -> STFT -> noise_convs -> noise_res.
    let (nr0, nr1) = computeNoiseContribs(
        f0Proj: f0Proj, nFrames: nFrames,
        acousticStyle: acousticStyle, weights: w)
    return generatorPipeline(decOut,
                             acousticStyle: acousticStyle,
                             noiseRes0: nr0, noiseRes1: nr1,
                             weights: w)
}

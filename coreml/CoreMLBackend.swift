import Foundation
@preconcurrency import CoreML
import os
// MLX is used only for loading voices.safetensors; the forward pass is
// all CoreML.
import MLX

/// CoreML-backed alternative to `KittenTTS`. Keeps the same public API
/// so `KittenApp` can A/B the two backends against each other.
///
/// **Multifunction** `.mlpackage` layout (as of Plan B in
/// INVESTIGATE.md): one file per (stage × variant); each file contains
/// a set of pre-compiled functions (`L_16`, `L_32`, …, `N_1024`) that
/// share weights on disk and in memory. Swift picks the right function
/// via `MLModelConfiguration.functionName`.
///
/// Dimensions the backend exposes for A/B:
/// - `Variant`: precision. `fp32` / `fp16` / `int8w`. (`int8wa`
///   blocked by a coremltools bug on int32 inputs — see
///   INVESTIGATE.md.)
/// - `Compute`: `.ane` / `.gpu` / `.cpu` — pin which device pairs with
///   CPU for the per-op scheduler. Maps to `MLComputeUnits` 1:1.
public final nonisolated class KittenTTSCoreML: @unchecked Sendable {

    public nonisolated struct Config: Sendable {
        public var speed:   Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed   = speed
            self.voiceID = voiceID
        }
    }

    public enum Variant: String, Sendable, CaseIterable {
        case fp32
        case fp16
        case int8w
        case int8wa
    }

    /// Variants whose `.mlmodelc` / `.mlpackage` are actually present
    /// in the running app bundle. Lets the UI/bench filter the picker
    /// so a fresh install can't land on a variant that has no shipped
    /// weights (the default repo bundles only `int8wa`; the larger
    /// variants are reproducible via `scripts/convert_to_coreml.py`).
    public static let availableVariants: [Variant] =
        Variant.allCases.filter { v in
            let textName = "kitten_text_\(v.rawValue)"
            let genName  = "kitten_generator_\(v.rawValue)"
            return KittenTTSCoreML.resourceURL(name: textName) != nil
                && KittenTTSCoreML.resourceURL(name: genName) != nil
        }

    /// CoreML compute pinning. We expose the three concrete CPU-paired
    /// targets (ANE, GPU, CPU) explicitly rather than the "Auto" =
    /// MLComputeUnits.all option — for R&D the explicit choice is more
    /// useful (you know exactly what you're measuring) and the per-op
    /// scheduler under .all isn't doing anything for our model that
    /// you can't get with one of the pinned modes. Note that ANE and
    /// GPU modes still allow a CPU fallback for ops their device can't
    /// run; that's MLComputeUnits semantics, not a choice we make.
    public enum Compute: String, Sendable, CaseIterable {
        case ane = "ANE"   // MLComputeUnits.cpuAndNeuralEngine
        case gpu = "GPU"   // MLComputeUnits.cpuAndGPU
        case cpu = "CPU"   // MLComputeUnits.cpuOnly
    }

    public typealias SpeakCallback =
        (UnsafePointer<Int16>, Int) -> Void

    /// Metrics emitted by the backend for UI / CLI display.
    public struct ChunkMetrics: Sendable {
        public let phonemes:         Int
        public let bucketL:          Int
        public let bucketN:          Int
        public let variant:          Variant
        public let compute:          Compute
        public let textStageMs:      Double
        public let generatorStageMs: Double
        public let samples:          Int
    }

    public var onBucketLoaded:
        ((_ name: String, _ elapsedMs: Double) -> Void)?
    public var onChunkMetrics: ((ChunkMetrics) -> Void)?

    // Shipped bucket (function) sizes. Must match the function names
    // inside the multifunction .mlpackages. Keep sorted ascending.
    static let textBuckets:      [Int] = [16, 32, 64, 128, 400]
    static let generatorBuckets: [Int] = [128, 256, 512, 1024]
    // 24000 Hz / 80 Hz frame rate × 2 (F0 upsample).
    static let audioPerFrame:    Int   = 300

    /// Cache key: different variant / compute / bucket combos are
    /// distinct MLModel instances because each binds its own
    /// MLModelConfiguration.
    private struct Key: Hashable {
        let stage:   String        // "text" | "generator"
        let variant: Variant
        let compute: Compute
        let bucket:  Int
    }

    private var models: [Key: MLModel] = [:]
    // Flattened 400 * 256 floats per voice.
    private var voiceEmbeds: [String: [Float]] = [:]
    private let loadLock = OSAllocatedUnfairLock()

    public static var voiceAliases: [String: String] {
        KittenTTS.voiceAliases
    }
    public static var voiceDisplayOrder: [String] {
        KittenTTS.voiceDisplayOrder
    }

    public init() {
        // MLX-shared voices load needs the bundled mlx.metallib next
        // to the executable in bare-binary CLI layouts. Inside a .app,
        // MLX finds its embedded default.metallib and this is a no-op.
        KittenTTS.installMetalLib()
    }

    public func preload() async throws {
        if voiceEmbeds.isEmpty { try loadVoices() }
    }

    /// Warm up every text-L and generator-N bucket for the given
    /// variant/compute, paying the per-device ANE compile cost once
    /// per install rather than on the first utterance whose length
    /// hits an un-warmed bucket. Emits an `onBucketLoaded` metric per
    /// bucket so the UI can show progress. Safe to call repeatedly —
    /// already-loaded (variant, compute, bucket) keys hit the model
    /// cache.
    public func warmUpAll(variant: Variant, compute: Compute) async {
        try? await preload()
        for L in Self.textBuckets {
            _ = try? await model(stage:   "text",
                                 variant: variant,
                                 compute: compute,
                                 bucket:  L)
        }
        for N in Self.generatorBuckets {
            _ = try? await model(stage:   "generator",
                                 variant: variant,
                                 compute: compute,
                                 bucket:  N)
        }
    }

    /// Drop all loaded MLModel instances and the voice prompt table
    /// (~3 MB of flattened style floats). Note: even after this RSS
    /// usually doesn't drop. CoreML retains process-wide caches in
    /// the e5rt / ANE compile pipeline (`com.apple.e5rt.e5bundlecache`
    /// in `~/Library/Caches/<bundle>/`) plus Metal pipeline state
    /// objects, and libc holds freed pages on its own free lists for
    /// reuse. Activity Monitor reports those as resident; they're
    /// reclaimed only under real memory pressure, not on unload.
    public func unload() {
        loadLock.withLock {
            models.removeAll()
            voiceEmbeds.removeAll()
        }
    }

    public func speak(
        text:     String,
        config:   Config = Config(),
        variant:  Variant = .int8w,
        compute:  Compute = .ane,
        callback: SpeakCallback? = nil
    ) async throws -> [Float] {
        try await preload()
        let voiceID = KittenTTSCoreML.voiceAliases[config.voiceID]
                          ?? config.voiceID
        let voiceRows: [Float]
        if let v = voiceEmbeds[voiceID]
                ?? voiceEmbeds["expr-voice-5-m"] {
            voiceRows = v
        } else {
            throw NSError(
                domain: "KittenTTSCoreML", code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "voice '\(voiceID)' not found"])
        }
        let effectiveSpeed = config.speed
                           * (KittenTTS.speedPriors[voiceID] ?? 1.0)
        let normalised = TextPreprocessor.process(text)
        // Paragraph-aware chunking with em-dash breaks. See
        // CPUBackend / TextChunker.phonemizedChunks for rationale.
        let chunks = TextChunker.phonemizedChunks(normalised)
        var allAudio: [Float] = []
        for (_, c) in chunks.enumerated() {
            // Stop / backend-switch cancels this Task — drop out
            // of the loop before starting the next chunk.
            if Task.isCancelled { break }
            let chunk = c.text
            let phonemes = c.phonemes
            let refId = min(chunk.count, 399)
            let style = Array(
                voiceRows[(refId * 256)..<((refId + 1) * 256)])
            let audio = try await speakOneChunk(
                phonemes: phonemes, style: style,
                speed: effectiveSpeed,
                variant: variant, compute: compute)
            if Task.isCancelled { break }
            // Silences scale inversely with effectiveSpeed (see
            // CPUBackend comment for rationale).
            let silenceN = Int(
                Double(c.priorSilenceMs) * 24.0
                / Double(effectiveSpeed))
            let emit: [Float] = silenceN <= 0
                ? audio
                : [Float](repeating: 0, count: silenceN) + audio
            if let cb = callback {
                let int16 = emit.map { f -> Int16 in
                    // NaN/Inf can sneak through fp16 CoreML on CPU;
                    // guard before Int(.rounded()) which traps on
                    // non-finite floats.
                    let safe: Float = f.isFinite ? f : 0
                    return Int16(clamping:
                        Int((safe * 32767.0).rounded()))
                }
                int16.withUnsafeBufferPointer { buf in
                    if let base = buf.baseAddress {
                        cb(base, buf.count)
                    }
                }
            }
            allAudio.append(contentsOf: emit)
        }
        return allAudio
    }

    private func speakOneChunk(
        phonemes: [Int], style: [Float], speed: Float,
        variant: Variant, compute: Compute
    ) async throws -> [Float] {
        let realL = phonemes.count
        let L = Self.textBuckets.first(where: { b in b >= realL })
                    ?? Self.textBuckets.last!
        let clippedL = min(realL, L)
        let textModel = try await model(stage:   "text",
                                        variant: variant,
                                        compute: compute,
                                        bucket:  L)
        let tTextStart = Date()
        let idsArr = try MLMultiArray(
            shape: [1, NSNumber(value: L)], dataType: .int32)
        let idsPtr = idsArr.dataPointer.bindMemory(
            to: Int32.self, capacity: L)
        let maskArr = try MLMultiArray(
            shape: [1, NSNumber(value: L)], dataType: .float32)
        let maskPtr = maskArr.dataPointer.bindMemory(
            to: Float.self, capacity: L)
        for i in 0..<L {
            idsPtr[i] = i < clippedL ? Int32(phonemes[i]) : 0
            maskPtr[i] = i < clippedL ? 1.0 : 0.0
        }
        let styleArr = try MLMultiArray(
            shape: [1, 256], dataType: .float32)
        let stylePtr = styleArr.dataPointer.bindMemory(
            to: Float.self, capacity: 256)
        for i in 0..<256 { stylePtr[i] = style[i] }
        let textIn = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: idsArr),
            "style":          MLFeatureValue(multiArray: styleArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let textOut = try await textModel.prediction(from: textIn)
        var prosodyOpt: MLMultiArray?
        var textOpt:    MLMultiArray?
        var durOpt:     MLMultiArray?
        for name in textOut.featureNames {
            if let val = textOut.featureValue(for: name)?
                                .multiArrayValue {
                let shape = val.shape.map { n in n.intValue }
                switch shape {
                case [1, 256, L]: prosodyOpt = val
                case [1, 128, L]: textOpt    = val
                case [1, L, 50]:  durOpt     = val
                default:          break
                }
            }
        }
        let prosodyNCL:   MLMultiArray
        let textFeatures: MLMultiArray
        let durSig:       MLMultiArray
        if let p = prosodyOpt, let t = textOpt, let d = durOpt {
            prosodyNCL   = p
            textFeatures = t
            durSig       = d
        } else {
            throw NSError(
                domain: "KittenTTSCoreML", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "unexpected TextStage outputs"])
        }
        let durSigFlat  = try Self.copyToFloat32(durSig)
        let prosodyFlat = try Self.copyToFloat32(prosodyNCL)
        let textFlat    = try Self.copyToFloat32(textFeatures)
        let textStageMs = Date().timeIntervalSince(tTextStart)
                              * 1000.0
        var durations: [Int] = []
        durations.reserveCapacity(clippedL)
        var totalFrames = 0
        for i in 0..<clippedL {
            var sum: Float = 0
            let base = i * 50
            for j in 0..<50 { sum += durSigFlat[base + j] }
            let raw = Int((sum / speed).rounded())
            let d = max(1, raw)
            durations.append(d)
            totalFrames += d
        }
        let nBucket = Self.generatorBuckets
                          .first(where: { b in b >= totalFrames })
                          ?? Self.generatorBuckets.last!
        if totalFrames > nBucket {
            var overflow = totalFrames - nBucket
            var i = durations.count - 1
            while overflow > 0 && i >= 0 {
                let shrink = min(durations[i] - 1, overflow)
                durations[i] -= shrink
                overflow -= shrink
                i -= 1
            }
            totalFrames = durations.reduce(0, +)
        }
        let generatorModel = try await model(stage:   "generator",
                                             variant: variant,
                                             compute: compute,
                                             bucket:  nBucket)
        let tGenStart = Date()
        let prosodyLR = try MLMultiArray(
            shape: [1, 256, NSNumber(value: nBucket)],
            dataType: .float32)
        let textLR = try MLMultiArray(
            shape: [1, 128, NSNumber(value: nBucket)],
            dataType: .float32)
        let prosodyLRPtr = prosodyLR.dataPointer.bindMemory(
            to: Float.self, capacity: 256 * nBucket)
        let textLRPtr = textLR.dataPointer.bindMemory(
            to: Float.self, capacity: 128 * nBucket)
        for i in 0..<(256 * nBucket) { prosodyLRPtr[i] = 0 }
        for i in 0..<(128 * nBucket) { textLRPtr[i] = 0 }
        var frame = 0
        var i = 0
        while frame < nBucket && i < durations.count {
            let d = durations[i]
            var k = 0
            while frame < nBucket && k < d {
                for c in 0..<256 {
                    prosodyLRPtr[c * nBucket + frame] =
                        prosodyFlat[c * L + i]
                }
                for c in 0..<128 {
                    textLRPtr[c * nBucket + frame] =
                        textFlat[c * L + i]
                }
                frame += 1
                k += 1
            }
            i += 1
        }
        let genIn = try MLDictionaryFeatureProvider(dictionary: [
            "prosody_lr": MLFeatureValue(multiArray: prosodyLR),
            "text_lr":    MLFeatureValue(multiArray: textLR),
            "style":      MLFeatureValue(multiArray: styleArr),
        ])
        let genOut = try await generatorModel.prediction(from: genIn)
        let wav: MLMultiArray
        if let w = genOut.featureNames.compactMap({ name in
                       genOut.featureValue(for: name)?
                             .multiArrayValue
                   }).first {
            wav = w
        } else {
            throw NSError(
                domain: "KittenTTSCoreML", code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "generator produced no output"])
        }
        let wavFlat = try Self.copyToFloat32(wav)
        // The generator's decoder has non-causal convolutions: its
        // output for the last few real frames is perturbed by the
        // zero-padded frames immediately after. Drop those frames
        // (tailDropFrames) before the fade-out so we discard the
        // artifact region entirely. `max(1, raw)` already minimally
        // pads trailing punctuation with silence-duration frames, so
        // clipping a couple of frames off the end costs no speech
        // content.
        let tailDropFrames = 3
        let trimFrames = max(0, totalFrames - tailDropFrames)
        let realSamples = trimFrames * Self.audioPerFrame * 2
        let take = min(realSamples, wavFlat.count)
        var output = Array(wavFlat.prefix(take))
        Self.applyFadeIn(&output,  fadeSamples: 72)    // 3 ms
        Self.applyFadeOut(&output, fadeSamples: 960)   // 40 ms
        let generatorStageMs = Date().timeIntervalSince(tGenStart)
                                   * 1000.0
        onChunkMetrics?(ChunkMetrics(
            phonemes:         realL,
            bucketL:          L,
            bucketN:          nBucket,
            variant:          variant,
            compute:          compute,
            textStageMs:      textStageMs,
            generatorStageMs: generatorStageMs,
            samples:          output.count))
        return output
    }

    /// Cosine fade from 0 -> 1 over the first `fadeSamples` of
    /// `samples`.
    private static func applyFadeIn(_ samples: inout [Float],
                                    fadeSamples: Int) {
        let n = min(fadeSamples, samples.count)
        if n > 0 {
            for i in 0..<n {
                let t = Float(i) / Float(n - 1 > 0 ? n - 1 : 1)
                // 0 at i=0, 1 at i=n-1.
                let gain = 0.5 - 0.5 * cos(.pi * t)
                samples[i] *= gain
            }
        }
    }

    /// Cosine fade from 1 -> 0 over the last `fadeSamples` of
    /// `samples`.
    private static func applyFadeOut(_ samples: inout [Float],
                                     fadeSamples: Int) {
        let n = min(fadeSamples, samples.count)
        if n > 0 {
            let start = samples.count - n
            for i in 0..<n {
                let t = Float(i) / Float(n - 1 > 0 ? n - 1 : 1)
                // 1 at i=0, 0 at i=n-1.
                let gain = 0.5 + 0.5 * cos(.pi * t)
                samples[start + i] *= gain
            }
        }
    }

    /// Copy an MLMultiArray into a packed row-major Float32 array,
    /// regardless of its native dtype (fp16 vs fp32) or padded
    /// strides.
    private static func copyToFloat32(_ a: MLMultiArray) throws
            -> [Float] {
        let shape = a.shape.map { n in n.intValue }
        let strides = a.strides.map { n in n.intValue }
        let packed = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: packed)
        var indices = Array(repeating: 0, count: shape.count)
        let rank = shape.count
        func advance() {
            // Carry-propagating increment of the multi-index. After
            // running this `packed` times, `indices` has visited
            // every coordinate in row-major order.
            var k = rank - 1
            indices[k] += 1
            while indices[k] >= shape[k] && k > 0 {
                indices[k] = 0
                k -= 1
                indices[k] += 1
            }
            if indices[k] >= shape[k] { indices[k] = 0 }
        }
        func walkFp16() {
            let src = a.dataPointer
                          .assumingMemoryBound(to: UInt16.self)
            for dst in 0..<packed {
                var srcOff = 0
                for k in 0..<rank {
                    srcOff += indices[k] * strides[k]
                }
                out[dst] = Float(Float16(bitPattern: src[srcOff]))
                advance()
            }
        }
        func walkFp32() {
            let src = a.dataPointer
                          .assumingMemoryBound(to: Float.self)
            for dst in 0..<packed {
                var srcOff = 0
                for k in 0..<rank {
                    srcOff += indices[k] * strides[k]
                }
                out[dst] = src[srcOff]
                advance()
            }
        }
        switch a.dataType {
        case .float16: walkFp16()
        case .float32: walkFp32()
        default:
            throw NSError(
                domain: "KittenTTSCoreML", code: 7,
                userInfo: [NSLocalizedDescriptionKey:
                    "unsupported output dtype " +
                    "\(a.dataType.rawValue)"])
        }
        return out
    }

    private func model(stage: String, variant: Variant,
                       compute: Compute, bucket: Int) async throws
            -> MLModel {
        let key = Key(stage:   stage,
                      variant: variant,
                      compute: compute,
                      bucket:  bucket)
        let cached = loadLock.withLock { models[key] }
        let result: MLModel
        if let c = cached {
            result = c
        } else {
            // e.g. kitten_text_fp16
            let packageName = "kitten_\(stage)_\(variant.rawValue)"
            let axis = stage == "text" ? "L" : "N"
            let functionName = "\(axis)_\(bucket)"
            let cfg = MLModelConfiguration()
            cfg.computeUnits = {
                switch compute {
                case .ane: return .cpuAndNeuralEngine
                case .gpu: return .cpuAndGPU
                case .cpu: return .cpuOnly
                }
            }()
            cfg.functionName = functionName
            let found: (url: URL, isCompiled: Bool)
            if let f = Self.resourceURL(name: packageName) {
                found = f
            } else {
                throw NSError(
                    domain: "KittenTTSCoreML", code: 5,
                    userInfo: [NSLocalizedDescriptionKey:
                        "bundle has no \(packageName).mlmodelc " +
                        "or \(packageName).mlpackage"])
            }
            let tLoadStart = Date()
            let loaded: MLModel
            if found.isCompiled {
                loaded = try MLModel(contentsOf: found.url,
                                     configuration: cfg)
            } else {
                let compiled = try await Self.compiledModelURL(
                    packageURL: found.url, name: packageName)
                loaded = try MLModel(contentsOf: compiled,
                                     configuration: cfg)
            }
            let loadMs = Date().timeIntervalSince(tLoadStart)
                             * 1000.0
            let tWarmStart = Date()
            try await warmup(model: loaded,
                             stage: stage, bucket: bucket)
            let warmMs = Date().timeIntervalSince(tWarmStart)
                             * 1000.0
            let label = "\(packageName) [\(functionName)/" +
                        "\(compute.rawValue)]  load " +
                        "\(Int(loadMs))ms + warmup"
            onBucketLoaded?(label, warmMs)
            loadLock.withLock { models[key] = loaded }
            result = loaded
        }
        return result
    }

    private func warmup(model: MLModel, stage: String,
                        bucket: Int) async throws {
        let input: MLDictionaryFeatureProvider
        if stage == "text" {
            let L = bucket
            let ids = try MLMultiArray(
                shape: [1, NSNumber(value: L)], dataType: .int32)
            let style = try MLMultiArray(
                shape: [1, 256], dataType: .float32)
            let mask = try MLMultiArray(
                shape: [1, NSNumber(value: L)], dataType: .float32)
            memset(ids.dataPointer,   0,
                   L * MemoryLayout<Int32>.size)
            memset(style.dataPointer, 0,
                   256 * MemoryLayout<Float>.size)
            memset(mask.dataPointer,  0,
                   L * MemoryLayout<Float>.size)
            input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids":      MLFeatureValue(multiArray: ids),
                "style":          MLFeatureValue(multiArray: style),
                "attention_mask": MLFeatureValue(multiArray: mask),
            ])
        } else {
            let N = bucket
            let prosody = try MLMultiArray(
                shape: [1, 256, NSNumber(value: N)],
                dataType: .float32)
            let text = try MLMultiArray(
                shape: [1, 128, NSNumber(value: N)],
                dataType: .float32)
            let style = try MLMultiArray(
                shape: [1, 256], dataType: .float32)
            memset(prosody.dataPointer, 0,
                   256 * N * MemoryLayout<Float>.size)
            memset(text.dataPointer,    0,
                   128 * N * MemoryLayout<Float>.size)
            memset(style.dataPointer,   0,
                   256 * MemoryLayout<Float>.size)
            input = try MLDictionaryFeatureProvider(dictionary: [
                "prosody_lr": MLFeatureValue(multiArray: prosody),
                "text_lr":    MLFeatureValue(multiArray: text),
                "style":      MLFeatureValue(multiArray: style),
            ])
        }
        _ = try await model.prediction(from: input)
    }

    private static func resourceURL(name: String)
            -> (url: URL, isCompiled: Bool)? {
        var found: (url: URL, isCompiled: Bool)? = nil
        let exts = ["mlmodelc", "mlpackage"]
        var i = 0
        while found == nil && i < exts.count {
            let ext = exts[i]
            if let u = Bundle.main.url(forResource: name,
                                       withExtension: ext) {
                found = (u, ext == "mlmodelc")
            }
            i += 1
        }
        if found == nil, let base = Bundle.main.resourceURL {
            var j = 0
            while found == nil && j < exts.count {
                let ext = exts[j]
                let u = base.appendingPathComponent(
                    "coreml/\(name).\(ext)")
                if FileManager.default.fileExists(atPath: u.path) {
                    found = (u, ext == "mlmodelc")
                }
                j += 1
            }
        }
        return found
    }

    /// Return a compiled `.mlmodelc` URL, reusing a cached copy if
    /// its mtime matches the bundled `.mlpackage`'s Manifest. First
    /// launch pays the compile cost once per package.
    private static func compiledModelURL(packageURL: URL,
                                         name: String) async throws
            -> URL {
        let fm = FileManager.default
        let cacheRoot = try fm.url(for: .cachesDirectory,
                                    in: .userDomainMask,
                                    appropriateFor: nil,
                                    create: true)
            .appendingPathComponent("KittenTTS", isDirectory: true)
        try? fm.createDirectory(at: cacheRoot,
                                withIntermediateDirectories: true)
        let manifestURL = packageURL.appendingPathComponent(
            "Manifest.json")
        let pkgMtime = (try? fm.attributesOfItem(
            atPath: manifestURL.path)[.modificationDate] as? Date)
            ?? Date.distantPast
        let stamp = Int(pkgMtime.timeIntervalSince1970)
        let compiledURL = cacheRoot.appendingPathComponent(
            "\(name)_\(stamp).mlmodelc", isDirectory: true)
        if !fm.fileExists(atPath: compiledURL.path) {
            if let entries = try? fm.contentsOfDirectory(
                    at: cacheRoot,
                    includingPropertiesForKeys: nil) {
                for url in entries
                        where url.lastPathComponent
                                  .hasPrefix("\(name)_") {
                    try? fm.removeItem(at: url)
                }
            }
            let freshlyCompiled = try await MLModel.compileModel(
                at: packageURL)
            try? fm.removeItem(at: compiledURL)
            try fm.moveItem(at: freshlyCompiled, to: compiledURL)
        }
        return compiledURL
    }

    private func loadVoices() throws {
        let modelDir: URL
        if let d = ModelLoader.bundledModelDir() {
            modelDir = d
        } else {
            throw NSError(
                domain: "KittenTTSCoreML", code: 6,
                userInfo: [NSLocalizedDescriptionKey:
                    "no bundled model dir"])
        }
        let url = modelDir.appendingPathComponent(
            "voices.safetensors")
        let raw = try loadArrays(url: url)
        for (name, arr) in raw {
            let flat = arr.asType(.float32).asArray(Float.self)
            voiceEmbeds[name] = flat
        }
    }
}

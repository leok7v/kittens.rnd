// v1 is macOS-only — the linked llama.cpp static libs in
// vendors/llama.cpp/build-cpu/ are built for macOS. iOS/xrOS support
// would require building those libs for each platform via xcframework.
#if os(macOS) || os(iOS)

import Foundation
import os
// MLX is shared with the other backends for voices.safetensors loading.
import MLX

/// ggml/llama.cpp-backed alternative to `KittenTTS` and
/// `KittenTTSCoreML`. Same public `speak` API so KittenApp can A/B all
/// three backends.
///
/// Implementation lives in `KittensGGML/kittens-tts.c` (see
/// KittensGGML.h for the C API). The bridging header exposes the C
/// symbols (`kt_create`, `kt_synthesize`, …) directly to Swift.
///
/// v1 caveats:
/// - CPU only. `Compute.metal` requires a custom Metal shader for
///   atan2 in the noise path; not yet wired up.
/// - One graph rebuilt per `kt_synthesize` call. Average speech-rate
///   latency on M-series is ~0.1× realtime per chunk after the first
///   call.
/// - Phonemizer is the shared CEPhonemizer C++ engine — same as the
///   other backends, so input phonemes match.
public final nonisolated class KittenTTSLlamaCpp: @unchecked Sendable {

    public nonisolated struct Config: Sendable {
        public var speed:   Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed   = speed
            self.voiceID = voiceID
        }
    }

    /// Single backend variant for v1 (CPU only). Kept as an enum so
    /// the UI can present a consistent control across backends.
    public enum Compute: String, Sendable, CaseIterable {
        case cpu = "CPU"
    }

    public typealias SpeakCallback =
        (UnsafePointer<Int16>, Int) -> Void

    public struct ChunkMetrics: Sendable {
        public let phonemes: Int
        public let frames:   Int
        public let compute:  Compute
        public let totalMs:  Double
        public let samples:  Int
    }

    public var onChunkMetrics: ((ChunkMetrics) -> Void)?

    // 24000 Hz / 80 Hz × 2 (matches CoreML backend).
    static let audioPerFrame: Int = 600

    private var ctx: OpaquePointer?                  // kt_ctx *
    private var voiceEmbeds: [String: [Float]] = [:] // flat 400*256
    private let loadLock = OSAllocatedUnfairLock()
    /// Serialises kt_synthesize calls. The C API is single-threaded
    /// per ctx (see KittensGGML.h). The depth-1 prefetch in `speak`
    /// spawns chunk N+1's task while chunk N is still inferring —
    /// without this lock the two would race on shared ggml backend
    /// buffers (manifests as EXC_BAD_ACCESS / objc_msgSend corruption
    /// on iOS).
    private let inferLock = NSLock()

    public static var voiceAliases: [String: String] {
        KittenTTS.voiceAliases
    }
    public static var voiceDisplayOrder: [String] {
        KittenTTS.voiceDisplayOrder
    }

    public init() {
        // Defer GGUF/voices load to preload() — keeps init() cheap.
    }

    deinit {
        if let c = ctx { kt_destroy(c) }
    }

    public func preload() async throws {
        try loadLock.withLock {
            if ctx == nil {
                let modelDir: URL
                if let d = ModelLoader.bundledModelDir() {
                    modelDir = d
                } else {
                    throw NSError(
                        domain: "KittenTTSLlamaCpp",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey:
                            "no bundled model dir"])
                }
                let ggufURL = modelDir.appendingPathComponent(
                    "kitten_full.gguf")
                if !FileManager.default.fileExists(
                        atPath: ggufURL.path) {
                    throw NSError(
                        domain: "KittenTTSLlamaCpp",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "kitten_full.gguf not found at " +
                            "\(ggufURL.path)"])
                }
                let cgguf    = ggufURL.path.cString(using: .utf8)!
                let cbackend = "cpu".cString(using: .utf8)!
                // kt_create returns OpaquePointer? in Swift (kt_ctx
                // is a forward-declared C struct); assign directly.
                ctx = kt_create(cgguf, cbackend)
                if ctx == nil {
                    throw NSError(
                        domain: "KittenTTSLlamaCpp",
                        code: 3,
                        userInfo: [NSLocalizedDescriptionKey:
                            "kt_create failed"])
                }
            }
            if voiceEmbeds.isEmpty {
                try loadVoices()
            }
        }
    }

    public func unload() {
        loadLock.withLock {
            if let c = ctx { kt_destroy(c); ctx = nil }
            voiceEmbeds.removeAll()
        }
    }

    public func speak(
        text:     String,
        config:   Config = Config(),
        compute:  Compute = .cpu,
        callback: SpeakCallback? = nil
    ) async throws -> [Float] {
        try await preload()
        let voiceID = KittenTTSLlamaCpp.voiceAliases[config.voiceID]
                          ?? config.voiceID
        let voiceRows: [Float]
        if let v = voiceEmbeds[voiceID]
                ?? voiceEmbeds["expr-voice-5-m"] {
            voiceRows = v
        } else {
            throw NSError(
                domain: "KittenTTSLlamaCpp",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "voice '\(voiceID)' not found"])
        }
        let effectiveSpeed = config.speed
                           * (KittenTTS.speedPriors[voiceID] ?? 1.0)
        let normalised = TextPreprocessor.process(text)
        // GGML allocates buffers dynamically — no L=400 bucket like
        // CoreML. Pass `.max` so the chunker only breaks on `.!?` and
        // never on commas; whole sentences stay together regardless
        // of length. Eliminates the mid-sentence comma-split pause.
        let chunks = TextChunker.chunk(normalised, maxLen: .max)
        var allAudio: [Float] = []
        if !chunks.isEmpty {
            // 120 ms inter-sentence silence — only between true
            // sentence breaks (prev chunk ends in `.!?`). The chunker
            // no longer splits on `;:`, so sub-clauses stay inside one
            // chunk and the listener doesn't hear an unexpected pause
            // mid-thought.
            let gap = [Float](repeating: 0, count: Int(0.12 * 24000))
            func gapAfter(_ prev: String) -> [Float] {
                var g: [Float] = []
                if let last = prev.last,
                   last == "." || last == "!" || last == "?" {
                    g = gap
                }
                return g
            }
            // Depth-1 prefetch:
            //
            //   sequential: [phonemize_N][synthesize_N][cb_N]
            //               [phonemize_{N+1}]...
            //   prefetched: [phonemize_N][synthesize_N+cb_N +
            //               phonemize_{N+1}+spawn_{N+1}]...
            //
            // Phonemize and the cb dispatch run on the foreground
            // task's thread; synthesize runs on a detached background
            // task. We spawn task_{N+1} BEFORE awaiting task_N, so by
            // the time we emit chunk N, chunk N+1's synthesize is
            // already underway. Phonemize_{N+1} and the cb for N
            // overlap with synthesize_{N+1} starting up.
            //
            // NOTE: kt_synthesize is single-threaded per ctx, so two
            // concurrent synthesizes on the same ctx would race. Tasks
            // are scheduled with a serialised infer queue so they run
            // in order; the depth-1 lookahead doesn't add inference
            // parallelism, just hides phonemize + cb cost. For real
            // parallelism (RTF ≈ 1 hardware), we'd need a second ctx —
            // costs an extra 35 MB of model weights but lets two
            // synthesizes run simultaneously.
            let speed = effectiveSpeed
            let speakSelf = self
            func taskFor(_ chunkText: String)
                    -> Task<[Float], Error> {
                let phonemes =
                    (try? Phonemizer.phonemize(chunkText)) ?? []
                let refId = min(chunkText.count, 399)
                let style = Array(
                    voiceRows[(refId * 256)..<((refId + 1) * 256)])
                return Task.detached(priority: .userInitiated) {
                    try speakSelf.synthesizeChunk(
                        phonemes: phonemes,
                        style:    style,
                        speed:    speed)
                }
            }
            // Kick off chunk[0] before the loop; each iter spawns
            // chunk[idx+1].
            var pendingTask: Task<[Float], Error>? =
                taskFor(chunks[0])
            for idx in chunks.indices {
                if let task = pendingTask {
                    let next = idx + 1
                    if next < chunks.count {
                        pendingTask = taskFor(chunks[next])
                    } else {
                        pendingTask = nil
                    }
                    let audio = try await task.value
                    let emit: [Float]
                    if idx == 0 {
                        emit = audio
                    } else {
                        emit = gapAfter(chunks[idx - 1]) + audio
                    }
                    if let cb = callback {
                        let int16 = emit.map { f -> Int16 in
                            // NaN/Inf guard before Int(.rounded())
                            // which traps on non-finite floats. ggml
                            // shouldn't produce them but the guard
                            // mirrors the MLX/CoreML paths for parity.
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
            }
        }
        return allAudio
    }

    private func synthesizeChunk(phonemes: [Int],
                                 style:    [Float],
                                 speed:    Float) throws -> [Float] {
        let c: OpaquePointer
        if let cur = ctx {
            c = cur
        } else {
            throw NSError(
                domain: "KittenTTSLlamaCpp",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey:
                    "context not initialised"])
        }
        precondition(style.count == 256,
                     "style256 must be 256 floats")
        let ids32: [Int32] = phonemes.map { p in Int32(p) }
        let t0 = Date()
        // Serialise — see inferLock doc comment.
        inferLock.lock()
        let audio: kt_audio = ids32.withUnsafeBufferPointer { idsBuf in
            style.withUnsafeBufferPointer { styBuf in
                kt_synthesize(c,
                              idsBuf.baseAddress, Int32(idsBuf.count),
                              styBuf.baseAddress, speed)
            }
        }
        inferLock.unlock()
        let elapsedMs = Date().timeIntervalSince(t0) * 1000.0
        let out: [Float]
        if let samples = audio.samples, audio.n_samples > 0 {
            defer { kt_audio_free(audio) }
            let n = Int(audio.n_samples)
            let buf = UnsafeBufferPointer(start: samples, count: n)
            out = Array(buf)
            onChunkMetrics?(ChunkMetrics(
                phonemes: phonemes.count,
                frames:   n / Self.audioPerFrame,
                compute:  .cpu,
                totalMs:  elapsedMs,
                samples:  n))
        } else {
            throw NSError(
                domain: "KittenTTSLlamaCpp",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey:
                    "kt_synthesize failed"])
        }
        return out
    }

    private func loadVoices() throws {
        let modelDir: URL
        if let d = ModelLoader.bundledModelDir() {
            modelDir = d
        } else {
            throw NSError(
                domain: "KittenTTSLlamaCpp",
                code: 7,
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

#endif // os(macOS) || os(iOS)

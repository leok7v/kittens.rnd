// CPUBackend.swift -- Swift wrapper around the pure-C / cblas kt_cpu
// backend. Cloned from ggml/GGMLBackend.swift; the only differences
// are the C symbol prefix (kt_ -> kt_cpu_), the class name, and the
// fixed-variant / compute labels reported to the UI.
//
// v1 is macOS + iOS. The C side is portable (cblas + libm); the only
// platform shim is in cpu/kt_tensor.c.
#if os(macOS) || os(iOS)

import Foundation
import os
// MLX is shared with the other backends for voices.safetensors loading.
import MLX

/// kt_tensor / Accelerate-backed alternative to KittenTTS and
/// KittenTTSCoreML. Same public `speak` API so KittensRnDApp can A/B
/// all four backends.
///
/// Implementation lives in `cpu/kittens-tts-cpu.c` (see KittensCPU.h
/// for the C API). The bridging header exposes the C symbols
/// (`kt_cpu_create`, `kt_cpu_synthesize`, ...) directly to Swift.
///
/// v1 caveats:
/// - CPU only.
/// - One inference per `kt_cpu_synthesize` call. Single-threaded per
///   ctx (matches the C API).
/// - kt_cpu_synthesize is currently a stub returning silence — the
///   four stages need porting from kittens-tts.c. See
///   cpu/kittens-tts-cpu.c for the TODO list.
public final nonisolated class KittenTTSCpu: @unchecked Sendable {

    public nonisolated struct Config: Sendable {
        public var speed:   Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed   = speed
            self.voiceID = voiceID
        }
    }

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

    // 24000 Hz / 80 Hz × 2 (matches GGML backend).
    static let audioPerFrame: Int = 600

    private var ctx: OpaquePointer?                  // kt_cpu_ctx *
    private var voiceEmbeds: [String: [Float]] = [:] // flat 400*256
    private let loadLock = OSAllocatedUnfairLock()
    /// Serialises synthesize calls — single-threaded per ctx by design.
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
        if let c = ctx { kt_cpu_destroy(c) }
    }

    public func preload() async throws {
        try loadLock.withLock {
            if ctx == nil {
                let modelDir: URL
                if let d = ModelLoader.bundledModelDir() {
                    modelDir = d
                } else {
                    throw NSError(
                        domain: "KittenTTSCpu",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey:
                            "no bundled model dir"])
                }
                let ggufURL = modelDir.appendingPathComponent(
                    "kitten_full.gguf")
                if !FileManager.default.fileExists(
                        atPath: ggufURL.path) {
                    throw NSError(
                        domain: "KittenTTSCpu",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "kitten_full.gguf not found at " +
                            "\(ggufURL.path)"])
                }
                let cgguf = ggufURL.path.cString(using: .utf8)!
                ctx = kt_cpu_create(cgguf)
                if ctx == nil {
                    throw NSError(
                        domain: "KittenTTSCpu",
                        code: 3,
                        userInfo: [NSLocalizedDescriptionKey:
                            "kt_cpu_create failed"])
                }
            }
            if voiceEmbeds.isEmpty {
                try loadVoices()
            }
        }
    }

    public func unload() {
        loadLock.withLock {
            if let c = ctx { kt_cpu_destroy(c); ctx = nil }
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
        let voiceID = KittenTTSCpu.voiceAliases[config.voiceID]
                          ?? config.voiceID
        let voiceRows: [Float]
        if let v = voiceEmbeds[voiceID]
                ?? voiceEmbeds["expr-voice-5-m"] {
            voiceRows = v
        } else {
            throw NSError(
                domain: "KittenTTSCpu",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "voice '\(voiceID)' not found"])
        }
        let effectiveSpeed = config.speed
                           * (KittenTTS.speedPriors[voiceID] ?? 1.0)
        let normalised = TextPreprocessor.process(text)
        // kt_cpu allocates buffers dynamically — no L=400 bucket like
        // CoreML. Whole sentences stay as a single chunk regardless of
        // length (chunker only breaks on .!?).
        // Paragraph-aware chunking with em-dash breaks: speaker turns
        // stay in one chunk for prosody continuity, narrator/dialog
        // interleaves ("That's right, — nodded the Halfling, — But
        // please call them hobbits.") split at " — " so each
        // sub-piece gets its own brief beat. priorSilenceMs encodes
        // the silence to emit before each chunk (0/60/120 ms).
        let chunks = TextChunker.phonemizedChunks(normalised)
        var allAudio: [Float] = []
        if !chunks.isEmpty {
            let speed = effectiveSpeed
            let speakSelf = self
            func taskFor(_ idx: Int) -> Task<[Float], Error> {
                let c = chunks[idx]
                let refId = min(c.text.count, 399)
                let style = Array(
                    voiceRows[(refId * 256)..<((refId + 1) * 256)])
                let phonemes = c.phonemes
                return Task.detached(priority: .userInitiated) {
                    try speakSelf.synthesizeChunk(
                        phonemes: phonemes,
                        style:    style,
                        speed:    speed)
                }
            }
            var pendingTask: Task<[Float], Error>? = taskFor(0)
            // Ensure any prefetched-but-unconsumed task gets cancelled
            // if we exit the loop (Stop pressed, error thrown, etc.) —
            // synthesizeChunk checks Task.isCancelled and skips the C
            // call instead of running it for output the caller has
            // already abandoned.
            defer { pendingTask?.cancel() }
            for idx in chunks.indices {
                // Top-of-iteration cancellation gate: when speakTask
                // was cancelled (Stop, backend switch) we still
                // have to drain the in-flight detached task by
                // awaiting it, but we MUST NOT start another chunk
                // or process its audio.
                if Task.isCancelled {
                    pendingTask?.cancel()
                    break
                }
                if let task = pendingTask {
                    let next = idx + 1
                    pendingTask = next < chunks.count
                        ? taskFor(next) : nil
                    let audio: [Float]
                    do {
                        audio = try await task.value
                    } catch is CancellationError {
                        // The prefetched task we just awaited was
                        // already cancelled — nothing to emit.
                        break
                    }
                    if Task.isCancelled { break }
                    // Silences scale inversely with effectiveSpeed —
                    // at speed=1.5 the spoken audio is 1/1.5x the
                    // duration, so a fixed 180 ms paragraph beat
                    // sounds disproportionately long. Divide by
                    // speed to keep the *perceived* pause consistent.
                    let silenceN = Int(
                        Double(chunks[idx].priorSilenceMs) * 24.0
                        / Double(speed))
                    let emit: [Float] = silenceN <= 0
                        ? audio
                        : [Float](repeating: 0, count: silenceN)
                          + audio
                    if let cb = callback {
                        let int16 = emit.map { f -> Int16 in
                            // Clamp to [-1, 1] BEFORE scaling so the
                            // intermediate Int can't overflow when the
                            // C side produces a stray huge value (e.g.
                            // mag = exp(big) blowing up iSTFT).
                            let clamped: Float
                            if !f.isFinite { clamped = 0 }
                            else if f >  1 { clamped =  1 }
                            else if f < -1 { clamped = -1 }
                            else            { clamped = f }
                            return Int16(clamping:
                                Int((clamped * 32767.0).rounded()))
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
                domain: "KittenTTSCpu",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey:
                    "context not initialised"])
        }
        precondition(style.count == 256,
                     "style256 must be 256 floats")
        let ids32: [Int32] = phonemes.map { p in Int32(p) }
        let t0 = Date()
        // Early-exit if this Task was cancelled before we even
        // started — saves a full synthesize round on prefetched
        // chunks that the caller has since abandoned (Stop pressed,
        // backend switched, etc.). The C call kt_cpu_synthesize is
        // not itself interruptible, so once it starts it runs to
        // completion; the most we can do is skip it.
        if Task.isCancelled { throw CancellationError() }
        inferLock.lock()
        if Task.isCancelled {
            inferLock.unlock()
            throw CancellationError()
        }
        let audio: kt_cpu_audio = ids32.withUnsafeBufferPointer { idsBuf in
            style.withUnsafeBufferPointer { styBuf in
                kt_cpu_synthesize(c,
                                  idsBuf.baseAddress, Int32(idsBuf.count),
                                  styBuf.baseAddress, speed)
            }
        }
        inferLock.unlock()
        let elapsedMs = Date().timeIntervalSince(t0) * 1000.0
        let out: [Float]
        if let samples = audio.samples, audio.n_samples > 0 {
            defer { kt_cpu_audio_free(audio) }
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
                domain: "KittenTTSCpu",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey:
                    "kt_cpu_synthesize failed"])
        }
        return out
    }

    private func loadVoices() throws {
        let modelDir: URL
        if let d = ModelLoader.bundledModelDir() {
            modelDir = d
        } else {
            throw NSError(
                domain: "KittenTTSCpu",
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

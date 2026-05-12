#if os(macOS) || os(iOS)
import Foundation
import CoreML

/// Headless A/B benchmark. Invoked when the binary starts with `--bench`.
///
/// Runs each selected config (backend × variant × compute) on a fixed
/// phrase, saves a `.wav` per run, prints CSV rows suitable for paste.
///
/// Usage:
///   KittenApp --bench [--text "..."] [--out ./tmp] [--runs 2]
///             [--voice Kiki]
///             [--configs "mlx,coreml/fp32/all,coreml/int8w/all,..."]
///   (omit --configs to run the full 9-row matrix)
enum KittenBench {

    static func run(args: [String]) async {
        var text = "Kitten TTS is now streaming audio chunks " +
                   "for lower latency."
        var outDir = URL(
            fileURLWithPath: "tmp",
            relativeTo: URL(fileURLWithPath:
                FileManager.default.currentDirectoryPath))
        var runs = 2
        var voice = "Kiki"
        var wanted: [String]? = nil
        var i = 1
        while i < args.count {
            let a = args[i]
            switch a {
            case "--text":
                i += 1
                if i < args.count { text = args[i] }
            case "--out":
                i += 1
                if i < args.count {
                    outDir = URL(fileURLWithPath: args[i])
                }
            case "--runs":
                i += 1
                if i < args.count {
                    runs = max(1, Int(args[i]) ?? runs)
                }
            case "--voice":
                i += 1
                if i < args.count { voice = args[i] }
            case "--configs":
                i += 1
                if i < args.count {
                    wanted = args[i].split(separator: ",")
                                    .map { s in String(s) }
                }
            default:
                break
            }
            i += 1
        }
        do {
            try FileManager.default.createDirectory(
                at: outDir, withIntermediateDirectories: true)
        } catch {
            let msg = "createDirectory failed for " +
                      "\(outDir.path): \(error)\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
        let resolvedOut = URL(fileURLWithPath: outDir.path)
                              .standardizedFileURL
        // Build the matrix from variants that are actually bundled, so
        // the documented default command produces a clean run instead
        // of a wall of "bundle has no kitten_text_<variant>.mlpackage"
        // errors. Users who want to bench an unbundled variant can pass
        // `--configs coreml/fp16/all,...` explicitly after running the
        // CoreML conversion script.
        let coreMLRows: [BenchConfig] =
            KittenTTSCoreML.availableVariants.flatMap { v in
                KittenTTSCoreML.Compute.allCases.map { c in
                    BenchConfig(
                        label: "coreml/\(v.rawValue)/" +
                               "\(c.rawValue.lowercased())",
                        backend: .coreml,
                        variant: v,
                        compute: c)
                }
            }
        var allConfigs: [BenchConfig] = [
            .init(label:   "mlx",
                  backend: .mlx,
                  variant: nil,
                  compute: nil),
        ]
        allConfigs.append(contentsOf: coreMLRows)
        allConfigs.append(.init(label:   "ggml",
                                backend: .ggml,
                                variant: nil,
                                compute: nil))
        allConfigs.append(.init(label:   "cpu",
                                backend: .cpu,
                                variant: nil,
                                compute: nil))
        let configs: [BenchConfig]
        if let want = wanted {
            configs = allConfigs.filter { c in
                want.contains(c.label)
            }
        } else {
            configs = allConfigs
        }
        writeOut("# text  : \(text)\n")
        writeOut("# voice : \(voice)\n")
        writeOut("# runs  : \(runs)\n")
        writeOut("# out   : \(resolvedOut.path)\n")
        writeOut("# configs: \(configs.count)\n\n")
        writeOut("label,run,ttf_ms,total_ms,audio_s,rtf,samples,wav\n")
        for cfg in configs {
            await runOne(cfg:    cfg,
                         text:   text,
                         voice:  voice,
                         runs:   runs,
                         outDir: outDir)
        }
    }

    private struct BenchConfig {
        let label:   String
        let backend: Backend
        let variant: KittenTTSCoreML.Variant?
        let compute: KittenTTSCoreML.Compute?
    }

    private static func runOne(cfg:    BenchConfig,
                               text:   String,
                               voice:  String,
                               runs:   Int,
                               outDir: URL) async {
        // Fresh instance per config so cold numbers are honest. All
        // three backends are constructed once and reused across runs;
        // run 0 captures the cold-start cost, runs 1+ are warm.
        let mlx = KittenTTS()
        let ml = KittenTTSCoreML()
#if os(macOS) || os(iOS)
        let ggml = KittenTTSLlamaCpp()
        let cpuB = KittenTTSCpu()
#endif
        for r in 0..<runs {
            let t0 = Date()
            var firstByte: Date? = nil
            var samples: [Float]? = nil
            do {
                switch cfg.backend {
                case .mlx:
                    samples = try await mlx.speak(
                        text: text,
                        config: KittenTTS.Config(
                            speed: 1.0, voiceID: voice),
                        callback: { _, _ in
                            if firstByte == nil { firstByte = Date() }
                        })
                case .coreml:
                    samples = try await ml.speak(
                        text: text,
                        config: KittenTTSCoreML.Config(
                            speed: 1.0, voiceID: voice),
                        variant: cfg.variant ?? .int8w,
                        compute: cfg.compute ?? .ane,
                        callback: { _, _ in
                            if firstByte == nil { firstByte = Date() }
                        })
#if os(macOS) || os(iOS)
                case .ggml:
                    samples = try await ggml.speak(
                        text: text,
                        config: KittenTTSLlamaCpp.Config(
                            speed: 1.0, voiceID: voice),
                        callback: { _, _ in
                            if firstByte == nil { firstByte = Date() }
                        })
                case .cpu:
                    samples = try await cpuB.speak(
                        text: text,
                        config: KittenTTSCpu.Config(
                            speed: 1.0, voiceID: voice),
                        callback: { _, _ in
                            if firstByte == nil { firstByte = Date() }
                        })
#endif
                }
            } catch {
                let msg = "ERROR \(cfg.label) run \(r): " +
                          "\(error.localizedDescription)\n"
                FileHandle.standardError.write(Data(msg.utf8))
            }
            if let s = samples {
                let total = Date().timeIntervalSince(t0) * 1000.0
                let ttf   = (firstByte ?? Date())
                                .timeIntervalSince(t0) * 1000.0
                let audio = Double(s.count) / 24000.0
                let rtf   = audio / (total / 1000.0)
                let safe  = cfg.label.replacingOccurrences(
                                of: "/", with: "_")
                let wavURL = outDir.appendingPathComponent(
                    "\(safe)_r\(r).wav")
                if let werr = writeWAV(samples: s, to: wavURL) {
                    let msg = "WAV write failed for " +
                              "\(wavURL.path): \(werr)\n"
                    FileHandle.standardError.write(Data(msg.utf8))
                }
                writeOut(String(
                    format: "%@,%d,%.0f,%.0f,%.3f,%.2f,%d,%@\n",
                    cfg.label, r, ttf, total, audio, rtf,
                    s.count, wavURL.path))
            }
        }
    }

    private static func writeOut(_ s: String) {
        FileHandle.standardOutput.write(Data(s.utf8))
    }

    private static func writeWAV(samples: [Float],
                                 to url: URL) -> Error? {
        let sampleRate: UInt32 = 24_000
        let numSamples = UInt32(samples.count)
        let dataBytes  = numSamples * 2
        let fileBytes  = 36 + dataBytes
        var data = Data(capacity: Int(44 + dataBytes))
        data.append("RIFF".data(using: .ascii)!)
        data.append(UInt32ToLE(fileBytes))
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.append(UInt32ToLE(16))              // PCM subchunk size
        data.append(UInt16ToLE(1))               // format = PCM
        data.append(UInt16ToLE(1))               // channels
        data.append(UInt32ToLE(sampleRate))
        data.append(UInt32ToLE(sampleRate * 2))  // byte rate
        data.append(UInt16ToLE(2))               // block align
        data.append(UInt16ToLE(16))              // bits per sample
        data.append("data".data(using: .ascii)!)
        data.append(UInt32ToLE(dataBytes))
        for s in samples {
            // NaN/Inf guard: fp16 CoreML on .cpuOnly occasionally
            // produces non-finite values during length regulation;
            // without this Int16(_:) traps with "Float value cannot
            // be converted to Int".
            let safe = s.isFinite ? max(-1.0, min(1.0, s)) : 0
            let i16 = Int16(safe * 32767.0)
            data.append(UInt16ToLE(UInt16(bitPattern: i16)))
        }
        var err: Error? = nil
        do {
            try data.write(to: url)
        } catch {
            err = error
        }
        return err
    }

    private static func UInt32ToLE(_ v: UInt32) -> Data {
        var x = v.littleEndian
        return Data(bytes: &x, count: 4)
    }

    private static func UInt16ToLE(_ v: UInt16) -> Data {
        var x = v.littleEndian
        return Data(bytes: &x, count: 2)
    }
}
#endif

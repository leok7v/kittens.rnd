import SwiftUI
import Combine
import MLX
import AVFoundation
#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// App entry point. When invoked with `--bench`, we run a headless
/// benchmark and exit without bringing up SwiftUI. Otherwise we hand
/// off to the app.
@main
enum AppEntryPoint {
    static func main() async {
        #if os(macOS)
        let args = CommandLine.arguments
        if args.contains("--bench") {
            await KittenBench.run(args: args)
            exit(0)
        }
        #endif
        KittensRnDApp.main()
    }
}

struct KittensRnDApp: App {
    init() {
        #if os(macOS)
        NSApplication.shared.setActivationPolicy(.regular)
        #endif
    }
    var body: some Scene {
        WindowGroup {
            KittenTTSView()
        }
    }
}

final class AudioPlayer: NSObject, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let mixer:  AVAudioMixerNode
    private let format: AVAudioFormat
    // Monotonic generation counter: every call to stop() bumps it,
    // and any stale chunks from an in-flight inference that arrive
    // AFTER stop() will see a mismatched id and be dropped. Prevents
    // audible "stop -> silence -> resume as another chunk lands"
    // behaviour.
    private let lock = NSLock()
    private var generation: UInt64 = 0

    override init() {
        self.mixer = engine.mainMixerNode
        self.format = AVAudioFormat(
            standardFormatWithSampleRate: 24000, channels: 1)!
        super.init()
        #if !os(macOS)
        // iOS / iPadOS / visionOS / watchOS: an unconfigured
        // AVAudioSession can default to a category that gets muted by
        // the silent switch (.soloAmbient) or routed nowhere. Force
        // .playback so speech output is audible regardless of
        // ring/silent state.
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio,
                                 options: [.duckOthers])
        try? session.setActive(true, options: [])
        #endif
        engine.attach(player)
        engine.connect(player, to: mixer, format: format)
        try? engine.start()
    }

    /// Returns the id of the new generation — pass back to
    /// `playChunk` so chunks scheduled before the next `stop()` still
    /// play and chunks scheduled after are dropped.
    func beginGeneration() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        generation &+= 1
        return generation
    }

    func playChunk(samples: [Int16], generation gen: UInt64,
                   sampleRate: Double = 24000) {
        lock.lock()
        let current = generation
        lock.unlock()
        if gen == current {
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count))!
            buffer.frameLength = buffer.frameCapacity
            for i in 0..<samples.count {
                buffer.floatChannelData![0][i] =
                    Float(samples[i]) / 32767.0
            }
            player.scheduleBuffer(buffer, at: nil,
                                  options: [],
                                  completionHandler: nil)
            if !player.isPlaying { player.play() }
        }
    }

    func stop() {
        lock.lock(); generation &+= 1; lock.unlock()
        player.stop()
    }
}

private extension String {
    func paddedRight(_ width: Int) -> String {
        let result: String
        if count >= width {
            result = self
        } else {
            let pad = String(repeating: " ",
                             count: width - count)
            result = self + pad
        }
        return result
    }
}

enum Backend: String, CaseIterable, Identifiable {
    case mlx = "MLX"
    case coreml = "CoreML"
#if os(macOS) || os(iOS)
    case ggml = "GGML"
#endif
    var id: String { rawValue }

    /// What the Variant/Compute rows should show when this backend is
    /// active. Only CoreML actually has user-toggleable variants; for
    /// the others the values are baked into the on-disk weights and
    /// the runtime, so we display them as facts rather than disabled
    /// pickers showing stale CoreML state.
    var fixedVariantLabel: String {
        let label: String
        switch self {
        case .mlx:    label = "fp16"   // mlx-swift's default dtype
        case .coreml: label = ""       // unused — CoreML uses picker
#if os(macOS) || os(iOS)
        case .ggml:   label = "fp32"   // kitten_full.gguf is fp32
#endif
        }
        return label
    }
    var fixedComputeLabel: String {
        let label: String
        switch self {
        case .mlx:    label = "GPU"    // MLX dispatches to Metal
        case .coreml: label = ""       // unused
#if os(macOS) || os(iOS)
        case .ggml:   label = "CPU"    // CPU-only by design
#endif
        }
        return label
    }
}

struct LogEntry: Identifiable {
    let id = UUID()
    let time: Date
    let text: String
    let kind: Kind
    enum Kind { case info, metric, warn }
}

@MainActor
final class MetricsLog: ObservableObject {
    @Published var entries: [LogEntry] = []
    @Published var ramMB: Double = 0
    /// Running average RTF for the current Speak: total audio
    /// produced divided by wall-clock since `beginSpeak()`. Per-chunk
    /// RTF is misleading with the depth-1 prefetch — chunk N+1's
    /// "totalMs" includes time blocked on the inferLock waiting for
    /// chunk N — so we report the throughput across the whole
    /// utterance instead. 0 = no speak yet (or hasn't produced any
    /// audio).
    @Published var avgRTF: Double = 0
    private var speakStartTime: Date?
    private var speakTotalAudioS: Double = 0

    func info(_ s: String) {
        append(.init(time: Date(), text: s, kind: .info))
    }
    func metric(_ s: String) {
        append(.init(time: Date(), text: s, kind: .metric))
    }
    func warn(_ s: String) {
        append(.init(time: Date(), text: s, kind: .warn))
    }

    func updateRAM() { ramMB = KittenMetrics.residentMB() }

    func beginSpeak() {
        speakStartTime = Date()
        speakTotalAudioS = 0
        avgRTF = 0
    }

    func recordChunkAudio(_ audioS: Double) {
        speakTotalAudioS += audioS
        if let start = speakStartTime {
            let elapsed = Date().timeIntervalSince(start)
            if elapsed > 0 {
                avgRTF = speakTotalAudioS / elapsed
            }
        }
    }

    private func append(_ e: LogEntry) {
        entries.append(e)
        if entries.count > 200 {
            entries.removeFirst(entries.count - 200)
        }
        ramMB = KittenMetrics.residentMB()
    }
}

// Bank of sample prompts for quick testing — short demos plus story
// passages from misc/ (dialogues with mixed sentence lengths stress
// the chunker; long paragraphs stress bucketing). Story bodies are
// bundled as .txt files under Resources/prompts/ and loaded on first
// access.
struct SamplePrompt: Identifiable, Hashable {
    let id:    String
    let label: String
    let text:  String
}

enum SamplePrompts {
    // (filename-stem, menu label) for stories bundled under
    // Resources/prompts/.
    private static let storyCatalog: [(String, String)] = [
        ("07_baba_yaga",              "Baba Yaga"),
        ("08_shambambukli_creation",  "Shambambukli: Creation"),
        ("09_shambambukli_humans",    "Shambambukli: Humans"),
        ("10_ai_hype",                "AI Hype Unit"),
        ("11_orc_culture",            "High Culture Day"),
        ("12_dark_lord",              "The Dark Lord"),
        ("13_hamish_dougal",          "Hamish and Dougal"),
        ("14_in_the_shire",           "In the Shire"),
        ("15_dragon_contract",        "Dragon Contract"),
        ("16_goblin_park",            "Goblin Park"),
        ("17_leprechaun",             "The Leprechaun"),
        ("18_magic_science",          "Magic is Science"),
        ("19_galactic_exam",          "Galactic Exam"),
    ]

    private static let builtIns: [SamplePrompt] = [
        .init(id: "streaming",
              label: "Streaming demo",
              text: "Kitten TTS is now streaming audio chunks " +
                    "for lower latency."),
        .init(id: "pangram",
              label: "Pangram trio",
              text: """
              The quick brown fox jumps over the lazy dog.

              She sells seashells by the seashore.

              How much wood would a woodchuck chuck if a woodchuck could chuck wood?
              """),
        .init(id: "paragraph",
              label: "Long paragraph",
              text: """
              Speech is one of the most fundamental ways humans communicate with each other, \
              conveying not only the literal meaning of words but also emotion, intent, urgency, \
              humor, and personality through prosody, pitch, and timing. A good text to speech \
              system must capture all of these subtle cues in addition to getting the words \
              themselves right, which is why researchers have spent decades developing better \
              acoustic models, better voice embeddings, and better neural architectures to \
              faithfully reproduce the nuance of the human voice.
              """),
        .init(id: "numbers",
              label: "Numbers & dates",
              text: "In 1969, Apollo 11 landed on the moon. "
                  + "The mission cost about 25.4 billion dollars "
                  + "and brought back 21.5 kilograms of lunar rock."),
    ]

    static let all: [SamplePrompt] = {
        var out = builtIns
        for (stem, label) in storyCatalog {
            if let url = resourceURL(stem: stem),
               let body = try? String(contentsOf: url,
                                      encoding: .utf8) {
                out.append(.init(
                    id:    stem,
                    label: label,
                    text:  body.trimmingCharacters(
                        in: .whitespacesAndNewlines)))
            }
        }
        return out
    }()

    /// Xcode's filesystem-synchronized groups may put prompts/*.txt
    /// at the bundle root or keep them under a `prompts/` subpath.
    /// Try both.
    private static func resourceURL(stem: String) -> URL? {
        var found: URL? = nil
        if let u = Bundle.main.url(forResource: stem,
                                   withExtension: "txt") {
            found = u
        } else if let base = Bundle.main.resourceURL {
            let u = base.appendingPathComponent(
                "prompts/\(stem).txt")
            if FileManager.default.fileExists(atPath: u.path) {
                found = u
            }
        }
        return found
    }
}

struct KittenTTSView: View {
    // Default prompt on first launch — the Long Paragraph exercises
    // the chunker and stresses all L / N buckets so the user hears
    // the full quality envelope without having to pick anything.
    private static let defaultPromptID = "paragraph"
    @State private var text: String = {
        SamplePrompts.all
            .first(where: { p in p.id == defaultPromptID })?.text
            ?? SamplePrompts.all[0].text
    }()
    @AppStorage("prompt") private var promptID: String =
        defaultPromptID
    @State private var isGenerating: Bool = false
    @State private var status: String = "Loading model..."
    @State private var modelReady: Bool = false
    // Voice + backend + variant + compute persist; speed is
    // session-only.
    @AppStorage("voice")   private var voice: String = "Kiki"
    // GGML has the cheapest cold-start (~250 ms preload — the C side
    // mmap's the GGUF and skips the Core ML / MLX graph-compile
    // step), so it's the friendliest default on first launch.
    // Existing installs keep whatever they had via @AppStorage; only
    // fresh user defaults see this fall through.
    @AppStorage("backend") private var backend: Backend = .ggml
    // Default to int8w — only weights are quantized to 8-bit,
    // activations stay in fp16, so it runs cleanly on both ANE and
    // CPU. Bundled variants: int8w (~26 MB) and fp32 (~96 MB) — the
    // two that the upstream Kittens app empirically ships as actually
    // working. fp16 hangs in MLModel load on some machines and int8wa
    // quantizes the activations as well as the weights, which makes
    // vowels sound breathy/whispery. Both are reproducible via
    // scripts/convert_to_coreml.py if you want to A/B them.
    @AppStorage("variant") private var variantRaw: String =
        KittenTTSCoreML.Variant.int8w.rawValue
    // Default to CPU: warm RTF is on par with ANE/GPU on this model
    // (~22.6× either way) and the cold-start is the fastest of the
    // three, so first-Speak latency is best on CPU. ANE/GPU have to
    // pay a one-shot kernel-compile that the user actually feels.
    @AppStorage("compute") private var computeRaw: String =
        KittenTTSCoreML.Compute.cpu.rawValue
    @State private var speed: Float = 1.0
    @State private var speakTask: Task<Void, Never>? = nil
    @StateObject private var log = MetricsLog()
    /// False between a Speak tap and the first synthesized chunk
    /// landing in the cb. Covers the time CoreML spends lazy-
    /// compiling a fresh bucket inside speak() — modelReady is
    /// already true at that point so the spinner overlay would
    /// otherwise disappear during the multi-second compile.
    @State private var firstChunkArrived: Bool = false
    /// Backends that have hit a catastrophic failure on this device.
    /// Persisted across launches because the worst failure mode (MLX
    /// hitting a C++ SEGV inside mlx::core::allocator on memory-
    /// constrained iPhones) is a hard crash, not a Swift throw — so
    /// the in-session catch path can't see it. Combined with the
    /// `lastBackendAttempt` sentinel below, a crash gets detected on
    /// the next launch and the offending backend stays hidden.
    /// Stored as a comma-separated rawValue list because @AppStorage
    /// can't hold Set<Enum> directly.
    @AppStorage("failedBackends") private var failedBackendsRaw:
        String = ""
    /// Set to the backend's rawValue right before each speak() call
    /// and cleared on success or normal-throw catch. If the app
    /// launches and finds this non-empty, the previous run crashed
    /// inside that backend; we add it to `failedBackends` and clear.
    @AppStorage("lastBackendAttempt") private var
        lastBackendAttempt: String = ""

    private var failedBackends: Set<Backend> {
        Set(failedBackendsRaw
                .split(separator: ",")
                .compactMap { s in Backend(rawValue: String(s)) })
    }
    private func markBackendFailed(_ b: Backend) {
        var s = failedBackends
        s.insert(b)
        failedBackendsRaw = s.map(\.rawValue)
                             .sorted()
                             .joined(separator: ",")
    }
    /// Non-nil = an alert is being shown explaining a backend
    /// fallback. Cleared when the user dismisses it.
    @State private var alertMessage: String? = nil

    private var variant: KittenTTSCoreML.Variant {
        // If the persisted variant is no longer in the bundle, fall
        // back to the first available one rather than throwing at
        // speak.
        get {
            let stored =
                KittenTTSCoreML.Variant(rawValue: variantRaw)
                    ?? .int8wa
            let available = KittenTTSCoreML.availableVariants
            let result: KittenTTSCoreML.Variant
            if available.contains(stored) {
                result = stored
            } else {
                result = available.first ?? .int8wa
            }
            return result
        }
        nonmutating set { variantRaw = newValue.rawValue }
    }
    private var compute: KittenTTSCoreML.Compute {
        get {
            KittenTTSCoreML.Compute(rawValue: computeRaw) ?? .cpu
        }
        nonmutating set { computeRaw = newValue.rawValue }
    }

    private let mlxTTS = KittenTTS()
    private let coreMLTTS = KittenTTSCoreML()
#if os(macOS) || os(iOS)
    private let ggmlTTS = KittenTTSLlamaCpp()
#endif
    private let player = AudioPlayer()

    private var voiceOptions: [String] {
        KittenTTS.voiceDisplayOrder
    }

    /// Bands picked from the bench: GGML idle ~60 MB, CoreML int8w
    /// warm ~150 MB, fp32+ANE warmup peaks ~700 MB. > 500 = "you're
    /// loading a heavy variant"; > 700 = "you're past where it should
    /// be running steady-state, expect jetsam pressure on small
    /// devices".
    private func ramColor(_ mb: Double) -> Color {
        let c: Color
        switch mb {
        case ..<500: c = .secondary
        case ..<700: c = .yellow
        default:     c = .red
        }
        return c
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
                Text("KittenTTS R&D").font(.title).bold()

                HStack {
                    Text("Prompt").font(.caption)
                                  .foregroundColor(.secondary)
                    Spacer()
                    // SwiftUI's Picker(.menu) renders its selection
                    // text with a system-provided font that ignores
                    // `.font(.caption)`, so on iPhone "Numbers &
                    // dates" wraps to two big lines. A Menu with a
                    // hand-rolled label respects the font modifier
                    // and gives us line-clamp + ellipsis at the
                    // requested size.
                    let currentPromptLabel = SamplePrompts.all
                        .first(where: { p in p.id == promptID })?
                        .label
                        ?? SamplePrompts.all[0].label
                    Menu {
                        ForEach(SamplePrompts.all) { p in
                            Button(p.label) { promptID = p.id }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text(currentPromptLabel)
                                .font(.caption)
                                .lineLimit(1)
                                .truncationMode(.tail)
                            Image(systemName:
                                    "chevron.up.chevron.down")
                                .font(.caption2)
                        }
                    }
                    .frame(maxWidth: 240, alignment: .trailing)
                }

                TextEditor(text: $text)
                    .font(.body)
                    .frame(minHeight: 120, maxHeight: .infinity)
                    .padding(6)
                    .overlay(RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.5)))
                    .layoutPriority(1)

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Voice").font(.caption)
                                     .foregroundColor(.secondary)
                        Spacer()
                        Picker("", selection: $voice) {
                            ForEach(voiceOptions, id: \.self) { v in
                                Text(v).tag(v)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .font(.caption)
                        .lineLimit(1)
                    }
                    HStack {
                        Text("Backend").font(.caption)
                                       .foregroundColor(.secondary)
                        Spacer()
                        Picker("", selection: $backend) {
                            ForEach(Backend.allCases.filter { b in
                                !failedBackends.contains(b)
                            }) { b in
                                Text(b.rawValue).tag(b)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 240)
                        .disabled(isGenerating)
                    }
                    HStack {
                        Text("Variant").font(.caption)
                                       .foregroundColor(.secondary)
                        Spacer()
                        if backend == .coreml {
                            Picker("", selection: $variantRaw) {
                                ForEach(
                                    KittenTTSCoreML
                                        .availableVariants,
                                    id: \.rawValue
                                ) { v in
                                    Text(v.rawValue)
                                        .tag(v.rawValue)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 240)
                            .disabled(isGenerating)
                        } else {
                            // MLX / GGML have a single fixed weight
                            // format — show it as a static label
                            // rather than a disabled picker still
                            // showing stale CoreML state.
                            Text(backend.fixedVariantLabel)
                                .font(.caption.monospaced())
                                .foregroundColor(.secondary)
                        }
                    }
                    HStack {
                        Text("Compute").font(.caption)
                                       .foregroundColor(.secondary)
                        Spacer()
                        if backend == .coreml {
                            Picker("", selection: $computeRaw) {
                                ForEach(
                                    KittenTTSCoreML
                                        .Compute.allCases,
                                    id: \.rawValue
                                ) { c in
                                    Text(c.rawValue)
                                        .tag(c.rawValue)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 240)
                            .disabled(isGenerating)
                        } else {
                            // MLX runs on the Metal GPU; ggml is CPU
                            // only (decided after benching — LSTM
                            // kernels don't saturate the GPU). Show
                            // the fact, not a knob.
                            Text(backend.fixedComputeLabel)
                                .font(.caption.monospaced())
                                .foregroundColor(.secondary)
                        }
                    }
                    HStack {
                        Text("Speed").font(.caption)
                                     .foregroundColor(.secondary)
                        Slider(value: $speed,
                               in: 0.5...2.0, step: 0.05)
                        Text(String(format: "%.2f×", speed))
                            .font(.caption.monospaced())
                            .frame(width: 52, alignment: .trailing)
                            .foregroundColor(.secondary)
                    }
                }

                HStack(spacing: 12) {
                    // Single toggle button — fixed width so layout
                    // doesn't jump between Speak <-> Stop states.
                    Button(action: {
                        if isGenerating {
                            stopGeneration()
                        } else {
                            generateAndStream()
                        }
                    }) {
                        Label(isGenerating ? "Stop" : "Speak",
                              systemImage: isGenerating
                                  ? "stop.fill" : "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .frame(width: 120)
                    .disabled(!modelReady
                              || (!isGenerating && text.isEmpty))
                    .buttonStyle(.borderedProminent)
                    .tint(isGenerating ? .red : .accentColor)

                    Spacer()

                    if log.avgRTF > 0 {
                        // Running average across the current Speak —
                        // audio produced / wall-clock since start.
                        // > 1 = faster than realtime; < 1 means
                        // inference can't keep up with playback and
                        // audio will stutter.
                        Text(String(format: "RTF %.1f×",
                                    log.avgRTF))
                            .font(.caption.monospaced())
                            .foregroundColor(log.avgRTF < 1.0
                                             ? .orange
                                             : .secondary)
                    }
                    Text(String(format: "RAM %.0f MB", log.ramMB))
                        .font(.caption.monospaced())
                        .foregroundColor(ramColor(log.ramMB))
                }

                Text(status).font(.caption)
                            .foregroundColor(.secondary)

                // Metrics log pane — selectable text, plus a "Copy"
                // button for quick paste into investigation notes /
                // AI sessions.
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        // Match the rest of the labels (Prompt /
                        // Voice / Backend / Variant / Compute /
                        // Speed) so the log header doesn't visually
                        // float at a different size from everything
                        // else.
                        Text("Log").font(.caption)
                                   .foregroundColor(.secondary)
                        Spacer()
                        Button("Copy") { copyLogToPasteboard() }
                            .buttonStyle(.bordered)
                            .font(.caption)
                            .disabled(log.entries.isEmpty)
                        Button("Clear") {
                            log.entries.removeAll()
                        }
                            .buttonStyle(.bordered)
                            .font(.caption)
                    }
                    .padding(.horizontal, 6)

                    ScrollViewReader { reader in
                        ScrollView {
                            LazyVStack(alignment: .leading,
                                       spacing: 2) {
                                ForEach(log.entries) { e in
                                    Text(format(e))
                                        .font(.caption2
                                                  .monospaced())
                                        .foregroundColor(
                                            color(for: e.kind))
                                        .id(e.id)
                                        .frame(
                                            maxWidth: .infinity,
                                            alignment: .leading)
                                        .textSelection(.enabled)
                                }
                            }
                            .padding(6)
                        }
                        .frame(minHeight: 120, maxHeight: 260)
                        .background(
                            Color.secondary.opacity(0.08))
                        .clipShape(
                            RoundedRectangle(cornerRadius: 6))
                        .onChange(of: log.entries.count) { _, _ in
                            if let last = log.entries.last {
                                withAnimation(
                                    .linear(duration: 0.08)
                                ) {
                                    reader.scrollTo(
                                        last.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                }
        }
        .padding()
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 660)
        #endif
        // Modal-ish progress overlay during preload / backend switch
        // / warm-up. Blocks the underlying controls (so the user
        // can't start a Speak while a CoreML bucket compile is
        // mid-flight) and gives a visible indicator that something
        // is happening. Tied to `modelReady` which is the same flag
        // the Speak button already keys off.
        .overlay {
            if !modelReady
                || (isGenerating && !firstChunkArrived) {
                ZStack {
                    Color.black.opacity(0.45)
                        .ignoresSafeArea()
                    VStack(spacing: 12) {
                        ProgressView()
                            .controlSize(.large)
                        Text(status)
                            .font(.callout)
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding(24)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.black.opacity(0.7)))
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: modelReady)
        .animation(.easeInOut(duration: 0.15),
                   value: firstChunkArrived)
        .alert(
            "Backend unavailable",
            isPresented: Binding(
                get: { alertMessage != nil },
                set: { v in if !v { alertMessage = nil } }),
            actions: {
                Button("OK", role: .cancel) {
                    alertMessage = nil
                }
            },
            message: { Text(alertMessage ?? "") })
        .task {
            // Crash post-mortem: if the previous run wrote a
            // sentinel before calling speak() and never cleared it,
            // that run crashed (SEGV, jetsam, force-quit) inside
            // that backend. Add it to the persisted failed set,
            // clear the sentinel, push the picker to GGML if needed,
            // and pop an alert so the user knows why this backend
            // disappeared.
            if !lastBackendAttempt.isEmpty,
               let crashed = Backend(
                       rawValue: lastBackendAttempt) {
                if crashed != .ggml {
                    markBackendFailed(crashed)
                    if backend == crashed { backend = .ggml }
                    log.warn(
                        "\(crashed.rawValue) crashed last run — " +
                        "hidden, switched to GGML")
                    alertMessage =
                        "\(crashed.rawValue) crashed on this " +
                        "device during the previous run.\n\n" +
                        "Likely cause: the model is too memory-" +
                        "heavy for this device, or the Metal " +
                        "compiler service was jetsammed under " +
                        "pressure. \(crashed.rawValue) is a hard " +
                        "C++ crash that Swift can't catch in " +
                        "flight, so the only safe response is to " +
                        "hide it.\n\n" +
                        "It has been removed from the Backend " +
                        "picker for this install. GGML is " +
                        "selected as the fallback. To re-enable " +
                        "\(crashed.rawValue), delete the app and " +
                        "reinstall, or reset its data in Settings."
                }
                lastBackendAttempt = ""
            }
            // @State `text` is initialized before @AppStorage reads
            // the persisted `promptID`, so without this the view
            // always boots showing the builtIns[0] text regardless
            // of saved selection.
            if let p = SamplePrompts.all.first(
                    where: { sp in sp.id == promptID }) {
                text = p.text
            }
            await preloadModel()
        }
        .onChange(of: backend) { _, newValue in
            stopGeneration()
            Task { await switchBackend(to: newValue) }
        }
        .onChange(of: variantRaw) { _, _ in
            // Variant / compute changes invalidate the loaded
            // MLModel set.
            stopGeneration()
            coreMLTTS.unload()
            log.info(
                "variant → \(variantRaw)  " +
                "(unloaded CoreML models)")
            gatedWarmUp()
        }
        .onChange(of: computeRaw) { _, _ in
            stopGeneration()
            coreMLTTS.unload()
            log.info(
                "compute → \(computeRaw)  " +
                "(unloaded CoreML models)")
            gatedWarmUp()
        }
        .onChange(of: voice) { _, _ in
            stopGeneration()
        }
        .onChange(of: promptID) { _, newID in
            stopGeneration()
            if let p = SamplePrompts.all.first(
                    where: { sp in sp.id == newID }) {
                text = p.text
            }
        }
    }

    private static let timeFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private func format(_ e: LogEntry) -> String {
        "\(Self.timeFmt.string(from: e.time))  \(e.text)"
    }

    private func copyLogToPasteboard() {
        let text = log.entries.map(format).joined(separator: "\n")
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #else
        UIPasteboard.general.string = text
        #endif
    }

    private func color(for kind: LogEntry.Kind) -> Color {
        let c: Color
        switch kind {
        case .info:   c = .secondary
        case .metric: c = .primary
        case .warn:   c = .orange
        }
        return c
    }

    private func preloadModel() async {
        log.updateRAM()
        log.info("launch  RAM \(Int(log.ramMB)) MB")
        // Install metric callbacks once — they survive backend
        // switches.
        coreMLTTS.onBucketLoaded = { [weak log] name, ms in
            let msStr = String(format: "%5.0f", ms)
            Task { @MainActor in
                log?.metric(
                    "CoreML loaded   " +
                    "\(name.paddedRight(26)) \(msStr) ms")
            }
        }
        coreMLTTS.onChunkMetrics = { [weak log] m in
            let audioS = Double(m.samples) / 24000.0
            let totalMs = m.textStageMs + m.generatorStageMs
            let rtf = audioS / (totalMs / 1000.0)
            let tag = "\(m.variant.rawValue)/\(m.compute.rawValue)"
            Task { @MainActor in
                log?.recordChunkAudio(audioS)
                log?.metric(String(
                    format:
                        "CoreML/\(tag) chunk  " +
                        "phonemes=%d L=%d N=%d  " +
                        "text %.0fms gen %.0fms  " +
                        "audio %.2fs  RTF %.1fx",
                    m.phonemes, m.bucketL, m.bucketN,
                    m.textStageMs, m.generatorStageMs,
                    audioS, rtf))
            }
        }
        mlxTTS.onChunkMetrics = { [weak log] m in
            let audioS = Double(m.samples) / 24000.0
            let rtf = audioS / (m.elapsedMs / 1000.0)
            Task { @MainActor in
                log?.recordChunkAudio(audioS)
                log?.metric(String(
                    format:
                        "MLX    chunk   phonemes=%d" +
                        "                 elapsed %.0fms  " +
                        "audio %.2fs  RTF %.1fx",
                    m.phonemes, m.elapsedMs, audioS, rtf))
            }
        }
#if os(macOS) || os(iOS)
        ggmlTTS.onChunkMetrics = { [weak log] m in
            let audioS = Double(m.samples) / 24000.0
            let rtf = audioS / (m.totalMs / 1000.0)
            Task { @MainActor in
                log?.recordChunkAudio(audioS)
                log?.metric(String(
                    format:
                        "GGML   chunk   phonemes=%d frames=%d" +
                        "         total %.0fms  audio %.2fs  " +
                        "RTF %.1fx",
                    m.phonemes, m.frames, m.totalMs,
                    audioS, rtf))
            }
        }
#endif
        // Only the currently-selected backend is resident at any
        // time.
        await loadBackend(backend)
        // ANE compile for the default bucket pair happens here,
        // before `modelReady = true`, so the Speak button stays
        // disabled until the first utterance can start without
        // paying the cold-compile tax. Per-device ANE compile is
        // cached on disk between launches
        // (NSCachesDirectory/KittenTTS/...), so this is only slow on
        // the first run per (variant × compute × device).
        if backend == .coreml {
#if os(macOS)
            status = "Warming up..."
            await coreMLTTS.warmUpAll(variant: variant,
                                      compute: compute)
#endif
        }
        modelReady = true
        status = "Ready"
    }

    /// Swap the active backend: drop the old one's memory, preload
    /// the new one.
    private func switchBackend(to newBackend: Backend) async {
        modelReady = false
        status = "Switching to \(newBackend.rawValue)..."
        // Unload every other backend so only the active one stays
        // resident.
        for b in Backend.allCases where b != newBackend {
            await unloadBackend(b)
        }
        await loadBackend(newBackend)
        if newBackend == .coreml {
#if os(macOS)
            status = "Warming up..."
            await coreMLTTS.warmUpAll(variant: variant,
                                      compute: compute)
#endif
        }
        modelReady = true
        status = "Ready"
    }

    private func backendTag(_ b: Backend) -> String {
        let tag: String
        switch b {
        case .mlx:    tag = "MLX".paddedRight(6)
        case .coreml: tag = "CoreML".paddedRight(6)
#if os(macOS) || os(iOS)
        case .ggml:   tag = "GGML".paddedRight(6)
#endif
        }
        return tag
    }

    private func loadBackend(_ b: Backend) async {
        let tag = backendTag(b)
        let ramBefore = KittenMetrics.residentMB()
        let t0 = Date()
        do {
            switch b {
            case .mlx:    try await mlxTTS.preload()
            case .coreml: try await coreMLTTS.preload()
#if os(macOS) || os(iOS)
            case .ggml:   try await ggmlTTS.preload()
#endif
            }
            let ms = Date().timeIntervalSince(t0) * 1000
            let ramAfter = KittenMetrics.residentMB()
            log.metric(String(
                format: "\(tag) preload  %5.0f ms   " +
                        "RAM %.0f → %.0f MB",
                ms, ramBefore, ramAfter))
        } catch {
            log.warn(
                "\(b.rawValue) preload failed: " +
                "\(error.localizedDescription)")
        }
    }

    private func unloadBackend(_ b: Backend) async {
        let tag = backendTag(b)
        let ramBefore = KittenMetrics.residentMB()
        switch b {
        case .mlx:    mlxTTS.unload()
        case .coreml: coreMLTTS.unload()
#if os(macOS) || os(iOS)
        case .ggml:   ggmlTTS.unload()
#endif
        }
        // Give autorelease pools a moment to actually drop the
        // pages, then ask libc to return free-list pages to the
        // kernel. Without this, RSS as reported by Activity Monitor
        // (KittenMetrics.residentMB) stays at the high-water mark
        // even after Swift releases the underlying objects, because
        // libsystem_malloc holds freed pages on per-zone free lists
        // for reuse. malloc_zone_pressure_relief tells every zone to
        // dump those lists. This doesn't free CoreML's e5rt /
        // Espresso compile cache or Metal's pipeline-state cache —
        // those are process-wide and only the OS reclaims under
        // memory pressure — but it does measurably drop RSS after
        // backend switches.
        try? await Task.sleep(nanoseconds: 50_000_000)
        malloc_zone_pressure_relief(nil, 0)
        let ramAfter = KittenMetrics.residentMB()
        log.metric(String(
            format: "\(tag) unload          " +
                    "RAM %.0f → %.0f MB",
            ramBefore, ramAfter))
    }

    private func stopGeneration() {
        speakTask?.cancel()
        speakTask = nil
        player.stop()
        isGenerating = false
        status = "Stopped"
    }

    /// Block until ANE compile for the default bucket pair completes
    /// for the current variant/compute. Flips `modelReady` around
    /// the call so the Speak button is disabled while warmup runs.
    ///
    /// On iOS we skip the all-bucket warmup. Old iPhones can't fit
    /// every L×N×compute bucket in memory simultaneously — the
    /// cumulative ANE allocations push the device into jetsam
    /// pressure, taking down the Metal shader compiler service
    /// mid-launch and breaking MLX too. Lazy load: each Speak
    /// compiles only the bucket that fits its chunk, paying the cost
    /// on first use of that bucket.
    private func gatedWarmUp() {
        if backend == .coreml {
#if os(macOS)
            modelReady = false
            status = "Warming up..."
            Task(priority: .userInitiated) {
                [coreMLTTS, variant, compute] in
                await coreMLTTS.warmUpAll(variant: variant,
                                          compute: compute)
                await MainActor.run {
                    modelReady = true
                    status = "Ready"
                }
            }
#endif
        }
    }

    private func generateAndStream() {
        isGenerating = true
        firstChunkArrived = false
        let tag = "\(voice) / \(backend.rawValue)"
        status = "Speaking [\(tag)]..."
        log.beginSpeak()
        // Write the "attempt in flight" sentinel BEFORE we hand off
        // to the speak Task. If the C++ side SEGVs inside this
        // attempt (MLX/Metal on memory-constrained devices does this
        // — the crash bypasses MLX.withError because it happens
        // before MLX can call its own error handler), the next app
        // launch sees a non-empty `lastBackendAttempt` and infers
        // the crash, hiding this backend permanently. Cleared on
        // normal completion or catch.
        lastBackendAttempt = backend.rawValue
        player.stop()
        // Every Speak gets a fresh generation id. Chunks scheduled
        // AFTER a subsequent stop() (either by the user or by
        // another Speak) will see a mismatched id and be dropped by
        // the player.
        let gen = player.beginGeneration()
        let startTime = Date()
        var firstByteTime: Date?
        let captured = backend
        // `.utility` keeps inference below the audio I/O thread's
        // priority — since we already generate at RTF > 1×, leaving
        // headroom for AVAudioEngine prevents scrolling / compute
        // from starving audio output on smaller devices (iPhone SE
        // has only 2 performance cores).
        var chunkCount = 0
        speakTask = Task(priority: .utility) {
            do {
                let cb: (UnsafePointer<Int16>, Int) -> Void = {
                    pointer, count in
                    if !Task.isCancelled {
                        if firstByteTime == nil {
                            firstByteTime = Date()
                        }
                        chunkCount += 1
                        let samples = Array(
                            UnsafeBufferPointer(
                                start: pointer, count: count))
                        let chunkN = chunkCount
                        let backendName = tag
                        // Schedule on the audio thread directly —
                        // AVAudioPlayerNode.scheduleBuffer is
                        // thread-safe, and bouncing every chunk
                        // through the main queue stalls audio
                        // during UI scrolling on slower devices.
                        self.player.playChunk(
                            samples: samples, generation: gen)
                        DispatchQueue.main.async {
                            if !Task.isCancelled {
                                self.status =
                                    "Streaming [\(tag)]..."
                                self.log.info(
                                    "\(backendName) cb#\(chunkN) " +
                                    "samples=\(count)")
                                // First chunk landed — drop the
                                // warm-up overlay.
                                if !self.firstChunkArrived {
                                    self.firstChunkArrived = true
                                }
                            }
                        }
                    }
                }
                let capturedSpeed = speed
                let totalSamples: Int
                switch captured {
                case .mlx:
                    let cfg = KittenTTS.Config(
                        speed: capturedSpeed, voiceID: voice)
                    let s = try await mlxTTS.speak(
                        text: text, config: cfg, callback: cb)
                    totalSamples = s.count
                case .coreml:
                    let cfg = KittenTTSCoreML.Config(
                        speed: capturedSpeed, voiceID: voice)
                    let s = try await coreMLTTS.speak(
                        text: text, config: cfg,
                        variant: variant, compute: compute,
                        callback: cb)
                    totalSamples = s.count
#if os(macOS) || os(iOS)
                case .ggml:
                    let cfg = KittenTTSLlamaCpp.Config(
                        speed: capturedSpeed, voiceID: voice)
                    let s = try await ggmlTTS.speak(
                        text: text, config: cfg, callback: cb)
                    totalSamples = s.count
#endif
                }
                if !Task.isCancelled {
                    let totalMs = Date().timeIntervalSince(
                        startTime) * 1000
                    let ttfMs = (firstByteTime ?? Date())
                                    .timeIntervalSince(startTime)
                                * 1000
                    let audioS = Double(totalSamples) / 24000.0
                    let rtf = audioS / (totalMs / 1000.0)
                    // Synthesis is done, but audio is still
                    // playing. The Speak/Stop button must stay on
                    // Stop until the listener actually hears the
                    // last sample, otherwise on macOS — where
                    // ggml/mlx synthesize at 8-9× and CoreML at
                    // 22-25× warm — synthesis completes a few
                    // hundred ms after the click and the button
                    // reverts to Speak before the user can register
                    // the toggle. Sleep for the remaining
                    // wall-clock until the audio queue should be
                    // empty (chunks were scheduled in order; the
                    // last one finishes audioS after the first
                    // chunk hit the player). If stopGeneration()
                    // cancels the task during this sleep,
                    // Task.sleep throws and we fall into the catch
                    // below — exactly the behaviour we want.
                    let audioStartedAt = firstByteTime ?? startTime
                    let audioEndsAt = audioStartedAt
                                          .addingTimeInterval(
                                              audioS)
                    let remaining = audioEndsAt
                                        .timeIntervalSinceNow
                    if remaining > 0 {
                        try await Task.sleep(
                            nanoseconds:
                                UInt64(remaining * 1_000_000_000))
                    }
                    if !Task.isCancelled {
                        await MainActor.run {
                            self.isGenerating = false
                            self.speakTask = nil
                            self.status = String(
                                format: "Done [%@] in %.2fs",
                                tag, totalMs / 1000)
                            let bTag = self.backendTag(captured)
                            self.log.metric(String(
                                format:
                                    "\(bTag) SPEAK   " +
                                    "TTF %.0fms total %.0fms  " +
                                    "audio %.2fs  RTF %.1fx",
                                ttfMs, totalMs, audioS, rtf))
                            // Clean exit — the attempt completed
                            // without SEGV'ing, so wipe the crash
                            // sentinel.
                            self.lastBackendAttempt = ""
                        }
                    }
                }
            } catch {
                let failedBackend = captured
                let wasCancelled = (error is CancellationError)
                                || Task.isCancelled
                await MainActor.run {
                    self.isGenerating = false
                    self.speakTask = nil
                    // Normal-throw catch (vs. the SEGV path the
                    // launch sentinel handles): clear the crash
                    // sentinel since we got control back without
                    // trapping.
                    self.lastBackendAttempt = ""
                    if !wasCancelled {
                        self.log.warn(
                            "speak failed: " +
                            "\(error.localizedDescription)")
                        if failedBackend == .ggml {
                            // GGML is the always-works fallback;
                            // if it also failed, just surface the
                            // error.
                            self.status =
                                "Error [\(tag)]: " +
                                "\(error.localizedDescription)"
                        } else {
                            // MLX / CoreML failed in a recoverable
                            // way (Metal compiler jetsam surfaced
                            // via MLX.withError, ANE alignment
                            // hang, etc.). Persist the failure so
                            // the picker keeps the backend hidden
                            // across launches, switch to GGML, and
                            // surface the message in an alert.
                            self.markBackendFailed(failedBackend)
                            self.alertMessage =
                                "\(failedBackend.rawValue) " +
                                "failed on this device:\n\n" +
                                "\(error.localizedDescription)" +
                                "\n\n" +
                                "Falling back to GGML."
                            self.backend = .ggml
                            self.status = "Switched to GGML"
                        }
                    }
                    // wasCancelled: User clicked Stop (or a backend
                    // switch cancelled us) — not a backend failure.
                    // Don't mark anything as broken or pop an
                    // alert. stopGeneration() has already updated
                    // status.
                }
            }
        }
    }
}

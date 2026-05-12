// Shared text/IO pipeline used by all three backends:
//   TextPreprocessor -> TextChunker -> Phonemizer -> ModelLoader.
//
// CEPhonemizer C symbols come in via the bridging header — no import.
// MLX is needed only because `ModelLoader.bundledVoiceIDs()` parses
// the bundled `voices.safetensors` via `loadArrays`. The CoreML and
// GGML backends both rely on the same parsed dictionary, so the
// dependency is shared rather than MLX-only.

import Foundation
import MLX

enum TextPreprocessor {
    static func process(_ text: String) -> String {
        var t = text
        t = expandCurrency(t)
        t = expandPercentages(t)
        t = expandOrdinals(t)
        t = expandNumbers(t)
        t = cleanPunctuation(t)
        t = normaliseWhitespace(t)
        return t
    }

    private static func expandCurrency(_ text: String) -> String {
        let pattern =
            #"([$€£¥])(\d+(?:[,\d]*\d)?(?:\.\d+)?)(K|M|B|T)?"#
        var result = text
        if let re = try? NSRegularExpression(pattern: pattern) {
            let ns = text as NSString
            let r = NSRange(location: 0, length: ns.length)
            let matches = re.matches(in: text, range: r).reversed()
            for match in matches {
                let r1 = match.range(at: 1)
                let r2 = match.range(at: 2)
                let r3 = match.range(at: 3)
                let symbol     = r1.location != NSNotFound
                                     ? ns.substring(with: r1) : ""
                var amountStr  = r2.location != NSNotFound
                                     ? ns.substring(with: r2) : "0"
                let multiplier = r3.location != NSNotFound
                                     ? ns.substring(with: r3) : ""
                amountStr = amountStr.replacingOccurrences(
                    of: ",", with: "")
                var amount = Double(amountStr) ?? 0
                let multiplierWord: String
                switch multiplier {
                case "K":
                    amount *= 1_000
                    multiplierWord = " thousand"
                case "M":
                    amount *= 1_000_000
                    multiplierWord = " million"
                case "B":
                    amount *= 1_000_000_000
                    multiplierWord = " billion"
                case "T":
                    amount *= 1_000_000_000_000
                    multiplierWord = " trillion"
                default:
                    multiplierWord = ""
                }
                let currencyName: String
                switch symbol {
                case "$":
                    currencyName = amount == 1 ? " dollar"
                                               : " dollars"
                case "€":
                    currencyName = amount == 1 ? " euro" : " euros"
                case "£":
                    currencyName = amount == 1 ? " pound"
                                               : " pounds"
                case "¥":
                    currencyName = " yen"
                default:
                    currencyName = ""
                }
                let words: String
                if amount == floor(amount) {
                    words = numberToWords(Int(amount))
                          + multiplierWord + currencyName
                } else {
                    let intPart = numberToWords(Int(amount))
                    let fracPart = amountStr
                                       .components(separatedBy: ".")
                                       .last ?? ""
                    let fracWords = fracPart
                        .map { c in digitWord(String(c)) }
                        .joined(separator: " ")
                    words = intPart + " point " + fracWords
                          + multiplierWord + currencyName
                }
                result = replaceRange(result,
                                      nsRange: match.range,
                                      with: words)
            }
        }
        return result
    }

    private static func expandPercentages(_ text: String) -> String {
        let pattern = #"(\d+(?:\.\d+)?)\s*%"#
        var result = text
        if let re = try? NSRegularExpression(pattern: pattern) {
            let ns = text as NSString
            let r = NSRange(location: 0, length: ns.length)
            let matches = re.matches(in: text, range: r).reversed()
            for match in matches {
                let numStr = ns.substring(with: match.range(at: 1))
                let words: String?
                if let intVal = Int(numStr) {
                    words = numberToWords(intVal) + " percent"
                } else if let dbl = Double(numStr) {
                    let parts = numStr.components(separatedBy: ".")
                    let intWords  = numberToWords(Int(dbl))
                    let fracWords = (parts.last ?? "")
                        .map { c in digitWord(String(c)) }
                        .joined(separator: " ")
                    words = intWords + " point "
                          + fracWords + " percent"
                } else {
                    words = nil
                }
                if let w = words {
                    result = replaceRange(result,
                                          nsRange: match.range,
                                          with: w)
                }
            }
        }
        return result
    }

    private static let ordinalMap: [String: String] = [
        "1st":   "first",     "2nd":   "second",
        "3rd":   "third",     "4th":   "fourth",
        "5th":   "fifth",     "6th":   "sixth",
        "7th":   "seventh",   "8th":   "eighth",
        "9th":   "ninth",     "10th":  "tenth",
        "11th":  "eleventh",  "12th":  "twelfth",
        "13th":  "thirteenth", "14th": "fourteenth",
        "15th":  "fifteenth", "16th":  "sixteenth",
        "17th":  "seventeenth", "18th": "eighteenth",
        "19th":  "nineteenth", "20th": "twentieth",
        "21st":  "twenty-first",
        "22nd":  "twenty-second",
        "23rd":  "twenty-third",
        "30th":  "thirtieth",
        "40th":  "fortieth",
        "50th":  "fiftieth",
        "100th": "one hundredth",
        "1000th": "one thousandth",
    ]

    private static func expandOrdinals(_ text: String) -> String {
        let pattern = #"\b(\d+)(st|nd|rd|th)\b"#
        var result = text
        if let re = try? NSRegularExpression(
                pattern: pattern, options: .caseInsensitive) {
            let ns = text as NSString
            let r = NSRange(location: 0, length: ns.length)
            let matches = re.matches(in: text, range: r).reversed()
            for match in matches {
                let full = ns.substring(with: match.range)
                              .lowercased()
                if let word = ordinalMap[full] {
                    result = replaceRange(result,
                                          nsRange: match.range,
                                          with: word)
                } else {
                    let num = ns.substring(with: match.range(at: 1))
                    let suffix = ns.substring(with: match.range(at: 2))
                                     .lowercased()
                    if let n = Int(num) {
                        result = replaceRange(
                            result, nsRange: match.range,
                            with: numberToWords(n) + suffix)
                    }
                }
            }
        }
        return result
    }

    private static func expandNumbers(_ text: String) -> String {
        var t = text
        // Decimals first so "3.14" doesn't become "three .14".
        if let re = try? NSRegularExpression(
                pattern: #"\b(\d+)\.(\d+)\b"#) {
            let ns = t as NSString
            let r = NSRange(location: 0, length: ns.length)
            let matches = re.matches(in: t, range: r).reversed()
            for match in matches {
                let intPart  = ns.substring(with: match.range(at: 1))
                let fracPart = ns.substring(with: match.range(at: 2))
                let intWords = numberToWords(Int(intPart) ?? 0)
                let fracWords = fracPart
                    .map { c in digitWord(String(c)) }
                    .joined(separator: " ")
                t = replaceRange(t, nsRange: match.range,
                                 with: intWords + " point "
                                       + fracWords)
            }
        }
        // Integers (with optional comma separators).
        if let re = try? NSRegularExpression(
                pattern: #"\b\d{1,3}(?:,\d{3})*\b|\b\d+\b"#) {
            let ns = t as NSString
            let r = NSRange(location: 0, length: ns.length)
            let matches = re.matches(in: t, range: r).reversed()
            for match in matches {
                let numStr = ns.substring(with: match.range)
                                 .replacingOccurrences(of: ",",
                                                       with: "")
                if let n = Int(numStr) {
                    t = replaceRange(t, nsRange: match.range,
                                     with: numberToWords(n))
                }
            }
        }
        return t
    }

    private static func cleanPunctuation(_ text: String) -> String {
        var t = text.replacingOccurrences(
            of: "<[^>]+>", with: " ",
            options: .regularExpression)
        t = t.replacingOccurrences(of: "–", with: "—")
             .replacingOccurrences(of: " - ", with: " — ")
        return t
    }

    private static func normaliseWhitespace(_ text: String) -> String {
        // Preserve newlines through preprocessing so downstream
        // chunking can see paragraph structure. Earlier this collapsed
        // every whitespace run (including \n\n) to a single space,
        // which made TextChunker fall back to sentence-only splitting
        // and broke dialog/story prosody.
        // Pipeline:
        //   1) horizontal whitespace (spaces, tabs) -> single space
        //   2) line-leading whitespace stripped (so "  • foo" -> "• foo")
        //   3) 3+ consecutive newlines collapsed to \n\n
        //   4) trim outer whitespace
        var t = text.replacingOccurrences(
            of: #"[ \t]+"#, with: " ",
            options: .regularExpression)
        t = t.replacingOccurrences(
            of: #"(?m)^[ \t]+"#, with: "",
            options: .regularExpression)
        t = t.replacingOccurrences(
            of: #"\n{3,}"#, with: "\n\n",
            options: .regularExpression)
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func numberToWords(_ n: Int) -> String {
        let result: String
        if n < 0 {
            result = "negative " + numberToWords(-n)
        } else if n == 0 {
            result = "zero"
        } else {
            var s = ""
            var remaining = n
            if remaining >= 1_000_000_000 {
                s += numberToWords(remaining / 1_000_000_000)
                   + " billion "
                remaining %= 1_000_000_000
            }
            if remaining >= 1_000_000 {
                s += numberToWords(remaining / 1_000_000)
                   + " million "
                remaining %= 1_000_000
            }
            if remaining >= 1_000 {
                s += numberToWords(remaining / 1_000)
                   + " thousand "
                remaining %= 1_000
            }
            if remaining >= 100 {
                s += ones[remaining / 100] + " hundred "
                remaining %= 100
            }
            if remaining >= 20 {
                s += tens[remaining / 10]
                remaining %= 10
                if remaining > 0 { s += "-" + ones[remaining] }
            } else if remaining > 0 {
                s += ones[remaining]
            }
            result = s.trimmingCharacters(in: .whitespaces)
        }
        return result
    }

    private static let ones = [
        "",       "one",      "two",     "three",   "four",
        "five",   "six",      "seven",   "eight",   "nine",
        "ten",    "eleven",   "twelve",  "thirteen","fourteen",
        "fifteen","sixteen",  "seventeen","eighteen","nineteen",
    ]
    private static let tens = [
        "",       "",         "twenty",  "thirty",  "forty",
        "fifty",  "sixty",    "seventy", "eighty",  "ninety",
    ]

    private static func digitWord(_ d: String) -> String {
        let result: String
        switch d {
        case "0": result = "zero"
        case "1": result = "one"
        case "2": result = "two"
        case "3": result = "three"
        case "4": result = "four"
        case "5": result = "five"
        case "6": result = "six"
        case "7": result = "seven"
        case "8": result = "eight"
        case "9": result = "nine"
        default:  result = d
        }
        return result
    }

    private static func replaceRange(_ s: String,
                                     nsRange: NSRange,
                                     with replacement: String)
            -> String {
        var result = s
        if let range = Range(nsRange, in: s) {
            result = s.replacingCharacters(in: range,
                                           with: replacement)
        }
        return result
    }
}

struct TextChunker {
    /// Split text into sentences suitable for independent TTS
    /// generation.
    ///
    /// Earlier versions batched several sentences into one chunk to
    /// fit the model's 400-character budget — but the LSTM duration
    /// head smeared prosody across the batched boundary. A short
    /// leading "Yes." followed by a longer follow-up was pronounced
    /// as a single elongated word.
    ///
    /// Now every sentence is its own chunk. The caller streams them
    /// one at a time, and the natural sentence-final intonation of
    /// each short line is preserved. Paragraph breaks (blank lines)
    /// are collapsed — the caller inserts the pause between
    /// paragraphs itself.
    ///
    /// `maxLen` is a *character* ceiling. The default 280 is sized
    /// for the CoreML backend, whose buckets top out at L=400
    /// phonemes and N=1024 audio frames (~12.8 s). English
    /// phonemizes at ~1.0–1.4 phonemes per char, so a 280-char chunk
    /// worst-cases at ~392 phonemes — within L=400 with a small
    /// margin. Going higher than ~290 risks two audible bugs on
    /// CoreML: (a) phonemes past L=400 silently get truncated; (b)
    /// audio past N=1024 frames triggers the overflow-trim in
    /// speakOneChunk, which squeezes trailing-phoneme durations and
    /// makes the tail race.
    ///
    /// MLX and GGML allocate dynamically and have no bucket
    /// constraint — they pass `maxLen: .max` so whole sentences stay
    /// as a single chunk regardless of length. The last-resort
    /// `splitIfTooLong` comma-split otherwise breaks mid-thought
    /// (e.g. "...prosody," / "pitch, and timing.") and produces an
    /// audible pause inside the sentence even with the inter-chunk
    /// gap suppressed.
    static func chunk(_ text: String, maxLen: Int = 280) -> [String] {
        return phonemizedChunks(text).map { c in c.text }
    }

    /// A chunk plus the phonemes it produces, plus the silence to
    /// emit BEFORE this chunk (encoded as ms). Returned by
    /// `phonemizedChunks` — pre-phonemized so the backend skips a
    /// redundant phonemize call, and silences are computed centrally
    /// so all four backends agree on dialog timing.
    ///
    /// Typical priorSilenceMs values:
    ///   0   -> first chunk in stream / mid-paragraph sub-split
    ///   60  -> em-dash break (narrator/dialog interleave inside one
    ///          speaker turn): brief pause without resetting the arc
    ///   120 -> paragraph / speaker-turn break: a real beat
    struct PhonemizedChunk {
        public let text: String
        public let phonemes: [Int]
        public let priorSilenceMs: Int
    }

    /// Back-compat wrapper for callers that just want the text list.
    static func chunkParagraphs(_ text: String,
                                maxLen: Int = 280) -> [String] {
        return phonemizedChunks(text).map { c in c.text }
    }

    /// Inter-chunk silences. Tunable in one place. Order of magnitude:
    /// the model already emits a natural sentence-final fall after
    /// `. ! ?`, so the BREAK silences are layered ON TOP of that — a
    /// little goes a long way.
    public static let paragraphSilenceMs = 180   // between speaker turns
    public static let sentenceSilenceMs  = 80    // between sentences inside a turn
    public static let emDashSilenceMs    = 30    // narrator aside inside a turn

    /// Phoneme-aware paragraph chunker with em-dash narrator handling.
    ///
    /// Pipeline per paragraph (`\n\n` or bullet-marker delimited):
    ///   1. Strip the leading bullet glyph (• - — –) so the phonemizer
    ///      doesn't have to deal with it.
    ///   2. Split on " — " — in dialog these separate speech from a
    ///      narrator aside ("That's right, — nodded the Halfling,
    ///      — But please call them hobbits."). Each piece becomes its
    ///      own chunk with a short (60 ms) inter-piece silence so the
    ///      listener hears the beat between "what was said" and
    ///      "what the narrator described".
    ///   3. Phonemize each piece. If the result fits in
    ///      `maxPhonemes`, emit as-is. Otherwise fall back to per-
    ///      sentence chunks (priorSilenceMs=0 for the sub-splits so
    ///      the long-thought sentences run back-to-back).
    ///
    /// `maxPhonemes` defaults to 480: a conservative ceiling under
    /// the model's max_pos=512. Heavy punctuation can balloon
    /// phonemize ratios to ~3-4x char count, so char-based limits
    /// alone won't catch the overflow.
    static func phonemizedChunks(_ text: String,
                                 maxPhonemes: Int = 480)
                                 -> [PhonemizedChunk] {
        var out: [PhonemizedChunk] = []
        var emittedAny = false
        for paragraph in splitIntoParagraphs(text) {
            let stripped = stripBulletMarker(
                paragraph
                    .replacingOccurrences(of: "\n", with: " ")
                    .trimmingCharacters(in: .whitespaces))
            if stripped.isEmpty { continue }
            // Three-level split: paragraph -> em-dash pieces ->
            // sentences. Each sentence is its own chunk, with the
            // pre-silence chosen by the kind of break preceding it:
            //   paragraph break (between speaker turns) - full beat
            //   em-dash break   (narrator aside)        - small beat
            //   sentence break  (new sentence in turn)  - medium beat
            //   first chunk overall                     - none
            // Even on long compound paragraphs, sentence-grain pacing
            // is what listeners notice; the prosody-arc lost by
            // sub-splitting at "." is a worthwhile trade for
            // narration to not feel rushed.
            // Most human-typed text uses the ASCII hyphen (`-`) for
            // a pause, sometimes doubled (`--`). The preprocessor
            // upgrades ` - ` to ` — `, but be defensive in case the
            // chunker is fed raw text or the preprocessor was
            // bypassed: normalize all spaced-dash variants to em-dash
            // here too.
            let dashNormalized = stripped
                .replacingOccurrences(of: " -- ", with: " — ")
                .replacingOccurrences(of: " - ",  with: " — ")
                .replacingOccurrences(of: " – ",  with: " — ")
            let pieces = dashNormalized
                .components(separatedBy: " — ")
                .map { p in
                    p.trimmingCharacters(in: .whitespaces)
                }
                .filter { p in !p.isEmpty }
            for (pieceIdx, piece) in pieces.enumerated() {
                let sentences = splitSentences(
                    piece, maxLen: Int.max)
                for (sentIdx, sentence) in sentences.enumerated() {
                    let silence: Int
                    if !emittedAny {
                        silence = 0
                    } else if pieceIdx == 0 && sentIdx == 0 {
                        silence = paragraphSilenceMs
                    } else if sentIdx == 0 {
                        silence = emDashSilenceMs
                    } else {
                        silence = sentenceSilenceMs
                    }
                    let ph =
                        (try? Phonemizer.phonemize(sentence)) ?? []
                    // Phoneme-budget guard: if a single sentence
                    // somehow overflows max_pos=512, emit it anyway
                    // — the C side clamps position ids defensively
                    // so the model degrades rather than crashes.
                    _ = maxPhonemes
                    out.append(PhonemizedChunk(
                        text: sentence, phonemes: ph,
                        priorSilenceMs: silence))
                    emittedAny = true
                }
            }
        }
        return out
    }

    /// Strip a leading bullet marker plus its trailing whitespace.
    /// Only strips when the marker IS followed by whitespace, so
    /// `*emphasis*` and `-rad` stay intact; only true list-style
    /// bullets ("* foo", "•\tfoo", "- foo") get cleaned. Keeps the
    /// rest of the paragraph (including embedded em-dashes etc.)
    /// intact so the em-dash splitter can still see them.
    private static func stripBulletMarker(_ s: String) -> String {
        let markers: Set<Character> = ["•", "*", "-", "—", "–"]
        guard let first = s.first, markers.contains(first),
              s.count >= 2,
              s[s.index(after: s.startIndex)].isWhitespace
        else {
            return s
        }
        return String(s.dropFirst())
            .trimmingCharacters(in: .whitespaces)
    }

    /// Split text into paragraphs. A paragraph break is either a blank
    /// line (\n\n) OR a single newline immediately followed by a
    /// bullet marker — common in dialogue formatting and the only way
    /// the caller can express "new speaker turn" within a run-on
    /// block. Recognised bullet glyphs: `•`, `*`, `-`, `—`, `–`. The
    /// marker MUST be followed by a whitespace character (any flavor —
    /// space, tab, NBSP, etc.) so emphasis like `*italic*` and
    /// compound words like `well-known` are NOT treated as bullets.
    private static func splitIntoParagraphs(_ text: String) -> [String] {
        let withBreaks = text.replacingOccurrences(
            of: #"\n(?=[•*\-—–]\s)"#,
            with: "\n\n",
            options: .regularExpression)
        return withBreaks.components(separatedBy: "\n\n")
    }

    /// Split on .!? while keeping the terminal punctuation with the
    /// sentence. We deliberately do NOT split on `;:` — those mark
    /// sub-clause boundaries inside one thought, and breaking there
    /// causes audible pauses (every chunk boundary scheduled an
    /// inter-sentence gap and risked an audio-thread underrun on the
    /// transition). Sub-clause punctuation is left embedded in the
    /// chunk so the LSTM duration head can render it as natural
    /// prosody. Sentences that somehow exceed `maxLen` are split on
    /// commas / spaces to stay within the model's L=400 phoneme
    /// bucket.
    private static func splitSentences(_ text: String,
                                       maxLen: Int) -> [String] {
        var out: [String] = []
        let chars = Array(text)
        var start = chars.startIndex
        let enders: Set<Character> = [".", "!", "?"]
        var i = chars.startIndex
        while i < chars.endIndex {
            if enders.contains(chars[i]) {
                // Consume consecutive enders ("?!", "...", etc.).
                var j = i
                while j < chars.endIndex,
                      enders.contains(chars[j]) {
                    j += 1
                }
                let piece = String(chars[start..<j])
                                .trimmingCharacters(in: .whitespaces)
                if !piece.isEmpty { out.append(piece) }
                start = j
                i = j
            } else {
                i += 1
            }
        }
        let tail = String(chars[start..<chars.endIndex])
                       .trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty {
            // Give trailing fragments a soft terminator so prosody
            // resolves.
            if let last = tail.last, ".!?,;:".contains(last) {
                out.append(tail)
            } else {
                out.append(tail + ",")
            }
        }
        // Keep oversized sentences under maxLen via a last-resort
        // comma split.
        return out.flatMap { piece in
            splitIfTooLong(piece, maxLen: maxLen)
        }
    }

    private static func splitIfTooLong(_ s: String,
                                       maxLen: Int) -> [String] {
        let result: [String]
        if s.count <= maxLen {
            result = [s]
        } else {
            var built: [String] = []
            var current = ""
            for part in s.components(separatedBy: ", ") {
                if current.isEmpty {
                    current = part
                } else if current.count + 2 + part.count <= maxLen {
                    current += ", " + part
                } else {
                    built.append(current + ",")
                    current = part
                }
            }
            if !current.isEmpty { built.append(current) }
            result = built
        }
        return result
    }
}

struct ModelLoader {
    /// Locate the bundled model directory (holds voices.safetensors,
    /// weights, en_list, en_rules). Xcode's filesystem-synchronized
    /// groups can surface the folder either as a blue folder
    /// reference (preserves "nano/" path) or flatten individual
    /// files into the bundle root — try both.
    static func bundledModelDir() -> URL? {
        var dir: URL? = nil
        if let base = Bundle.main.resourceURL {
            let nano = base.appendingPathComponent(
                "nano", isDirectory: true)
            let nanoFile = nano.appendingPathComponent(
                "voices.safetensors")
            let baseFile = base.appendingPathComponent(
                "voices.safetensors")
            let fm = FileManager.default
            if fm.fileExists(atPath: nanoFile.path) {
                dir = nano
            } else if fm.fileExists(atPath: baseFile.path) {
                // Flat layout: files landed at bundle root.
                dir = base
            }
        }
        return dir
    }

    public static func bundledVoiceIDs() -> [String] {
        var ids: [String] = []
        if let dir = bundledModelDir() {
            let url = dir.appendingPathComponent(
                "voices.safetensors")
            if let voices = try? loadArrays(url: url) {
                ids = voices.keys.sorted()
            }
        }
        return ids
    }
}

struct Phonemizer {
    static let pad = "$"
    // Punctuation tier of the symbol table.
    private static let punctChars =
        ";:,.!?¡¿—…\"«»\"\" "
    private static let asciiAlpha =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
        "abcdefghijklmnopqrstuvwxyz"
    // IPA + diacritics tier. Keep in one literal to avoid risk of
    // splitting a combining sequence (`'̩'ᵻ` near the end).
    private static let ipaChars =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    static let symbols: [String] = {
        var s = ["$"]
        for char in punctChars { s.append(String(char)) }
        for char in asciiAlpha { s.append(String(char)) }
        for scalar in ipaChars.unicodeScalars {
            s.append(String(scalar))
        }
        return s
    }()
    static let dict: [String: Int] = Dictionary(
        symbols.enumerated().map { (idx, sym) in (sym, idx) },
        uniquingKeysWith: { _, new in new })

    /// Lazily-created C++ phonemizer handle backed by bundled
    /// en_rules / en_list. The C engine is thread-safe for reads
    /// after construction; Swift just can't prove it because
    /// UnsafeMutableRawPointer is non-Sendable.
    private nonisolated(unsafe) static let handle: PhonemizerHandle? = {
        var h: PhonemizerHandle? = nil
        if let modelDir = ModelLoader.bundledModelDir() {
            let rulesPath = modelDir
                .appendingPathComponent("en_rules").path
            let listPath  = modelDir
                .appendingPathComponent("en_list").path
            h = phonemizer_create(rulesPath, listPath, "en-us")
        }
        return h
    }()

    static func phonemize(_ text: String) throws -> [Int] {
        let ipa = try cePhonemize(text)
        var tokens: [Int] = [0]
        for scalar in ipa.unicodeScalars {
            if let idx = dict[String(scalar)] {
                tokens.append(idx)
            }
        }
        tokens.append(10)
        tokens.append(0)
        return tokens
    }

    /// Use the in-process CEPhonemizer C++ engine to get IPA.
    private static func cePhonemize(_ text: String) throws -> String {
        let h: PhonemizerHandle
        if let cur = handle {
            h = cur
        } else {
            throw NSError(
                domain: "KittenTTS", code: 3,
                userInfo: [NSLocalizedDescriptionKey:
                    "CEPhonemizer failed to load " +
                    "(missing en_rules/en_list?)"])
        }
        let result: UnsafeMutablePointer<CChar>
        if let r = phonemizer_phonemize(h, text) {
            result = r
        } else {
            throw NSError(
                domain: "KittenTTS", code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "CEPhonemizer failed to phonemize text"])
        }
        let ipa = String(cString: result)
        phonemizer_free_string(result)
        // Preserve trailing punctuation (CEPhonemizer may strip it,
        // upstream keeps it).
        var out = ipa
        let punctSet: Set<Character> =
            [".", "!", "?", ",", ";", ":", "…"]
        if let last = text.last, punctSet.contains(last),
           !out.hasSuffix(String(last)) {
            out += String(last)
        }
        return out
    }
}

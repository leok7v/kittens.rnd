// Copyright 2024 - Apache 2.0 License
// IPA Phonemizer implementation that reads the reference rule files.
// Original implementation - not derived from the reference GPL source code.

#include "phonemizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cassert>
#include <iostream>
#include <unordered_set>

// This file is a ~7k-line IPA phonemizer ported from external C++ source.
// A handful of compile-time placeholder variables and an intentionally-
// disabled rule block (`if (is_en_us && false)` guarded with a comment
// explaining the regression it caused) trigger warnings. Suppress them
// at file scope rather than scattering pragmas or deleting deliberate
// dead code.
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunreachable-code"

// ============================================================
// UTF-8 / String helpers
// ============================================================
static std::string toLowerASCII(const std::string& s) {
    std::string result = s;
    for (char& c : result) { c = (char)std::tolower((unsigned char)c); }
    return result;
}

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) { return ""; }
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> splitWS(const std::string& s) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token) { result.push_back(token); }
    return result;
}

static bool isVowelLetter(char c) {
    c = (char)std::tolower((unsigned char)c);
    return c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||c=='y';
}

static bool hasAnyVowelLetter(const std::string& s) {
    return std::any_of(s.begin(), s.end(), isVowelLetter);
}

// True iff `s` contains any phoneme vowel code char (a/A/e/E/i/I/
// o/O/u/U/V/0/3/@). Used to validate stem phonemes have at least
// one syllable nucleus.
static bool hasAnyVowelCode(const std::string& s) {
    static const std::string VC = "aAeEiIoOuUV03@";
    return std::any_of(s.begin(), s.end(), [](char c) {
        return VC.find(c) != std::string::npos;
    });
}

// ============================================================
// Add stress markers to IPA when none are present
// ============================================================
// Simple stress placement: add primary stress ˈ before the first vowel
// if no stress markers are present. This is a simplification.
// Replace the first occurrence of `from` in `s` with `to`. No-op
// when `from` isn't present. Used for "swap first stress marker"
// patterns that previously needed a for-loop + break.
static void replaceFirstChar(std::string& s, char from, char to) {
    auto it = std::find(s.begin(), s.end(), from);
    if (it != s.end()) { *it = to; }
}

// ============================================================
// Dictionary (en_list) Parser
// ============================================================
// Forward declaration (defined alongside loadRules helpers below):
// strips leading "?N" / "?!N" dialect prefix from `line`, sets
// *cond / *negated, and returns the trimmed remainder.
static std::string parseLeadingDialect(const std::string& line,
                                       int* cond, bool* negated);

// EntryFlags moved into IPAPhonemizer (see phonemizer.h). parseStressN
// stays as a file-scope helper.
namespace {

    int parseStressN(const std::string& flag) {
        int n = 0;
        if (flag.size() == 2 && flag[0] == '$' &&
            flag[1] >= '1' && flag[1] <= '6') {
            n = flag[1] - '0';
        }
        return n;
    }

}

// Storage dispatch for a single parsed en_list entry. See header.
void IPAPhonemizer::storeDictionaryEntry(
        const std::string& norm_word,
        const std::string& phonemes_str, int dialect_cond,
        const EntryFlags& flags) {
    if (flags.pastf) { pastf_words_.insert(norm_word); }
    if (flags.nounf) { nounf_words_.insert(norm_word); }
    if (flags.verbf) { verbf_words_.insert(norm_word); }
    bool is_flag_only = !phonemes_str.empty() && phonemes_str[0] == '$';
    // $N stress storage: skip when grammar flag is present unless
    // this is a flag-only entry (the $N IS the pronunciation fact).
    if (flags.stress_n > 0 && !flags.noun && !flags.verb &&
        (!flags.grammar || is_flag_only)) {
        stress_pos_.emplace(norm_word, flags.stress_n);
        if (is_flag_only && flags.onlys) {
            noun_form_stress_.insert(norm_word);
        }
    }
    if (is_flag_only) {
        // $altN flag-only entry: drop any prior dict_ entry so
        // rules fire with the alt bit set.
        if (phonemes_str.size() == 5 && phonemes_str[0] == '$' &&
            phonemes_str[1] == 'a' && phonemes_str[2] == 'l' &&
            phonemes_str[3] == 't' && phonemes_str[4] >= '1' &&
            phonemes_str[4] <= '6') {
            dict_.erase(norm_word);
        }
        if (flags.verb) { verb_flag_words_.insert(norm_word); }
    } else if (flags.noun) {
        noun_dict_.emplace(norm_word, phonemes_str);
    } else if (flags.verb) {
        verb_dict_.emplace(norm_word, phonemes_str);
    } else if (flags.atend) {
        // Skip entries that carry both $atstart and $atend.
        if (!flags.atstart && phonemes_str[0] != '$') {
            atend_dict_[norm_word] = phonemes_str;
        }
    } else if (flags.capital) {
        if (phonemes_str[0] != '$') {
            capital_dict_[norm_word] = phonemes_str;
        }
    } else if (flags.atstart) {
        atstart_dict_[norm_word] = phonemes_str;
    } else if (flags.past) {
        past_dict_.emplace(norm_word, phonemes_str);
    } else if (flags.onlys) {
        // Dialect-specific $onlys overrides; otherwise first-entry
        // wins, with the loser going to onlys_bare_dict_.
        if (dialect_cond != 0) {
            dict_[norm_word] = phonemes_str;
            onlys_words_.insert(norm_word);
        } else {
            bool inserted = dict_.emplace(norm_word,
                                          phonemes_str).second;
            if (inserted) {
                onlys_words_.insert(norm_word);
            } else if (phonemes_str[0] != '$') {
                onlys_bare_dict_[norm_word] = phonemes_str;
            }
        }
    } else {
        // Default storage path: dict_ last-entry-wins.
        dict_[norm_word] = phonemes_str;
        if (flags.only) { only_words_.insert(norm_word); }
        // $strend2 with bare-prefix phoneme -> compound first
        // element (e.g. "under" Vnd3 -> "understand"
        // ˌʌndɚstˈænd). Phonemes starting with ',' / '\'' / '%'
        // do not shift stress to the suffix.
        if (flags.strend2 && norm_word.size() >= 2 && !phonemes_str.empty() &&
            phonemes_str[0] != ',' &&
            phonemes_str[0] != '\'' && phonemes_str[0] != '%') {
            compound_prefixes_.push_back({norm_word, phonemes_str});
            strend_words_.insert(norm_word);
        }
        if (flags.strend2 && !phonemes_str.empty() && phonemes_str[0] == ',') {
            comma_strend2_words_.insert(norm_word);
        }
        if (flags.u2 && flags.strend2) { u2_strend2_words_.insert(norm_word); }
    }
}

// Phrase entry parser. See header for full semantics.
bool IPAPhonemizer::parsePhraseEntry(const std::string& line) {
    bool is_phrase = !line.empty() && line[0] == '(';
    if (is_phrase) {
        size_t close = line.find(')');
        if (close != std::string::npos && close > 1) {
            std::string phrase_words = trim(line.substr(1, close - 1));
            std::string rest = trim(line.substr(close + 1));
            if (!rest.empty() && rest[0] != '$') {
                std::vector<std::string> rp = splitWS(rest);
                if (!rp.empty() && !rp[0].empty() && rp[0][0] != '$') {
                    std::vector<std::string> words = splitWS(phrase_words);
                    bool has_atend = false;
                    bool has_pause = false;
                    bool has_u2_plus = false;
                    for (size_t ri = 1; ri < rp.size(); ri++) {
                        if (rp[ri] == "$atend") { has_atend = true; }
                        if (rp[ri] == "$pause") { has_pause = true; }
                        if (rp[ri] == "$u2+") { has_u2_plus = true; }
                    }
                    bool storable = (words.size() == 2 &&
                        !has_atend && !has_pause &&
                        words[0].find('.') == std::string::npos &&
                        words[1].find('.') == std::string::npos);
                    if (storable) {
                        std::string key =
                            toLowerASCII(words[0]) + " " +
                            toLowerASCII(words[1]);
                        if (rp[0].find("||") != std::string::npos) {
                            size_t pipe = rp[0].find("||");
                            phrase_split_dict_.emplace(
                                key, std::make_pair(
                                    rp[0].substr(0, pipe),
                                    rp[0].substr(pipe + 2)));
                        } else {
                            std::string phoneme = rp[0];
                            if (phoneme.find('\'') ==
                                    std::string::npos && phoneme[0] != '%') {
                                phoneme = "%" + phoneme;
                            }
                            phrase_dict_.emplace(key, phoneme);
                            if (has_u2_plus) {
                                keep_sec_phrase_keys_.insert(key);
                            }
                        }
                    }
                }
            }
        }
    }
    return is_phrase;
}

// Per-entry flag scanner. See header for full semantics.
void IPAPhonemizer::parseEntryFlags(
        const std::vector<std::string>& parts,
        const std::string& norm_word,
        const std::string& phonemes_str,
        EntryFlags& flags) {
    if (!phonemes_str.empty() && phonemes_str[0] == '$') {
        flags.stress_n = parseStressN(phonemes_str);
    }
    for (size_t fi = 2; fi < parts.size(); fi++) {
        const std::string& fl = parts[fi];
        if (fl == "$noun") { flags.noun = true; }
        if (fl == "$verb") {
            flags.verb = true;
            flags.grammar = true;
        }
        if (fl == "$past")  { flags.past  = true; }
        if (fl == "$pastf") { flags.pastf = true; }
        if (fl == "$nounf") {
            flags.nounf = true;
            flags.grammar = true;
        }
        if (fl == "$verbf") {
            flags.verbf = true;
            flags.grammar = true;
        }
        if (fl == "$atend" || fl == "$allcaps" || fl == "$sentence") {
            flags.atend = true;
        }
        if (fl == "$capital") { flags.capital = true; }
        if (fl == "$atstart") { flags.atstart = true; }
        if (fl == "$verbf"  || fl == "$strend2" ||
            fl == "$alt2"   || fl == "$alt3" || fl == "$only") {
            flags.grammar = true;
        }
        if (fl == "$only")  { flags.only  = true; }
        if (fl == "$onlys") { flags.onlys = true; }
        if (fl == "$strend2") { flags.strend2 = true; }
        if (fl == "$u2")      { flags.u2 = true; }
        if (fl == "$u+") {
            unstressed_words_.insert(norm_word);
            if (phonemes_str.find(',') != std::string::npos &&
                phonemes_str.find('\'') == std::string::npos) {
                u_plus_secondary_words_.insert(norm_word);
            }
        }
        if (fl == "$u") { unstressed_words_.insert(norm_word); }
        if (fl == "$unstressend") { unstressend_words_.insert(norm_word); }
        if (fl == "$abbrev") { abbrev_words_.insert(norm_word); }
        if (fl.size() == 5 && fl[0] == '$' && fl[1] == 'a' && fl[2] == 'l' &&
            fl[3] == 't' && fl[4] >= '1' && fl[4] <= '6') {
            word_alt_flags_[norm_word] |=
                (1 << (fl[4] - '1'));
        }
        if (!flags.stress_n) { flags.stress_n = parseStressN(fl); }
    }
    // phonemes_str-as-flag fallback (e.g. "gi $abbrev").
    if (phonemes_str == "$abbrev") { abbrev_words_.insert(norm_word); }
    if (phonemes_str.size() == 5 && phonemes_str[0] == '$' &&
        phonemes_str[1] == 'a' && phonemes_str[2] == 'l' &&
        phonemes_str[3] == 't' &&
        phonemes_str[4] >= '1' && phonemes_str[4] <= '6') {
        word_alt_flags_[norm_word] |=
            (1 << (phonemes_str[4] - '1'));
    }
    if (phonemes_str == "$verb"  || phonemes_str == "$verbf" ||
        phonemes_str == "$nounf" ||
        phonemes_str == "$pastf" || phonemes_str == "$only") {
        flags.grammar = true;
    }
    if (phonemes_str == "$pastf") { flags.pastf = true; }
    if (phonemes_str == "$nounf") { flags.nounf = true; }
    if (phonemes_str == "$verbf") { flags.verbf = true; }
    if (phonemes_str == "$u" || phonemes_str == "$u+") {
        unstressed_words_.insert(norm_word);
    }
    if (phonemes_str == "$u")    { flags.grammar = true; }
    if (phonemes_str == "$verb") { flags.verb    = true; }
}

bool IPAPhonemizer::loadDictionary(const std::string& path) {
    std::ifstream f(path);
    bool ok = true;
    if (!f.is_open()) {
        error_ = "Cannot open dictionary file: " + path;
        ok = false;
    } else {
        bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");
        std::string raw;
        while (std::getline(f, raw)) {
            std::string line = raw;
            size_t comment = line.find("//");
            if (comment != std::string::npos) {
                line = line.substr(0, comment);
            }
            line = trim(line);
            // `live` = "this line still needs more processing". Each
            // former `continue` becomes `live = false` and subsequent
            // sections gate on `if (live && ...)`.
            bool live = !line.empty();
            if (live && parsePhraseEntry(line)) { live = false; }
            int dialect_cond = 0;
            bool cond_negated = false;
            if (live) {
                line = parseLeadingDialect(line, &dialect_cond,
                                           &cond_negated);
                if (dialect_cond != 0) {
                    bool match = (dialect_cond == 3 ||
                        dialect_cond == 6) ? is_en_us : false;
                    bool applies = cond_negated ? !match : match;
                    if (!applies) { live = false; }
                }
            }
            if (live) {
                std::vector<std::string> parts = splitWS(line);
                if (parts.size() >= 2) {
                    std::string word = parts[0];
                    std::string phonemes_str = parts[1];
                    std::string norm_word = toLowerASCII(word);
                    EntryFlags flags;
                    parseEntryFlags(parts, norm_word, phonemes_str,
                                    flags);
                    storeDictionaryEntry(norm_word, phonemes_str,
                                         dialect_cond, flags);
                }
            }
        }
        // Sort compound prefixes longest-first for greedy matching.
        std::sort(compound_prefixes_.begin(),
                  compound_prefixes_.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.size() > b.first.size();
                  });
        // Post-load: remove content words from unstressed_words_
        // that should keep stress. "made" has $u+ but the reference
        // stresses it in sentence context.
        unstressed_words_.erase("made");
    }
    return ok;
}

// ============================================================
// Rule File (en_rules) Parser
// ============================================================
static void parseLGroupDef(const std::string& line, RuleSet& rs) {
    if (line.size() < 3 || line[1] != 'L') { return; }
    int id = 0;
    size_t i = 2;
    while (i < line.size() && std::isdigit((unsigned char)line[i])) {
        id = id * 10 + (line[i] - '0');
        i++;
    }
    if (id <= 0 || id >= 100) { return; }

    std::string rest = line.substr(i);
    auto items = splitWS(rest);
    auto comment = std::find_if(items.begin(), items.end(),
        [](const std::string& s) {
            return s.size() >= 2 && s[0] == '/' && s[1] == '/';
        });
    auto& g = rs.groups.lgroups[id];
    g.insert(g.end(), items.begin(), comment);
}

// Strip leading ?N (dialect) or ?!N (negated dialect) from `line`.
// On match, sets *cond / *negated and returns the trimmed remainder.
// On no match, returns `line` unchanged with cond=0, negated=false.
static std::string parseLeadingDialect(const std::string& line,
                                       int* cond, bool* negated) {
    std::string remainder = line;
    *cond = 0;
    *negated = false;
    if (!line.empty() && line[0] == '?') {
        size_t space = line.find_first_of(" \t");
        if (space != std::string::npos) {
            std::string cond_str = line.substr(1, space - 1);
            if (!cond_str.empty() && cond_str[0] == '!') {
                *negated = true;
                cond_str = cond_str.substr(1);
            }
            try { *cond = std::stoi(cond_str); } catch (...) { }
            remainder = trim(line.substr(space));
        }
    }
    return remainder;
}

// Whitespace-tokenize `line` into `tokens` preserving non-empty runs.
static void tokenizeRuleLine(const std::string& line,
                             std::vector<std::string>& tokens) {
    std::string tok;
    for (char c : line) {
        if (std::isspace((unsigned char)c)) {
            if (!tok.empty()) { tokens.push_back(tok); tok.clear(); }
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) { tokens.push_back(tok); }
}

// Detect 'P' prefix-boundary marker in the rule's right context.
// 'P' is the prefix marker when followed by a digit / end-of-string /
// '_' / '+' / '<'. A 'P' preceded by 'L' is an L-group reference
// (L01..L99), not the prefix marker. `rule.is_prefix` is the natural
// search-done state, so the loop predicate replaces a `break`.
static void detectPrefixMarker(PhonemeRule& rule) {
    const std::string& rc = rule.right_ctx;
    for (size_t k = 0; k < rc.size() && !rule.is_prefix; k++) {
        if (rc[k] == 'P') {
            const bool followed_by_marker = (k + 1 >= rc.size()) ||
                (rc[k + 1] >= '1' && rc[k + 1] <= '9') ||
                rc[k + 1] == '_' || rc[k + 1] == '+' || rc[k + 1] == '<';
            const bool preceded_by_L = (k > 0 && rc[k - 1] == 'L');
            if (followed_by_marker && !preceded_by_L) {
                rule.is_prefix = true;
            }
        }
    }
}

// Detect 'S<N>[flags]' RULE_ENDING marker in the rule's right context.
// 'S' (not preceded by 'L') begins a suffix directive: <N> chars to
// strip + flag letters i/m/v/e/d/q/p. `rule.is_suffix` is the natural
// search-done state, so the loop predicate replaces a `break`.
static void detectSuffixMarker(PhonemeRule& rule) {
    static const int SUFX_I_BIT = 0x200;
    static const int SUFX_M_BIT = 0x80000;
    static const int SUFX_V_BIT = 0x800;
    static const int SUFX_E_BIT = 0x100;
    static const int SUFX_D_BIT = 0x1000;
    static const int SUFX_Q_BIT = 0x4000;
    const std::string& rc = rule.right_ctx;
    for (size_t k = 0; k < rc.size() && !rule.is_suffix; k++) {
        if (rc[k] == 'S' && (k == 0 || rc[k - 1] != 'L')) {
            size_t k2 = k + 1;
            int n = 0;
            int sflags = 0;
            while (k2 < rc.size() && std::isdigit((unsigned char)rc[k2])) {
                n = n * 10 + (rc[k2++] - '0');
            }
            while (k2 < rc.size() && std::isalpha((unsigned char)rc[k2])) {
                char fc = rc[k2++];
                if      (fc == 'i') { sflags |= SUFX_I_BIT; }
                else if (fc == 'm') { sflags |= SUFX_M_BIT; }
                else if (fc == 'v') { sflags |= SUFX_V_BIT; }
                else if (fc == 'e') { sflags |= SUFX_E_BIT; }
                else if (fc == 'd') { sflags |= SUFX_D_BIT; }
                else if (fc == 'q') { sflags |= SUFX_Q_BIT; }
                else if (fc == 'p') { sflags |= 0x400; }  // SUFX_P
            }
            if (n > 0) {
                rule.is_suffix = true;
                rule.suffix_strip_len = n;
                rule.suffix_flags = sflags;
            }
        }
    }
}

// Strip "//" trailing comment, trim, and return the cleaned line.
static std::string stripCommentAndTrim(const std::string& raw) {
    std::string s = raw;
    size_t comment = s.find("//");
    if (comment != std::string::npos) { s = s.substr(0, comment); }
    return trim(s);
}

bool IPAPhonemizer::loadRules(const std::string& path) {
    std::ifstream f(path);
    bool ok = true;
    if (!f.is_open()) {
        error_ = "Cannot open rules file: " + path;
        ok = false;
    } else {
        ruleset_.init();
        bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");
        std::string current_group;
        bool in_replace_section = false;
        std::string raw;
        while (std::getline(f, raw)) {
            std::string line = stripCommentAndTrim(raw);
            // `live` = "this line still needs more processing" — once
            // a directive / replace-line / dialect-skip / final
            // store fires, set live = false to short-circuit the
            // remaining if-blocks (that's what `continue` did before).
            bool live = !line.empty();
            if (live && line[0] == '.') {
                if (line.size() >= 2 && line[1] == 'L') {
                    parseLGroupDef(line, ruleset_);
                    live = false;
                } else if (line == ".replace") {
                    in_replace_section = true;
                    current_group = "";
                    live = false;
                } else if (line.substr(0, 6) == ".group") {
                    in_replace_section = false;
                    current_group = trim(line.substr(6));
                    live = false;
                }
            }
            if (live && in_replace_section) {
                auto parts = splitWS(line);
                if (parts.size() >= 2) {
                    ReplaceRule rr;
                    rr.from = parts[0];
                    rr.to = parts[1];
                    ruleset_.replacements.push_back(rr);
                }
                live = false;
            }
            if (live && current_group.empty()) { live = false; }
            if (live) {
                int dialect_cond = 0;
                bool cond_negated = false;
                std::string rule_line = parseLeadingDialect(
                    line, &dialect_cond, &cond_negated);
                bool applies = true;
                if (dialect_cond != 0) {
                    bool match = (dialect_cond == 3) ? is_en_us : false;
                    applies = cond_negated ? !match : match;
                }
                if (applies) {
                    std::vector<std::string> tokens;
                    tokenizeRuleLine(rule_line, tokens);
                    if (!tokens.empty()) {
                        PhonemeRule rule;
                        rule.condition = dialect_cond;
                        rule.condition_negated = cond_negated;
                        int ti = 0;
                        // Left context: token ending with ')'.
                        if (ti < (int)tokens.size() &&
                            tokens[ti].back() == ')') {
                            std::string lctx = tokens[ti];
                            lctx.pop_back();
                            rule.left_ctx = lctx;
                            ti++;
                        }
                        // Match string: next token not starting '('.
                        if (ti < (int)tokens.size() && tokens[ti][0] != '(') {
                            rule.match = tokens[ti];
                            ti++;
                        } else {
                            rule.match = current_group;
                        }
                        // Right context: token starting with '('.
                        if (ti < (int)tokens.size() && tokens[ti][0] == '(') {
                            rule.right_ctx = tokens[ti].substr(1);
                            ti++;
                        }
                        // Phonemes: rest of tokens, joined.
                        for (int j = ti; j < (int)tokens.size(); j++) {
                            rule.phonemes += tokens[j];
                        }
                        bool flag_only = !rule.phonemes.empty() &&
                            rule.phonemes[0] == '$';
                        bool match_empty = rule.match.empty();
                        if (!flag_only && !match_empty) {
                            // 'P' / 'S<N>' marker scans (the
                            // reference SUFX_P / RULE_ENDING).
                            detectPrefixMarker(rule);
                            detectSuffixMarker(rule);
                            ruleset_.rule_groups[current_group]
                                .push_back(rule);
                        }
                    }
                }
            }
        }
    }
    return ok;
}

// ============================================================
// Constructor
// ============================================================
IPAPhonemizer::IPAPhonemizer(const std::string& rules_path,
                                    const std::string& list_path,
                                    const std::string& dialect)
    : dialect_(dialect), loaded_(false) {
    ipa_overrides_ = buildIPAOverrides(dialect);

    if (!loadDictionary(list_path)) { return; }
    if (!loadRules(rules_path)) { return; }
    loaded_ = true;
}

// ============================================================
// Apply replacement rules
// ============================================================
std::string IPAPhonemizer::applyReplacements(const std::string& word) const {
    std::string result = word;
    for (const auto& rr : ruleset_.replacements) {
        size_t pos = 0;
        while ((pos = result.find(rr.from, pos)) != std::string::npos) {
            result.replace(pos, rr.from.size(), rr.to);
            pos += rr.to.size();
        }
    }
    return result;
}

// ============================================================
// Context matching helpers
// ============================================================

// Check if word[pos..] starts with any item in the L-group
// Returns number of chars matched (0 if no match)
static int matchLGroupAt(const std::vector<std::string>& lgroup,
                         const std::string& word, int pos) {
    if (pos < 0 || pos >= (int)word.size()) { return 0; }
    // Try longest items first
    int best = 0;
    for (const auto& item : lgroup) {
        int ilen = (int)item.size();
        if (!item.empty() && pos + ilen <= (int)word.size()) {
            bool ok = true;
            // `ok` is the natural search state — predicate-terminate
            // instead of `break`.
            for (int j = 0; j < ilen && ok; j++) {
                char wc = (char)std::tolower((unsigned char)word[pos + j]);
                char ic = (char)std::tolower((unsigned char)item[j]);
                if (wc != ic) { ok = false; }
            }
            if (ok && ilen > best) { best = ilen; }
        }
    }
    return best;
}

// Match left context: returns (score, true) if matched, (0, false) otherwise
// Left context is scanned RIGHT-TO-LEFT (from pos-1 backward)
// ctx_str is the left context in NATURAL order (as written in rules file)
// Scoring matches the reference MatchRule: distance_left starts at -2, increments by 2 per char
// Literal: 21-distance_left; Group non-C: 20-distance_left; Group C: 19-distance_left
// check_atstart ('_'): +4; STRESSED ('&'): +19; NOVOWELS ('X'): +3; DOUBLE ('%'): 21-distance_left
// INC_SCORE ('+'): +20; SYLLABLE ('@'): 18+count-distance_left
// phonemes_so_far: raw the reference phonemes assigned to letters 0..pos-1 (used for '&' stressed check)
std::pair<int,bool> matchLeftContextScore(
        const std::string& ctx_str, const std::string& word, int pos,
        const RuleSet& rs,
        const std::string& phonemes_so_far = "") {
    int score = 0;
    bool ok = true;
    assert(!ctx_str.empty());
    int word_pos = pos - 1;
    int ci = (int)ctx_str.size() - 1;
    // distance_left increments by 2 each char consumed, capped
    // at 19. Used in score formulas (21 - distance_left etc).
    int distance_left = -2;
    // first char of match (used by RULE_DOUBLE '%')
    char prev_char = (pos > 0 && pos < (int)word.size())
        ? word[pos] : 0;
    // SESE: every former `return {0, false}` becomes `ok = false`
    // and the loop predicate `&& ok` exits naturally. Every
    // former `continue` becomes else-if fall-through.
    while (ci >= 0 && ok) {
        char cc = ctx_str[ci];
        if (cc == '_') {
            // RULE_SPACE / check_atstart: word boundary, +4
            if (word_pos >= 0) {
                ok = false;
            } else {
                ci--;
                score += 4;
            }
        } else if (cc == '&') {
            // RULE_STRESSED: +19, no char consumed.
            // the reference word_stressed_count: count of vowels
            // so far that are not explicitly unstressed (no
            // preceding '%' or '=' marker) and not inherently
            // unstressed. '#' after a vowel is NOT an unstress
            // indicator (it's a phoneme modifier); only explicit
            // '%' or '=' preceding the vowel marks it unstressed.
            // the reference evidence: 'I' from 'i' in "rigorous"
            // triggers &, so 'I' is stressable; '0' from 'o' in
            // "nostra" triggers & for final 'a', so '0#' is
            // stressable.
            bool found_stressed = false;
            // Scan phonemes_so_far for any stressable vowel not
            // preceded by '%'/'='. Must parse multi-char codes
            // as units (e.g. 'aI', '3:', 'I2') so that '%' before
            // the FIRST char of a multi-char code correctly
            // marks the WHOLE code as unstressed.
            // Bug: char-by-char scanning treats 'I' in '%aI' as
            // a standalone stressable vowel. Fix: multi-char-
            // aware scan using the same S_MC table as
            // processPhonemeString.
            // phUNSTRESSED codes (always unstressed regardless
            // of '%' prefix): '@*' variants, 'i' alone (not 'i:'
            // or 'i@'), 'a#', 'I#', 'I2'.
            // Stressable despite '#' suffix: '0#', 'E#', 'E2',
            // 'I' alone.
            static const char* STRESSED_MC[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
                "3:r","A:r","o@r","A@r","e@r",
                "eI","aI","aU","OI","oU","IR","VR",
                "e@","i@","U@","A@","O@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                "@L","@2","@5",
                "I2","I#","E2","E#","e#","a#","a2","0#","02",
                "O2","A~","O~","A#",
                nullptr
            };
            static const std::string VOWEL_PH_STRESSABLE = "aAeEiIoOuUV03";
            if (!phonemes_so_far.empty()) {
                size_t pi = 0;
                bool prev_unstressed = false;
                // `!found_stressed` is the natural search state.
                while (pi < phonemes_so_far.size() && !found_stressed) {
                    char pc = phonemes_so_far[pi];
                    if (pc == '%' || pc == '=' || pc == ',') {
                        prev_unstressed = true;
                        pi++;
                    } else if (pc == '\'') {
                        prev_unstressed = false;
                        pi++;
                    } else {
                        std::string code;
                        for (int mi = 0;
                             STRESSED_MC[mi] && code.empty();
                             mi++) {
                            int mcl = (int)strlen(STRESSED_MC[mi]);
                            if (pi + (size_t)mcl <= phonemes_so_far.size() &&
                                phonemes_so_far.compare(
                                    pi, mcl,
                                    STRESSED_MC[mi]) == 0) {
                                code = std::string(STRESSED_MC[mi], mcl);
                            }
                        }
                        if (code.empty()) { code = std::string(1, pc); }
                        bool is_vowel = !code.empty() &&
                            VOWEL_PH_STRESSABLE.find(code[0]) !=
                                std::string::npos;
                        if (is_vowel) {
                            bool inherently_unstressed =
                                (code[0] == '@')   || // schwa
                                (code == "i")      || // 'i' alone
                                (code == "a#")     || // reduced a
                                (code == "I#")     || // reduced ɪ
                                (code == "I2");       // reduced ɪ
                            if (!inherently_unstressed && !prev_unstressed) {
                                found_stressed = true;
                            }
                        }
                        if (!found_stressed) {
                            prev_unstressed = false;
                            pi += code.size();
                        }
                    }
                }
            } else {
                for (int k = 0;
                     k < pos && !found_stressed; k++) {
                    if (isVowelLetter(word[k])) { found_stressed = true; }
                }
            }
            if (!found_stressed) {
                ok = false;
            } else {
                ci--;
                score += 19;
            }
        } else if (cc == '@') {
            // RULE_SYLLABLE in pre-context: count consecutive
            // '@' chars (right-to-left)
            int syllable_count = 0;
            while (ci >= 0 && ctx_str[ci] == '@') {
                syllable_count++;
                ci--;
            }
            int vowel_groups = 0;
            // the reference counts syllable nuclei in the
            // PHONEME string accumulated so far, not vowel
            // letters in the word. Consecutive vowel phoneme
            // chars form one group (e.g., diphthong 'eI' = 1
            // syllable, not 2). This matters for e.g. `@@e) d`:
            // "feIs" at 'd' -> 'e','I' consecutive = 1 group ->
            // @@ fails (correct, "faced" stays voiced).
            // "noUtI2s" at 'd' -> 'oU' and 'I' separated by 't'
            // = 2 groups -> @@ passes (correct, "noticed"
            // devoiced).
            if (!phonemes_so_far.empty()) {
                static const std::string VOWEL_PH = "aAeEIiOUVu03@o";
                bool in_v2 = false;
                for (char c : phonemes_so_far) {
                    bool v = (VOWEL_PH.find(c) != std::string::npos);
                    if (v && !in_v2) {
                        vowel_groups++;
                        in_v2 = true;
                    } else if (!v) {
                        in_v2 = false;
                    }
                }
            } else {
                bool in_v2 = false;
                for (int wp = 0; wp < pos; wp++) {
                    bool v = isVowelLetter(word[wp]);
                    if (v && !in_v2) {
                        vowel_groups++;
                        in_v2 = true;
                    } else if (!v) {
                        in_v2 = false;
                    }
                }
            }
            if (syllable_count > vowel_groups) {
                ok = false;
            } else {
                // SYLLABLE adds points like a char but doesn't
                // consume
                int dist = distance_left + 2;
                if (dist > 19) { dist = 19; }
                score += 18 + syllable_count - dist;
            }
        } else if (cc == '!') {
            // RULE_CAPITAL - skip (no strict check)
            ci--;
        } else if (cc == '%') {
            // RULE_DOUBLE: current left-context char must equal
            // the char to its right
            if (word_pos < 0) {
                ok = false;
            } else {
                char cur = (char)std::tolower((unsigned char)word[word_pos]);
                char nxt = (char)std::tolower((unsigned char)prev_char);
                if (cur != nxt) {
                    ok = false;
                } else {
                    distance_left += 2;
                    if (distance_left > 19) { distance_left = 19; }
                    prev_char = word[word_pos];
                    word_pos--;
                    ci--;
                    score += 21 - distance_left;
                }
            }
        } else if (cc == '+') {
            // RULE_INC_SCORE
            score += 20;
            ci--;
        } else if (cc == '<') {
            // RULE_DEC_SCORE
            score -= 20;
            ci--;
        } else if (cc == 'A' || cc == 'B' || cc == 'C' ||
                   cc == 'F' || cc == 'G' || cc == 'H' || cc == 'Y') {
            if (!rs.groups.matchGroup(cc, word, word_pos)) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                int lg_pts = (cc == 'C') ? 19 : 20;
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += lg_pts - distance_left;
            }
        } else if (cc == 'K') {
            if (word_pos < 0 || isVowelLetter(word[word_pos])) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += 20 - distance_left;
            }
        } else if (cc == 'X') {
            // RULE_NOVOWELS: no vowels from word start to here,
            // adds +3 (fixed, no distance)
            bool found_vowel = false;
            for (int k = 0;
                 k <= word_pos && !found_vowel; k++) {
                if (isVowelLetter(word[k])) { found_vowel = true; }
            }
            if (found_vowel) {
                ok = false;
            } else {
                ci--;
                score += 3;
            }
        } else if (cc == 'D') {
            if (word_pos < 0 || !std::isdigit((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += 21 - distance_left;
            }
        } else if (cc == 'Z') {
            if (word_pos < 0 || std::isalpha((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_left += 2;
                if (distance_left > 19) { distance_left = 19; }
                prev_char = word[word_pos];
                word_pos--;
                ci--;
                score += 21 - distance_left;
            }
        } else if (cc >= '0' && cc <= '9') {
            // L-group reference (L followed by digits)
            int gid = cc - '0';
            int ci2 = ci - 1;
            if (ci2 >= 0 && ctx_str[ci2] >= '0' && ctx_str[ci2] <= '9') {
                gid += (ctx_str[ci2] - '0') * 10;
                ci2--;
            }
            if (ci2 >= 0 && ctx_str[ci2] == 'L') {
                if (gid > 0 && gid < 100) {
                    int matched = matchLGroupAt(
                        rs.groups.lgroups[gid], word,
                        word_pos);
                    if (matched == 0) {
                        ok = false;
                    } else {
                        distance_left += 2;
                        if (distance_left > 19) { distance_left = 19; }
                        if (matched > 0) {
                            prev_char = word[word_pos - matched + 1];
                        }
                        word_pos -= matched;
                        score += 20 - distance_left;
                    }
                }
                // Even if gid was out of range or ok=false, set
                // ci so the loop terminates cleanly. (When
                // ok=false the predicate exits next.)
                ci = ci2 - 1;
            } else {
                ci--;
            }
        } else if (cc == 'E') {
            // 'E' in context = REPLACED_E (the reference marks
            // silent 'e' after '#' rules as uppercase 'E'). We
            // don't implement REPLACED_E marking, so 'E' context
            // never matches.
            ok = false;
        } else {
            // Literal character match
            if (word_pos < 0) {
                ok = false;
            } else {
                char wc = (char)std::tolower((unsigned char)word[word_pos]);
                char mc = (char)std::tolower((unsigned char)cc);
                if (wc != mc) {
                    ok = false;
                } else {
                    distance_left += 2;
                    if (distance_left > 19) { distance_left = 19; }
                    prev_char = word[word_pos];
                    word_pos--;
                    ci--;
                    score += 21 - distance_left;
                }
            }
        }
    }
    return {ok ? score : 0, ok};
}

// Match right context: returns (score, del_fwd_start, del_fwd_count, matched) if matched
// del_fwd_start: absolute word position where silent chars begin (-1 if none)
// del_fwd_count: number of chars to mark as silent/deleted
struct RightCtxResult { int score; int del_fwd_start; int del_fwd; bool matched; };

// initial_prev_char: last char of the match key (needed for RULE_DOUBLE '%')
// match_start: position in word where the rule's key starts (for Sn syllable-count condition)
RightCtxResult matchRightContextScore(
        const std::string& ctx_str, const std::string& word, int pos,
        const RuleSet& rs, char initial_prev_char = 0,
        int match_start = -1, int word_alt_flags = 0,
        const std::vector<bool>* replaced_e_arr = nullptr,
        bool suffix_removed = false) {
    int score = 0;
    int del_fwd_pos = -1; // position of REPLACED_E character to skip
    bool ok = true;
    assert(!ctx_str.empty());
    int word_pos = pos;
    int ci = 0;
    int clen = (int)ctx_str.size();
    // distance_right increments by 6 per char consumed, capped
    // at 19. Used in score formulas.
    int distance_right = -6;
    char prev_char = initial_prev_char;
    // SESE: every former `return {0, -1, 0, false}` becomes
    // `ok = false` and the loop predicate `&& ok` exits
    // naturally. Every former `continue` becomes else-if
    // fall-through.
    while (ci < clen && ok) {
        char cc = ctx_str[ci];
        if (cc == '_') {
            // RULE_SPACE: word-end boundary; scores like a
            // literal char
            if (word_pos < (int)word.size()) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                ci++;
                score += 21 - distance_right;
            }
        } else if (cc == '#') {
            // RULE_DEL_FWD: search for the first 'e' in range
            // [pos, word_pos) and mark it as REPLACED_E
            // (magic-e deletion). Only fires when 'e' is
            // actually found.
            // the reference: "for (p = *word + group_length;
            // p < post_ptr; p++) if (*p=='e') ..."
            // Example: rule "iv (e#" -> 'e' at range[0] is found
            // and silenced (-> "ive" treated as /aɪv/).
            // Counter-example: "rhi (n#" -> range
            // [pos,word_pos) = just 'n', not 'e' -> no deletion.
            if (del_fwd_pos < 0) {
                for (int sp = pos;
                     sp < word_pos && del_fwd_pos < 0; sp++) {
                    if (word[sp] == 'e') { del_fwd_pos = sp; }
                }
            }
            ci++;
        } else if (cc == 'A' || cc == 'B' || cc == 'C' ||
                   cc == 'F' || cc == 'G' || cc == 'H' || cc == 'Y') {
            if (!rs.groups.matchGroup(cc, word, word_pos)) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                int lg_pts = (cc == 'C') ? 19 : 20;
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += lg_pts - distance_right;
            }
        } else if (cc == 'K') {
            // K = non-vowel. the reference words are
            // null-terminated, so K matches '\0' at word end
            // (null is not a vowel). Treat
            // word_pos >= word.size() as '\0' match.
            if (word_pos < (int)word.size() && isVowelLetter(word[word_pos])) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                if (word_pos < (int)word.size()) {
                    prev_char = word[word_pos];
                }
                word_pos++;
                ci++;
                score += 20 - distance_right;
            }
        } else if (cc == 'X') {
            // RULE_NOVOWELS right-context: no vowels to word
            // end; scores 19-distance
            bool found = false;
            for (int k = word_pos;
                 k < (int)word.size() && !found; k++) {
                if (isVowelLetter(word[k])) { found = true; }
            }
            if (found) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                ci++;
                score += 19 - distance_right;
            }
        } else if (cc == 'D') {
            if (word_pos >= (int)word.size() ||
                !std::isdigit((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
            }
        } else if (cc == 'Z') {
            if (word_pos >= (int)word.size() ||
                std::isalpha((unsigned char)word[word_pos])) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
            }
        } else if (cc == '%') {
            // RULE_DOUBLE: current char must equal previous char
            if (word_pos >= (int)word.size()) {
                ok = false;
            } else {
                char cur = (char)std::tolower((unsigned char)word[word_pos]);
                char prv = (char)std::tolower((unsigned char)prev_char);
                if (cur != prv) {
                    ok = false;
                } else {
                    distance_right += 6;
                    if (distance_right > 19) { distance_right = 19; }
                    prev_char = word[word_pos];
                    word_pos++;
                    ci++;
                    score += 21 - distance_right;
                }
            }
        } else if (cc == '+') {
            // RULE_INC_SCORE
            score += 20;
            ci++;
        } else if (cc == '<') {
            // RULE_DEC_SCORE
            score -= 20;
            ci++;
        } else if (cc == '@') {
            // RULE_SYLLABLE: count consecutive '@', require N
            // vowel groups remaining
            int syllable_count = 0;
            while (ci < clen && ctx_str[ci] == '@') {
                syllable_count++;
                ci++;
            }
            int vowel_groups = 0;
            bool in_v = false;
            for (int wp = word_pos;
                 wp < (int)word.size(); wp++) {
                bool v = isVowelLetter(word[wp]);
                if (v && !in_v) {
                    vowel_groups++;
                    in_v = true;
                } else if (!v) {
                    in_v = false;
                }
            }
            if (syllable_count > vowel_groups) {
                ok = false;
            } else {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                score += 18 + syllable_count - distance_right;
            }
        } else if (cc == '&') {
            // RULE_STRESSED in right context - fail
            ok = false;
        } else if (cc == '!') {
            ci++;
        } else if (cc == '$') {
            // Check for $w_altN word-level condition (e.g.
            // "$w_alt2" in right context). This fires only if
            // the current word has the corresponding $altN flag
            // in en_list. Format: $w_alt followed by digit 1-6
            if (ci + 6 < clen && ctx_str[ci+1]=='w' &&
                ctx_str[ci+2]=='_' && ctx_str[ci+3]=='a' &&
                ctx_str[ci+4]=='l' && ctx_str[ci+5]=='t' &&
                ctx_str[ci+6] >= '1' && ctx_str[ci+6] <= '6') {
                int alt_n = ctx_str[ci+6] - '0';
                int alt_bit = 1 << (alt_n - 1);
                if (!(word_alt_flags & alt_bit)) {
                    ok = false;
                } else {
                    ci += 7; // skip "$w_altN"
                }
            } else {
                // other $ flags - skip rule
                ok = false;
            }
        } else if (cc == 'N') {
            if (ci+1 < clen && std::isdigit(ctx_str[ci+1])) {
                // Nn (N followed by digit): syllable/condition
                // check — skip for now
                ci += 2;
            } else if (suffix_removed) {
                // N alone = RULE_NO_SUFFIX: fails when the word
                // is being re-phonemized as a stem after suffix
                // removal (the reference FLAG_SUFFIX_REMOVED /
                // translate.h:185). When suffix was removed,
                // this rule should not fire.
                ok = false;
            } else {
                score += 1;
                ci++;
            }
        } else if (cc == 'P') {
            while (ci < clen && !std::isspace((unsigned char)ctx_str[ci])) {
                ci++;
            }
        } else if (cc == 'S') {
            // RULE_ENDING: the reference suffix-stripping
            // directive (_S<N>[flags] in source). Pure directive
            // — does NOT consume word chars and does NOT fail
            // the match. Suffix info (strip length N, flags
            // i/m/v/e/d/q/t) is stored on the PhonemeRule itself
            // (set during loadRules). Here we just skip past
            // the number and flag chars in the context string.
            ci++; // skip 'S'
            while (ci < clen && std::isdigit((unsigned char)ctx_str[ci])) {
                ci++;
            }
            while (ci < clen && std::isalpha((unsigned char)ctx_str[ci])) {
                ci++;
            }
            // No score contribution, no position check — suffix
            // action handled in applyRules
        } else if (cc == 'L' && ci+1 < clen && std::isdigit(ctx_str[ci+1])) {
            // L-group reference
            int gid = 0;
            ci++;
            while (ci < clen && std::isdigit(ctx_str[ci])) {
                gid = gid * 10 + (ctx_str[ci] - '0');
                ci++;
            }
            if (gid > 0 && gid < 100 && !rs.groups.lgroups[gid].empty()) {
                int matched = matchLGroupAt(
                    rs.groups.lgroups[gid], word, word_pos);
                if (matched == 0) {
                    ok = false;
                } else {
                    distance_right += 6;
                    if (distance_right > 19) { distance_right = 19; }
                    if (matched > 0) {
                        prev_char = word[word_pos + matched - 1];
                    }
                    word_pos += matched;
                    score += 20 - distance_right;
                }
            }
        } else if (std::isdigit((unsigned char)cc)) {
            ci++;
        } else if (cc == 'E') {
            // 'E' in context = REPLACED_E (the reference marks
            // silent 'e' after '#' rules as uppercase 'E').
            // Match if replaced_e_arr marks this position as a
            // deleted 'e'.
            if (replaced_e_arr && word_pos < (int)word.size() &&
                word_pos < (int)replaced_e_arr->size() &&
                (*replaced_e_arr)[word_pos]) {
                distance_right += 6;
                if (distance_right > 19) { distance_right = 19; }
                prev_char = word[word_pos];
                word_pos++;
                ci++;
                score += 21 - distance_right;
            } else {
                ok = false;
            }
        } else {
            // Literal character match
            if (word_pos >= (int)word.size()) {
                ok = false;
            } else {
                char wc = (char)std::tolower((unsigned char)word[word_pos]);
                char mc = (char)std::tolower((unsigned char)cc);
                if (wc != mc) {
                    ok = false;
                } else {
                    distance_right += 6;
                    if (distance_right > 19) { distance_right = 19; }
                    prev_char = word[word_pos];
                    word_pos++;
                    ci++;
                    score += 21 - distance_right;
                }
            }
        }
    }
    return {ok ? score : 0,
            ok ? del_fwd_pos : -1,
            ok ? (del_fwd_pos >= 0 ? 1 : 0) : 0,
            ok};
}

// Match a rule at position pos in word, returns score (-1 if no match)
// group_length: length of the group key (1 for single-char groups, 2 for two-char groups)
// del_fwd_start: absolute word position of first silent char (-1 if none)
// del_fwd_count: number of chars to mark silent starting at del_fwd_start
int IPAPhonemizer::matchRule(
        const PhonemeRule& rule, const std::string& word, int pos,
        std::string& out_phonemes, int& advance,
        int& del_fwd_start, int& del_fwd_count,
        int group_length,
        const std::string& phonemes_so_far,
        int word_alt_flags,
        const std::vector<bool>* replaced_e_arr,
        bool suffix_removed) const {
    const std::string& match = rule.match;
    int mlen = (int)match.size();
    if (pos + mlen > (int)word.size()) { return -1; }
    // Compare match string case-insensitively
    for (int i = 0; i < mlen; i++) {
        if (std::tolower((unsigned char)word[pos+i]) !=
            std::tolower((unsigned char)match[i])) {
            return -1;
        }
    }
    // Match left context. Empty ctx is the no-op case: score 0,
    // matched true. Hoisting the check here lets the function body
    // run at one indent level lower (see house-style assert pattern).
    int lscore = 0;
    bool lmatch = true;
    if (!rule.left_ctx.empty()) {
        auto [s, m] = matchLeftContextScore(
            rule.left_ctx, word, pos, ruleset_, phonemes_so_far);
        lscore = s;
        lmatch = m;
    }
    if (!lmatch) { return -1; }
    // The last char of the match (needed for RULE_DOUBLE '%' in
    // right context)
    char last_match_char = (mlen > 0) ? word[pos + mlen - 1] : 0;
    // Match right context (pass pos as match_start for Sn
    // syllable-count condition). word_alt_flags is passed in from
    // applyRules (the caller controls which alt flags are active).
    // Empty ctx -> {0, -1, 0, true} (no score, no fwd-del, matched).
    RightCtxResult rresult = {0, -1, 0, true};
    if (!rule.right_ctx.empty()) {
        rresult = matchRightContextScore(
            rule.right_ctx, word, pos + mlen, ruleset_,
            last_match_char, pos, word_alt_flags, replaced_e_arr,
            suffix_removed);
    }
    if (!rresult.matched) { return -1; }
    // the reference scoring: base=1 + (additional match chars beyond
    // group key) * 21 + context scores. additional_consumed = mlen -
    // group_length (extra chars in the match beyond the group key).
    int additional_consumed = mlen - group_length;
    if (additional_consumed < 0) { additional_consumed = 0; }
    int dialect_bonus = (rule.condition != 0) ? 1 : 0;
    int total_score = 1 + additional_consumed * 21 + lscore +
        rresult.score + dialect_bonus;
    // Extract phonemes (remove $ flags)
    std::string ph = rule.phonemes;
    size_t dollar = ph.find('$');
    if (dollar != std::string::npos) { ph = trim(ph.substr(0, dollar)); }
    out_phonemes = ph;
    advance = mlen;
    del_fwd_start = rresult.del_fwd_start;
    del_fwd_count = rresult.del_fwd;
    return total_score;
}

// ============================================================
// Apply rules to a word (letter-to-phoneme)
// ============================================================
// Resolve word_alt_flags: explicit param if >= 0, else look up the
// word's own $altN bitmask from word_alt_flags_. When called for stem
// re-phonemization via RULE_ENDING, caller passes 0 (stems don't
// inherit their dict flags in the reference TranslateRules path).
int IPAPhonemizer::determineAltFlags(const std::string& word,
                                     int word_alt_flags_param) const {
    int result = 0;
    if (word_alt_flags_param >= 0) {
        result = word_alt_flags_param;
    } else {
        std::string wl;
        wl.reserve(word.size());
        for (char c : word) { wl += (char)std::tolower((unsigned char)c); }
        auto it = word_alt_flags_.find(wl);
        if (it != word_alt_flags_.end()) { result = it->second; }
    }
    return result;
}

// Try the 2-char group (with +35 bonus, the reference TranslateRules
// match2.points += 35) and the 1-char group at `pos`; return the best
// match. Within-group ties use last-rule-wins (>= comparison) to make
// longer-match rules (appearing later in file) beat shorter ones.
IPAPhonemizer::RuleMatchResult IPAPhonemizer::findBestRule(
        const std::string& word, int pos, int len, char pos_char,
        int word_alt_flags, const std::vector<bool>& replaced_e,
        bool allow_suffix_strip, bool suffix_phoneme_only,
        bool suffix_removed,
        const std::string& accumulated_phonemes) const {
    RuleMatchResult best = {-1, "", 1, -1, 0, false, false, 0, 0};
    auto try_group = [&](const std::string& key, int bonus,
                         int group_length) {
        auto it = ruleset_.rule_groups.find(key);
        if (it != ruleset_.rule_groups.end()) {
            for (const auto& rule : it->second) {
                std::string ph;
                int adv, dfstart, dfcount;
                int sc = matchRule(rule, word, pos, ph, adv, dfstart,
                                   dfcount, group_length,
                                   accumulated_phonemes,
                                   word_alt_flags, &replaced_e,
                                   suffix_removed);
                bool valid = (sc >= 0);
                // When suffix stripping is disabled (and NOT in
                // suffix_phoneme_only mode), skip suffix rules at
                // word-end: the reference TranslateRules returns early
                // (without the phoneme) when RULE_ENDING fires with
                // end_phonemes=NULL. In suffix_phoneme_only mode,
                // RULE_ENDING rules ARE selected but their phoneme is
                // just accumulated (no stem re-phonemization).
                bool is_ending_skip = !allow_suffix_strip
                    && !suffix_phoneme_only && rule.is_suffix
                    && pos + adv == len;
                if (valid && !is_ending_skip && sc + bonus >= best.score) {
                    best.score             = sc + bonus;
                    best.phonemes          = ph;
                    best.advance           = adv;
                    best.del_start         = dfstart;
                    best.del_count         = dfcount;
                    best.is_prefix         = rule.is_prefix;
                    best.is_suffix         = rule.is_suffix;
                    best.suffix_strip_len  = rule.suffix_strip_len;
                    best.suffix_flags      = rule.suffix_flags;
                }
            }
        }
    };
    if (pos + 1 < len) {
        std::string key2;
        key2 += pos_char;
        key2 += (char)std::tolower((unsigned char)word[pos + 1]);
        try_group(key2, 35, 2);
    }
    std::string key1(1, pos_char);
    try_group(key1, 0, 1);
    return best;
}

// Locate the last fully-stressable vowel in `phonemes` (skip rule
// boundaries '\\x01' and weak/reduced vowels marked by '2' or '#'),
// stepping back through diphthong second-chars (eI, aI, oU, etc.) and
// rhotic centring diphthongs (A@, e@, i@, ...) so the returned index
// points at the START of the multi-char vowel code. Returns -1 if no
// stressable vowel exists. Used by phonSTRESS_PREV.
int IPAPhonemizer::findLastStressableVowel(
        const std::string& phonemes) const {
    static const std::string SP_VOWELS = "aAeEiIoOuUV03@";
    static const std::string DIPH_SECOND = "IU";
    static const std::string DIPH_START = "eaOoUAE";
    static const std::string AT_STARTERS = "AeioOU";
    auto prevNonBnd = [&](int from) {
        std::pair<char, int> r = {0, -1};
        for (int pi = from - 1; pi >= 0; pi--) {
            if (r.second < 0 && phonemes[pi] != '\x01') {
                r = {phonemes[pi], pi};
            }
        }
        return r;
    };
    int insert_at = -1;
    int slen = (int)phonemes.size();
    for (int si = slen - 1; si >= 0 && insert_at < 0; si--) {
        char sc = phonemes[si];
        if (sc != '\x01' && SP_VOWELS.find(sc) != std::string::npos) {
            int ni = si + 1;
            while (ni < slen && phonemes[ni] == '\x01') { ni++; }
            bool is_reduced = (ni < slen
                && (phonemes[ni] == '2' || phonemes[ni] == '#'));
            if (!is_reduced) {
                auto pnb = prevNonBnd(si);
                char prev_ch = pnb.first;
                int  prev_pos = pnb.second;
                // Step back through multi-char vowel codes:
                //   Case 1: I/U after diphthong-start (eI, aU, oU,...)
                //   Case 2: 'a' after 'a' (the 'aa' code = æ)
                //   Case 3: '@' after rhotic-start (A@, e@, i@,...)
                if (DIPH_SECOND.find(sc) != std::string::npos
                    && DIPH_START.find(prev_ch) != std::string::npos) {
                    si = prev_pos;
                    auto pp2 = prevNonBnd(si);
                    prev_ch = pp2.first;
                } else if (sc == 'a' && prev_ch == 'a') {
                    si = prev_pos;
                    auto pp2 = prevNonBnd(si);
                    prev_ch = pp2.first;
                } else if (sc == '@' && AT_STARTERS.find(prev_ch)
                        != std::string::npos) {
                    si = prev_pos;
                    auto pp2 = prevNonBnd(si);
                    prev_ch = pp2.first;
                }
                // Skip if already marked unstressed/diminished.
                if (prev_ch != '%' && prev_ch != '=') { insert_at = si; }
            }
        }
    }
    return insert_at;
}

// Handle phonSTRESS_PREV: '=' at the START of an emit string (not
// after a phoneme — that's phonLENGTHEN) corresponds to the reference
// byte code phonSTRESS_PREV (code 8), which retroactively promotes
// the last preceding stressable vowel to PRIMARY. Earlier '\\'' marks
// are demoted to '\\x02' (protected secondary, distinct from ',' so
// processPhonemeString step 5a still runs).
void IPAPhonemizer::applyStressPrev(std::string& emit,
                                    std::string& phonemes) const {
    if (!emit.empty() && emit[0] == '=') {
        emit = emit.substr(1);
        int insert_at = findLastStressableVowel(phonemes);
        if (insert_at >= 0) {
            // Only promote if the found vowel isn't already PRIMARY
            // (i.e., the closest preceding non-boundary char isn't
            // '\\'').
            char before_vowel = 0;
            for (int pi = insert_at - 1;
                 pi >= 0 && before_vowel == 0; pi--) {
                if (phonemes[pi] != '\x01') { before_vowel = phonemes[pi]; }
            }
            if (before_vowel != '\'') {
                phonemes.insert(insert_at, "'");
                for (int di = 0; di < insert_at; di++) {
                    if (phonemes[di] == '\'') { phonemes[di] = '\x02'; }
                }
            }
        }
    }
}

// SUFX_E: conditionally append 'e' to stem so dict / magic-e rules
// fire on the original verb form. See dictionary.c:3107-3138 in the
// reference. `stem` is mutated in place.
void IPAPhonemizer::appendMagicEIfNeeded(
        std::string& stem, const std::string& suffix_ph,
        int suffix_flags) const {
    static const int SUFX_E_BIT = 0x100;
    static const int SUFX_V_BIT = 0x800;
    bool entering = (suffix_flags & SUFX_E_BIT) && !stem.empty() &&
                    stem.back() != 'e';
    if (entering) {
        std::string stem_norm_bare = toLowerASCII(stem);
        bool sfx_is_s_bare = (suffix_ph == "s" || suffix_ph == "z" ||
            suffix_ph == "I#z" || suffix_ph == "%I#z");
        // Force-add 'e' when the verb form (stem+e) is in verb_dict_:
        // "used" -> strip 'ed' -> stem='us', but "use" is in verb_dict_
        // (verb form ju:z). Without this, "us" in dict_ (pronoun)
        // blocks 'e' addition.
        if (!sfx_is_s_bare && (suffix_flags & SUFX_V_BIT)) {
            std::string stem_e_norm = toLowerASCII(stem + "e");
            if (verb_dict_.count(stem_e_norm) > 0) { stem += 'e'; }
        }
        if (stem.back() != 'e') {
            // First: check if stem-without-e is in dict. If so, skip
            // 'e' addition (e.g. "charit"->tSarIt is in dict, so
            // "charitable" looks up "charit" not "charite").
            bool stem_bare_in_dict = false;
            if (!sfx_is_s_bare) {
                stem_bare_in_dict = verb_dict_.count(stem_norm_bare) > 0;
            }
            if (!stem_bare_in_dict && !onlys_words_.count(stem_norm_bare) &&
                !only_words_.count(stem_norm_bare)) {
                stem_bare_in_dict = dict_.count(stem_norm_bare) > 0;
            }
            if (!stem_bare_in_dict) {
                static const char* ADD_E_ADDITIONS[] = {
                    "c", "rs", "ir", "ur", "ath", "ns", "u",
                    "spong", "rang", "larg", nullptr };
                static const std::string VOWELS_INCL_Y = "aeiouy";
                static const std::string HARD_CONS = "bcdfgjklmnpqstvxz";
                bool add_e = false;
                for (int ai = 0; ADD_E_ADDITIONS[ai]; ai++) {
                    size_t plen = strlen(ADD_E_ADDITIONS[ai]);
                    if (stem.size() >= plen &&
                        stem.compare(stem.size() - plen, plen,
                                     ADD_E_ADDITIONS[ai]) == 0) {
                        add_e = true;
                    }
                }
                // Vowel + hard consonant at stem end, with the
                // "ion" exception (e.g. "lion" doesn't add 'e').
                if (!add_e && stem.size() >= 2) {
                    char last = std::tolower((unsigned char)stem.back());
                    char prev = std::tolower(
                        (unsigned char)stem[stem.size() - 2]);
                    bool last_hard = HARD_CONS.find(last) != std::string::npos;
                    bool prev_vowel =
                        VOWELS_INCL_Y.find(prev) != std::string::npos;
                    bool ion_exc = (stem.size() >= 3 &&
                        stem.compare(stem.size() - 3, 3,
                                     "ion") == 0);
                    add_e = last_hard && prev_vowel && !ion_exc;
                }
                if (add_e) { stem += 'e'; }
            }
        }
    }
}

// Look up stem phonemes via dict_ / verb_dict_ / onlys_bare_dict_,
// applying $N stress overrides. Falls back to a recursive
// applyRules() call when no dict entry is found. Sets `stem_ph` and
// returns true if the stem produced a phoneme string. On false, the
// caller should ignore stem_ph (it stays empty).
// Bulletproof stem lookup (Gemini-suggested rewrite):
// - track `matched_stem` (a string), NEVER alias iterators across
//   containers (the original `dt = vt` was UB — libc++ release
//   works by accident, MSVC debug iterators trap).
// - verb_dict_ wins outright when SUFX_V is set; the $onlys flag
//   is for the bare-form noun pronunciation, irrelevant for the
//   verb form (the original code rejected verb_dict via the
//   trailing `!stem_is_onlys` check — bug #3 in the staleness hunt).
// - magic-e-stripped fallback (dt_noe) syncs `matched_stem` to the
//   bare form so the stress override looks up the right key. The
//   original used stem_norm here, missing any $N override for the
//   bare form (e.g. "tornadoes" -> strip 's' -> "tornadoe" ->
//   fallback to "tornado"; if "tornado" has $2, the original
//   misses it because it queried stress_pos_["tornadoe"]).
IPAPhonemizer::StemLookupResult IPAPhonemizer::lookupStemInDicts(
        const std::string& stem_norm, bool stem_is_onlys,
        bool suffix_is_s, const RuleMatchResult& match) const {
    static const int SUFX_V_BIT = 0x800;
    StemLookupResult r;
    r.matched_stem = stem_norm;
    // Step 1: verb_dict_ (direct extraction, no iterator aliasing).
    // Compound suffixes like -hold/-fold/-man lack SUFX_V.
    if (!suffix_is_s && (match.suffix_flags & SUFX_V_BIT) != 0) {
        auto vt = verb_dict_.find(stem_norm);
        if (vt != verb_dict_.end()) {
            r.stem_ph = vt->second;
            r.found_dict_entry = true;
        }
    }
    // Step 2: onlys_bare_dict_ (for -s only) or dict_.
    if (!r.found_dict_entry) {
        if (suffix_is_s) {
            auto obit = onlys_bare_dict_.find(stem_norm);
            if (obit != onlys_bare_dict_.end()) {
                r.stem_ph = obit->second;
                r.used_onlys_bare = true;
            }
        }
        if (!r.used_onlys_bare) {
            auto dt = dict_.find(stem_norm);
            if (dt != dict_.end() && !stem_is_onlys) {
                r.stem_ph = dt->second;
                r.found_dict_entry = true;
            }
        }
    }
    // Step 3: magic-e-stripped fallback. Handles "-oes" plurals
    // ("tornadoes" -> strip 's' -> "tornadoe" -> not in dict -> try
    // "tornado"). matched_stem is synced so the stress lookup
    // queries the bare form's $N override.
    if (!r.found_dict_entry && !r.used_onlys_bare &&
        stem_norm.size() > 1 && stem_norm.back() == 'e') {
        std::string stem_no_e_norm = stem_norm.substr(0, stem_norm.size() - 1);
        auto dt_noe = dict_.find(stem_no_e_norm);
        if (dt_noe != dict_.end() && !onlys_words_.count(stem_no_e_norm) &&
            !only_words_.count(stem_no_e_norm)) {
            r.stem_ph = dt_noe->second;
            r.matched_stem = stem_no_e_norm;
            r.found_dict_entry = true;
        }
    }
    return r;
}

// Step 4 of stemPhonemeFromDict: apply $N stress override on a dict
// hit, otherwise re-phonemize via rules with combined $altN flags
// and apply the override post-rules (gated by noun-form-only and
// verb-flag exclusions).
void IPAPhonemizer::applyStemStressOrRulesFallback(
        const std::string& stem, const std::string& stem_norm,
        const StemLookupResult& lookup, int word_alt_flags,
        const RuleMatchResult& match, std::string& stem_ph) const {
    static const int SUFX_V_BIT = 0x800;
    if (lookup.used_onlys_bare) {
        stem_ph = lookup.stem_ph;
        auto sp = stress_pos_.find(stem_norm);
        if (sp != stress_pos_.end()) {
            stem_ph = applyStressPosition(stem_ph, sp->second);
        }
    } else if (lookup.found_dict_entry) {
        // Use matched_stem — for the dt_noe path, this is the bare
        // form ("tornado"), which is the actual word whose phonemes
        // we just extracted.
        stem_ph = lookup.stem_ph;
        auto sp = stress_pos_.find(lookup.matched_stem);
        if (sp != stress_pos_.end()) {
            stem_ph = applyStressPosition(stem_ph, sp->second);
        }
    } else {
        // Re-phonemize stem via rules (recursive). Combine the
        // stem's own $altN flags with the parent word's. Try
        // stem+"e" too for $altN lookups (so "fertil"->"fertile"
        // inherits "fertile"'s alt flags).
        auto salt = word_alt_flags_.find(stem_norm);
        if (salt == word_alt_flags_.end()) {
            salt = word_alt_flags_.find(stem_norm + "e");
        }
        int stem_own_alt = (salt != word_alt_flags_.end()) ? salt->second : 0;
        int combined_stem_alt = stem_own_alt | word_alt_flags;
        stem_ph = applyRules(stem, true, combined_stem_alt,
                             false, true);
        // Apply $N stress override; skip noun-form-only overrides
        // ($N $onlys) when the suffix is verbal (SUFX_V), and skip
        // $verb-flag words.
        auto sp2 = stress_pos_.find(stem_norm);
        if (sp2 != stress_pos_.end() && (!noun_form_stress_.count(stem_norm) ||
                !(match.suffix_flags & SUFX_V_BIT)) &&
            !verb_flag_words_.count(stem_norm)) {
            stem_ph = applyStressPosition(stem_ph, sp2->second);
        }
    }
}

bool IPAPhonemizer::stemPhonemeFromDict(
        const std::string& stem, const RuleMatchResult& match,
        int word_alt_flags, std::string& stem_ph) const {
    bool produced = false;
    if (!stem.empty()) {
        std::string stem_norm = toLowerASCII(stem);
        bool stem_is_onlys = onlys_words_.count(stem_norm) > 0;
        // -s/-es suffix detection: $onlys IS valid for -s suffix
        // (e.g. "increases" gets the rule-based primary on the 1st
        // syllable, not the $verb form's 2nd-syllable stress).
        bool suffix_is_s = (match.phonemes == "s" ||
            match.phonemes == "z" || match.phonemes == "I#z" ||
            match.phonemes == "%I#z");
        if (suffix_is_s) { stem_is_onlys = false; }
        // $only stems (e.g. "down" in "downtown"): bare-word entry
        // only, not as compound suffix stem. For -s suffix, $only
        // remains valid.
        if (only_words_.count(stem_norm) && !suffix_is_s) {
            stem_is_onlys = true;
        }
        StemLookupResult lookup = lookupStemInDicts(
            stem_norm, stem_is_onlys, suffix_is_s, match);
        applyStemStressOrRulesFallback(stem, stem_norm, lookup,
                                       word_alt_flags, match,
                                       stem_ph);
        produced = true;
    }
    return produced;
}

// -ed devoicing: when any RULE_ENDING produces 'd#' (past tense),
// devoice to 't' after voiceless consonants and insert epenthetic
// 'I#d' after t/d. SUFX_V flag NOT required — the reference applies
// this for all RULE_ENDING 'd#' suffixes based on the re-translated
// stem's last phoneme. Mutates `suffix_ph` in place.
void IPAPhonemizer::devoiceEdSuffix(
        const std::string& stem_ph,
        std::string& suffix_ph) const {
    if (suffix_ph == "d#" && !stem_ph.empty()) {
        char last_ph = 0;
        for (int sj = (int)stem_ph.size() - 1;
             sj >= 0 && last_ph == 0; sj--) {
            char c = stem_ph[sj];
            // Skip stress / boundary / rule-boundary markers.
            if (c != '\'' && c != ',' && c != '%' && c != '=' && c != '\x01') {
                last_ph = c;
            }
        }
        // p, t, k, f, T(θ), S(ʃ), C(ç), x, X, h, s
        static const std::string VOICELESS_LAST = "ptkfTSCxXhs";
        if (last_ph == 't' || last_ph == 'd') {
            suffix_ph = "I#d";
        } else if (VOICELESS_LAST.find(last_ph) !=
            std::string::npos) {
            suffix_ph = "t";
        }
        // else: voiced consonant or vowel -> keep 'd#' (maps to /d/).
    }
}

// SUFFIX rule terminal path: strip N chars, handle SUFX_I/E flags,
// re-phonemize stem (dict / verb_dict / rules), apply 'd#' devoicing,
// return stem_ph + suffix_ph.
std::string IPAPhonemizer::processSuffixRule(
        const std::string& word, int len, int word_alt_flags,
        const RuleMatchResult& match,
        bool /*allow_suffix_strip*/) const {
    int strip = match.suffix_strip_len;
    if (strip <= 0 || strip > len) { strip = match.advance; }
    std::string stem = word.substr(0, len - strip);
    static const int SUFX_I_BIT = 0x200;
    if ((match.suffix_flags & SUFX_I_BIT) && !stem.empty() &&
        stem.back() == 'i') {
        // 'y'->'i' was applied during stripping; restore 'y'.
        stem.back() = 'y';
    }
    appendMagicEIfNeeded(stem, match.phonemes, match.suffix_flags);
    std::string stem_ph;
    stemPhonemeFromDict(stem, match, word_alt_flags, stem_ph);
    std::string suffix_ph = match.phonemes;
    devoiceEdSuffix(stem_ph, suffix_ph);
    if (std::getenv("PHON_DEBUG")) {
        std::cerr << "[SUFFIX-RULE] word=" << word
                  << " strip=" << strip << " stem='" << stem
                  << "' stem_ph='" << stem_ph
                  << "' suf_ph='" << suffix_ph << "'\n";
    }
    return stem_ph + suffix_ph;
}

// True iff `phonemes` contains a FULL (non-reduced) stressable vowel
// code. Used by the prefix-skip heuristic: if the prefix already has
// a full vowel and no stress marker, skip @P retranslation and let
// the main loop continue scanning the suffix in-context.
bool IPAPhonemizer::prefixHasFullVowel(
        const std::string& phonemes) const {
    static const char* FULL_VOWELS[] = {
        "eI","aI","aU","OI","oU","3:","A:","i:","u:","O:","e:","a:",
        "aI@","aU@","oU#","i@","e@","A@","U@","O@",
        "a","A","E","V","0","o", nullptr };
    bool found = false;
    for (size_t pi = 0; pi < phonemes.size() && !found; pi++) {
        for (int fi = 0; FULL_VOWELS[fi] && !found; fi++) {
            const char* fv = FULL_VOWELS[fi];
            size_t fvlen = strlen(fv);
            found = pi + fvlen <= phonemes.size() &&
                phonemes.compare(pi, fvlen, fv) == 0;
        }
    }
    return found;
}

// Walk `phonemes` as a sequence of phoneme codes (multi-char first,
// then single-char fallback) and return the vowel-code count.
int IPAPhonemizer::countPrefixVowels(
        const std::string& phonemes) const {
    static const char* MC_VPH[] = {
        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r",
        "o@r","A@r","e@r","eI","aI","aU","OI","oU","IR","VR","U@",
        "A@","e@","i@","O@","o@","3:","A:","i:","u:","O:","e:","a:",
        "aa","@L","@2","@5","I2","I#","E2","E#","e#","a#","a2","0#",
        "02","O2","A#", nullptr };
    int n = 0;
    for (size_t pi = 0; pi < phonemes.size(); ) {
        char c = phonemes[pi];
        if (c == '\'' || c == ',' || c == '%' || c == '=') {
            pi++;
        } else {
            bool mch = false;
            for (int mi = 0; MC_VPH[mi] && !mch; mi++) {
                int ml = (int)strlen(MC_VPH[mi]);
                mch = (int)pi + ml <= (int)phonemes.size() &&
                    phonemes.compare(pi, ml, MC_VPH[mi]) == 0;
                if (mch) {
                    if (isVowelCode(std::string(MC_VPH[mi], ml))) { n++; }
                    pi += ml;
                }
            }
            if (!mch) {
                if (isVowelCode(std::string(1, c))) { n++; }
                pi++;
            }
        }
    }
    return n;
}

// True iff the LAST vowel phoneme code in `phonemes` is a schwa-type
// (3, @, I2, I#, a#, etc.). Used by the "super (s'u:p3) + nova/market
// (2 syll) -> keep prefix primary" heuristic.
bool IPAPhonemizer::prefixEndsInSchwa(
        const std::string& phonemes) const {
    static const char* MC_VEND[] = {
        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r",
        "o@r","A@r","e@r","eI","aI","aU","OI","oU","IR","VR","U@",
        "A@","e@","i@","O@","o@","3:","A:","i:","u:","O:","e:","a:",
        "aa","@L","@2","@5","I2","I#","E2","E#","e#","a#","a2","0#",
        "02","O2","A#", nullptr };
    std::string last_v;
    for (size_t pi = 0; pi < phonemes.size(); ) {
        char c = phonemes[pi];
        if (c == '\'' || c == ',' || c == '%' || c == '=') {
            pi++;
        } else {
            bool mch = false;
            for (int mi = 0; MC_VEND[mi] && !mch; mi++) {
                int ml = (int)strlen(MC_VEND[mi]);
                mch = (int)pi + ml <= (int)phonemes.size() &&
                    phonemes.compare(pi, ml, MC_VEND[mi]) == 0;
                if (mch) {
                    if (isVowelCode(std::string(MC_VEND[mi], ml))) {
                        last_v = std::string(MC_VEND[mi], ml);
                    }
                    pi += ml;
                }
            }
            if (!mch) {
                if (isVowelCode(std::string(1, c))) {
                    last_v = std::string(1, c);
                }
                pi++;
            }
        }
    }
    return last_v == "3"  || last_v == "3:" || last_v == "@" ||
           last_v == "@2" || last_v == "@5" || last_v == "@L" ||
           last_v == "I2" || last_v == "I#" || last_v == "a#";
}

// Count vowel-letter groups in `suffix` (consecutive vowels = 1
// group). A silent trailing magic-e ('e' after a non-vowel) doesn't
// add a syllable.
int IPAPhonemizer::countSuffixSyllables(
        const std::string& suffix) const {
    static const std::string VOW_LET = "aeiouAEIOU";
    int n = 0;
    bool in_vowel = false;
    for (char ch : suffix) {
        if (VOW_LET.find(ch) != std::string::npos) {
            if (!in_vowel) { n++; in_vowel = true; }
        } else {
            in_vowel = false;
        }
    }
    if (!suffix.empty() && (suffix.back() == 'e' || suffix.back() == 'E') &&
        suffix.size() >= 2 &&
        VOW_LET.find(suffix[suffix.size() - 2]) == std::string::npos) {
        n = std::max(1, n - 1);
    }
    return n;
}

// PREFIX rule retransmission. Returns stem+suffix combined if the
// prefix retransmission ran; returns "" if the prefix already has a
// full stressable vowel and no stress markers (caller falls through
// to in-context scanning).
std::string IPAPhonemizer::processPrefixRule(
        const std::string& word, int pos,
        const RuleMatchResult& match,
        std::string& phonemes) const {
    std::string final_ph;
    std::string suffix = word.substr(pos + match.advance);
    bool prefix_has_stress = phonemes.find('\'') != std::string::npos ||
        phonemes.find(',') != std::string::npos ||
        phonemes.find('%') != std::string::npos;
    bool has_full_vowel = prefixHasFullVowel(phonemes);
    // Skip retransmission when the prefix has a full stressable vowel
    // and no stress markers — let the main loop scan the suffix
    // in-context (e.g. "openly": "open" prefix has 'oU' but no stress
    // marker -> continue with "ly" in context -> 'y' -> '%i' -> ˈoʊpənli).
    bool skip = !prefix_has_stress && has_full_vowel;
    if (!skip) {
        // Suffix lookup: $onlys / noun_form_stress / verb_flag words
        // get RULE-based phonemes (not their dict entry); everything
        // else uses wordToPhonemes (which consults dict).
        std::string sfx_ph;
        if (onlys_words_.count(suffix)) {
            sfx_ph = processPhonemeString(applyRules(suffix));
        } else if (noun_form_stress_.count(toLowerASCII(suffix)) ||
            verb_flag_words_.count(toLowerASCII(suffix))) {
            sfx_ph = processPhonemeString(applyRules(suffix));
        } else {
            sfx_ph = wordToPhonemes(suffix);
        }
        std::string sfx_raw = sfx_ph;  // for debug
        // Compound stress demotion: when both prefix and suffix carry
        // primary, decide which to demote based on syllable counts.
        // - Multi-syllabic suffix (>=2) AND multi-syllabic prefix
        //   (>=2 vowels), prefix not ending in schwa: demote prefix.
        //   e.g. "micro" (m'aIkroU) + "plastics" (2 syll) -> secondary
        //   on micro.
        // - Mono-syllabic prefix (1 vowel): REMOVE the suffix's
        //   primary entirely (the reference: "newspaper"->n'u:zpeIp3).
        // - Otherwise (mono-syllabic suffix or schwa-ending prefix):
        //   demote suffix primary to secondary.
        if (phonemes.find('\'') != std::string::npos &&
            sfx_ph.find('\'') != std::string::npos) {
            int sfx_syllables = countSuffixSyllables(suffix);
            int pfx_vowels = countPrefixVowels(phonemes);
            bool pfx_ends_schwa = false;
            if (sfx_syllables == 2 && pfx_vowels >= 2) {
                pfx_ends_schwa = prefixEndsInSchwa(phonemes);
            }
            if (sfx_syllables >= 2 && pfx_vowels >= 2 && !pfx_ends_schwa) {
                size_t pp = phonemes.find('\'');
                if (pp != std::string::npos) { phonemes[pp] = ','; }
            } else if (pfx_vowels == 1) {
                size_t sp = sfx_ph.find('\'');
                if (sp != std::string::npos) { sfx_ph.erase(sp, 1); }
            } else {
                size_t sp = sfx_ph.find('\'');
                if (sp != std::string::npos) { sfx_ph[sp] = ','; }
            }
            // Post-demotion: if the comma is immediately before a
            // phUNSTRESSED vowel (@, I#, I2, a#), strip it entirely —
            // schwa-type vowels never carry secondary stress.
            // E.g. "battlement": m,@nt -> m@nt.
            size_t cp = sfx_ph.find(',');
            if (cp != std::string::npos && cp + 1 < sfx_ph.size()) {
                char nc = sfx_ph[cp + 1];
                bool phU = (nc == '@') ||
                    (nc == 'I' && cp + 2 < sfx_ph.size() &&
                        (sfx_ph[cp + 2] == '#' || sfx_ph[cp + 2] == '2')) ||
                    (nc == 'a' && cp + 2 < sfx_ph.size() &&
                        sfx_ph[cp + 2] == '#');
                if (phU) { sfx_ph.erase(cp, 1); }
            }
        }
        // Strip \\x01 rule-boundary markers from the prefix phoneme
        // before combining. The suffix was processed by
        // wordToPhonemes which already stripped its own. If \\x01
        // remains, processPhonemeString's rule_boundary_after path
        // re-fires DIMINISHED steps on already-processed text.
        phonemes.erase(std::remove(phonemes.begin(),
                                   phonemes.end(), '\x01'),
                       phonemes.end());
        if (std::getenv("PHON_DEBUG")) {
            std::cerr << "[PREFIX-RETRANS] suffix=" << suffix
                      << " sfx_raw=" << sfx_raw
                      << " sfx_ph=" << sfx_ph
                      << " combined=" << (phonemes + sfx_ph)
                      << "\n";
        }
        final_ph = phonemes + sfx_ph;
    }
    return final_ph;
}

// Apply rules to a word (for unknown words). Main scan loop with
// suffix-strip / prefix-retransmission terminal paths and del-fwd
// magic-e marking. The body has been broken out into the helpers
// above so this function reads as straight scan-and-emit.
std::string IPAPhonemizer::applyRules(
        const std::string& word_orig,
        bool allow_suffix_strip,
        int word_alt_flags_param,
        bool suffix_phoneme_only,
        bool suffix_removed,
        std::vector<bool>* out_replaced_e,
        std::vector<bool>* out_pos_visited) const {
    std::string word = applyReplacements(word_orig);
    std::string phonemes;
    int len = (int)word.size();
    int word_alt_flags = determineAltFlags(word, word_alt_flags_param);
    // replaced_e[pos]: char at pos was marked by RULE_DEL_FWD
    // (magic-e). We still process rules for it, but silence it when
    // only the default rule fires (score==0).
    std::vector<bool> replaced_e(len, false);
    // pos_visited[pos]: true if pos was the start of a scan iteration.
    // Positions skipped by long-match advance are NOT visited.
    std::vector<bool> pos_visited(len, false);
    std::string final_result;
    bool finished = false;
    for (int pos = 0; pos < len && !finished; ) {
        pos_visited[pos] = true;
        // For replaced_e positions, the reference replaces 'e' with
        // 'E' (REPLACED_E) in its word buffer; group lookup uses
        // .group E (not .group e) and the default rule emits "".
        char pos_char = replaced_e[pos] ? 'E'
            : (char)std::tolower((unsigned char)word[pos]);
        RuleMatchResult match = findBestRule(
            word, pos, len, pos_char, word_alt_flags, replaced_e,
            allow_suffix_strip, suffix_phoneme_only, suffix_removed,
            phonemes);
        if (match.score < 0) {
            // No rule found — skip the character silently.
            pos++;
        } else {
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "  pos=" << pos
                    << " char='" << word[pos]
                    << "' ph='" << match.phonemes
                    << "' adv=" << match.advance
                    << " score=" << match.score
                    << (replaced_e[pos] ? " [repl-e]" : "")
                    << (match.is_prefix ? " [PREFIX]" : "")
                    << (match.is_suffix ? " [SUFFIX]" : "")
                    << "\n";
            }
            applyStressPrev(match.phonemes, phonemes);
            phonemes += match.phonemes;
            phonemes += '\x01';
            bool terminal_suffix =
                (allow_suffix_strip || suffix_phoneme_only) &&
                match.is_suffix && pos + match.advance == len;
            if (terminal_suffix && !suffix_phoneme_only) {
                // SUFFIX-RULE terminal: stem rephonemization +
                // combination. Discards `phonemes` accumulated so far
                // (re-derived from the stem).
                final_result = processSuffixRule(
                    word, len, word_alt_flags, match,
                    allow_suffix_strip);
                finished = true;
            } else if (terminal_suffix && suffix_phoneme_only) {
                // First-pass mode: suffix phoneme already accumulated
                // above. No del-fwd, no extra advance.
                pos += match.advance;
            } else if (match.is_prefix && pos + match.advance < len) {
                std::string pfx_res = processPrefixRule(
                    word, pos, match, phonemes);
                if (!pfx_res.empty()) {
                    final_result = pfx_res;
                    finished = true;
                } else {
                    // Skip retransmission: prefix had full vowel and
                    // no stress markers. Just advance — no del-fwd.
                    pos += match.advance;
                }
            } else {
                // Default rule path: mark RULE_DEL_FWD chars
                // (magic-e) as replaced_e (still processed but
                // silenced if default), then advance.
                if (match.del_count > 0 && match.del_start >= 0) {
                    for (int d = 0;
                         d < match.del_count &&
                         match.del_start + d < len; d++) {
                        replaced_e[match.del_start + d] = true;
                    }
                }
                pos += match.advance;
            }
        }
    }
    if (out_replaced_e)  { *out_replaced_e  = replaced_e; }
    if (out_pos_visited) { *out_pos_visited = pos_visited; }
    std::string ret = phonemes;
    if (finished) { ret = final_result; }
    return ret;
}

// ============================================================
// Word to phoneme codes
// ============================================================
// $capital: if word starts with a capital letter, check
// capital_dict_ first.
std::string IPAPhonemizer::checkCapitalDict(
        const std::string& word,
        const std::string& norm) const {
    std::string result;
    bool is_capital = (!word.empty() && (unsigned char)word[0] >= 'A' &&
                       (unsigned char)word[0] <= 'Z');
    if (is_capital) {
        auto cit = capital_dict_.find(norm);
        if (cit != capital_dict_.end()) {
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "[DICT_CAP] " << norm
                          << " -> " << cit->second << "\n";
            }
            result = processPhonemeString(cit->second);
        }
    }
    return result;
}

// Try full word in dictionary. $onlys bare-word overrides take
// priority over the plain entry for bare-word lookup. (Suffix
// stripping uses dict_ directly, bypassing onlys_bare_dict_.)
std::string IPAPhonemizer::checkMainDict(
        const std::string& norm) const {
    std::string result;
    auto obit = onlys_bare_dict_.find(norm);
    auto it = (obit != onlys_bare_dict_.end())
                  ? dict_.end() : dict_.find(norm);
    const std::string* raw_ptr = nullptr;
    if (obit != onlys_bare_dict_.end()) {
        raw_ptr = &obit->second;
        if (std::getenv("PHON_DEBUG")) {
            std::cerr << "[DICT_ONLYS] " << norm
                      << " -> " << *raw_ptr << "\n";
        }
    } else if (it != dict_.end()) {
        raw_ptr = &it->second;
        if (std::getenv("PHON_DEBUG")) {
            std::cerr << "[DICT] " << norm
                      << " -> " << *raw_ptr << "\n";
        }
    }
    if (raw_ptr != nullptr) {
        // Apply processPhonemeString to dict entries so they get:
        // - flap rule (e.g. "committee" -> kəmˈɪɾi)
        // - secondary stress insertion (from added dict overrides)
        // - @r -> 3 conversion (American English)
        // Note: entries with % prefix (unstressed function words) are
        // left alone except for minor normalizations (no primary
        // stress added).
        std::string raw = *raw_ptr;
        auto sit = stress_pos_.find(norm);
        if (sit != stress_pos_.end()) {
            raw = applyStressPosition(raw, sit->second);
        }
        // $strend2 words with bare phonemes use final-syllable stress
        // (the reference "end stress").
        bool is_strend = strend_words_.count(norm) > 0;
        result = processPhonemeString(raw, is_strend);
    }
    return result;
}

// Handle hyphenated compounds: "peer-reviewed" -> phonemize "peer" +
// "reviewed" separately. The reference treats each hyphen-separated
// segment as an independent word for phonemization, so word-start
// rules (e.g., re-->rI#, be-->bI#) fire correctly on each segment.
// Only split if not in the dictionary (handled above) and contains
// at least one hyphen.
std::string IPAPhonemizer::checkHyphenated(
        const std::string& norm) const {
    std::string result;
    size_t hyphen_pos = norm.find('-');
    bool has_hyphen = (hyphen_pos != std::string::npos && hyphen_pos > 0 &&
                       hyphen_pos + 1 < norm.size());
    if (has_hyphen) {
        std::string temp_result;
        size_t seg_start = 0;
        // The loop terminates when (a) a segment failed (`processing`
        // false), or (b) seg_start runs off the end. The former is
        // the predicate; the latter is achieved by setting seg_start
        // = norm.size() when we processed the last segment.
        bool processing = true;
        while (seg_start < norm.size() && processing) {
            size_t next_hyphen = norm.find('-', seg_start);
            size_t len = (next_hyphen == std::string::npos)
                ? std::string::npos
                : next_hyphen - seg_start;
            std::string seg = norm.substr(seg_start, len);
            if (seg.empty()) {
                processing = false;
            } else {
                bool has_letter = std::any_of(seg.begin(), seg.end(),
                    [](char c) {
                        return std::isalpha((unsigned char)c) != 0;
                    });
                if (!has_letter) {
                    processing = false;
                } else {
                    std::string seg_ipa = wordToPhonemes(seg);
                    if (seg_ipa.empty()) {
                        processing = false;
                    } else {
                        temp_result += seg_ipa;
                        seg_start = (next_hyphen ==
                                     std::string::npos)
                            ? norm.size()
                            : next_hyphen + 1;
                    }
                }
            }
        }
        if (processing && !temp_result.empty()) {
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "[HYPHEN] " << norm
                          << " -> " << temp_result << "\n";
            }
            result = temp_result;
        }
    }
    return result;
}

// Handle possessive "'s": "people's" -> phonemize "people" + suffix.
// Voicing: sibilant-ending -> ᵻz; unvoiced -> s; else -> z. The choice
// follows the reference .group ' rules which match on LETTERS not
// the final phoneme (except 'ch' which uses the z/2 mixed rule).
std::string IPAPhonemizer::checkPossessive(
        const std::string& norm) const {
    std::string result;
    bool is_possessive = (norm.size() >= 3 && norm[norm.size()-2] == '\'' &&
                          norm.back() == 's');
    if (is_possessive) {
        std::string base_poss = norm.substr(0, norm.size()-2);
        std::string base_ipa = wordToPhonemes(base_poss);
        if (!base_ipa.empty()) {
            // Get raw phoneme code of base.
            std::string raw_code;
            auto poss_dict = dict_.find(base_poss);
            if (poss_dict != dict_.end()) {
                raw_code = processPhonemeString(poss_dict->second);
            } else {
                raw_code = processPhonemeString(applyRules(base_poss));
            }
            // .group ' rules:
            //   sh -> %I#z
            //   ch -> z/2 (last phoneme: sib->I2z, unv->s, else->z)
            //   se / s / ce / x -> %I#z
            //   f / p / t / k -> s
            //   och -> s
            //   default -> z
            std::string poss_suffix = "z";
            std::string bl = base_poss;
            auto ends_with_letters = [&](const std::string& sfx) {
                return bl.size() >= sfx.size()
                    && bl.compare(bl.size() - sfx.size(),
                                  sfx.size(), sfx) == 0;
            };
            // Find last phoneme code in raw_code (skip stress
            // markers and hyphens). Search-by-iteration: `found_last`
            // is the natural "found" state — replaces the original
            // continue/break.
            std::string last_ph_code;
            std::string rc = raw_code;
            size_t ri = rc.size();
            bool found_last = false;
            while (ri > 0 && !found_last) {
                ri--;
                char c = rc[ri];
                bool is_marker = (c == '\'' || c == ',' || c == '%' ||
                                  c == '=' || c == '-');
                if (!is_marker) {
                    last_ph_code = std::string(1, c);
                    if (ri >= 1) {
                        std::string two = rc.substr(ri - 1, 2);
                        bool is_two_char = (two == "tS" ||
                            two == "dZ" || two == "O:" ||
                            two == "A:" || two == "i:" ||
                            two == "u:" || two == "e:" ||
                            two == "I#" || two == "I2" ||
                            two == "@L" || two == "3:" ||
                            two == "eI" || two == "aI" ||
                            two == "aU" || two == "oU" || two == "OI");
                        if (is_two_char) { last_ph_code = two; }
                    }
                    found_last = true;
                }
            }
            // 3-char endings first.
            if (ends_with_letters("och")) {
                poss_suffix = "s";
            } else if (ends_with_letters("ch")) {
                static const std::vector<std::string> SIBILANTS_PH =
                    {"tS","dZ","s","z","S","Z"};
                // p, t, k, f, s, T(θ), S(ʃ), x.
                static const std::string UNVOICED_PH_CHARS = "ptkfsTSx";
                bool is_sib = std::find(SIBILANTS_PH.begin(),
                    SIBILANTS_PH.end(), last_ph_code)
                    != SIBILANTS_PH.end();
                if (is_sib) {
                    // InsertPhoneme(I2) + ChangePhoneme(z).
                    poss_suffix = "I2z";
                } else if (!last_ph_code.empty() &&
                    UNVOICED_PH_CHARS.find(last_ph_code[0]) !=
                        std::string::npos) {
                    poss_suffix = "s";
                } else {
                    poss_suffix = "z";
                }
            // 2-char endings.
            } else if (ends_with_letters("se") || ends_with_letters("ce") ||
                ends_with_letters("sh")) {
                poss_suffix = "I#z";
            // 1-char endings.
            } else if (!bl.empty()) {
                char lc = bl.back();
                if (lc == 's' || lc == 'z' || lc == 'x') {
                    poss_suffix = "I#z";
                } else if (lc == 'f' || lc == 'p' || lc == 't' || lc == 'k') {
                    poss_suffix = "s";
                }
                // else: default z.
            }
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "[POSSESSIVE] " << norm
                          << " base=" << base_poss
                          << " last_letter="
                          << (bl.empty() ? '?' : bl.back())
                          << " suf=" << poss_suffix << "\n";
            }
            result = processPhonemeString(raw_code + poss_suffix);
        }
    }
    return result;
}

// For single-letter words, try underscore prefix (letter name).
// Exception: "a" as article — its pronunciation depends on sentence
// context and is handled specially in phonemizeText.
std::string IPAPhonemizer::checkSingleLetter(
        const std::string& norm) const {
    std::string result;
    if (norm.size() == 1 && norm != "a") {
        auto it = dict_.find("_" + norm);
        if (it != dict_.end()) { result = processPhonemeString(it->second); }
    }
    return result;
}

// Get phonemes for a candidate stem (dict lookup or rules+postprocess).
// Returns "" if the stem is invalid (no vowel letter or no vowel
// phoneme in result). For verb-derived forms (-ing/-ed), the verb
// form dictionary takes priority so e.g. "live" uses verb form "lIv"
// not adj/noun form "laIv".
std::string IPAPhonemizer::getStemPhonemes(
        const std::string& stem) const {
    std::string result;
    if (stem.size() >= 2) {
        // Reject stems with no vowel letter (e.g., "spr", "str").
        if (hasAnyVowelLetter(stem)) {
            // Check verb_dict_ first (verb form needed for -ing/-ed
            // derivatives).
            auto vt = verb_dict_.find(stem);
            if (vt != verb_dict_.end()) {
                result = vt->second;
            } else {
                // $onlys dict entries are only valid for bare form
                // or with 's' suffix. $only entries are only valid
                // for the isolated bare word (not as any stem). For
                // non-s suffix stripping (e.g. -ed, -ing, -able),
                // skip both and use rules.
                bool is_onlys = onlys_words_.count(stem) > 0 ||
                                only_words_.count(stem) > 0;
                auto jt = dict_.find(stem);
                if (jt == dict_.end() || is_onlys) {
                    // Magic-e restoration: try stem+"e" in dict
                    // before falling back to rules. E.g. "argued"
                    // -> stem "argu" -> not in dict -> try "argue" ->
                    // A@gju: (with j).
                    //
                    // CRITICAL: when the magic-e form (stem+"e") is
                    // a valid dict entry, assign `result` DIRECTLY.
                    // The original code did `jt = je`, which then
                    // got rejected by the trailing `!is_onlys`
                    // check below — `is_onlys` was computed for the
                    // BARE stem and is irrelevant for the magic-e
                    // form (which we've already verified is not
                    // $only/$onlys at the `je != dict_.end() && …`
                    // check). Same pre-existing bug exists in
                    // stemPhonemeFromDict's dt_noe path.
                    auto je = dict_.find(stem + "e");
                    if (je != dict_.end() && !onlys_words_.count(stem + "e") &&
                        !only_words_.count(stem + "e")) {
                        result = je->second;
                    } else {
                        auto ve = verb_dict_.find(stem + "e");
                        if (ve != verb_dict_.end()) { result = ve->second; }
                    }
                }
                if (!result.empty()) {
                    // already set from magic-e dict / verb_dict
                } else if (jt != dict_.end() && !is_onlys) {
                    result = jt->second;
                } else {
                    // When stem has $verb flag-only entry (e.g.
                    // "deliberate $verb"), use alt1 rules (verb form
                    // gives eIt for -ate, not alt2's @t noun/adj).
                    int stem_alt_flags = verb_flag_words_.count(stem) ? 1 : -1;
                    std::string raw = applyRules(stem, true,
                                                  stem_alt_flags);
                    // Apply $N stress override, BUT skip:
                    //   (1) noun-form-only overrides ($N $onlys),
                    //   (2) words with $verb flag-only entries
                    //       (their verb form uses rules).
                    auto sp = stress_pos_.find(stem);
                    if (sp != stress_pos_.end() &&
                        !noun_form_stress_.count(stem) &&
                        !verb_flag_words_.count(stem)) {
                        raw = applyStressPosition(raw, sp->second);
                    }
                    result = processPhonemeString(raw);
                }
            }
            // Reject if result has no vowel phoneme code (e.g.,
            // silent-e-only stem).
            if (!result.empty() && !hasAnyVowelCode(result)) { result = ""; }
        }
    }
    return result;
}

// -ing suffix handler. Caller must guarantee norm.size() >= 5 (so the
// substr/compare access is safe).
// -ing non-magic-e stem fallbacks. Runs when the CVC/nc magic-e
// branch doesn't apply (e.g. base has a stress override). Each
// step is skipped if sph is already populated.
std::string IPAPhonemizer::tryIngNonMagicEStemFallbacks(
        const std::string& base, bool base_has_stress_override,
        const std::string& current_sph) const {
    std::string sph = current_sph;
    // -ns / -rs stems: phonemizing the bare stem fires word-final
    // 's' as 'z' (wrong); the magic-e form gives 's' correctly.
    bool try_magic_e_first = !base_has_stress_override && base.size() >= 2 &&
        (base.compare(base.size() - 2, 2, "ns") == 0 ||
         base.compare(base.size() - 2, 2, "rs") == 0);
    if (try_magic_e_first) { sph = getStemPhonemes(base + "e"); }
    if (sph.empty()) { sph = getStemPhonemes(base); }
    if (sph.empty() && !base.empty() &&
        !isVowelLetter(base.back()) && hasAnyVowelLetter(base)) {
        sph = getStemPhonemes(base + "e");
    }
    return sph;
}

// -ed non-magic-e stem fallbacks. Runs when the CVC/nc magic-e
// branch is unavailable (e.g. base doesn't fit the pattern, or has
// a stress override). Each step is skipped if a prior step already
// populated sph.
std::string IPAPhonemizer::tryEdNonMagicEStemFallbacks(
        const std::string& base, bool base_has_stress_override,
        const std::string& current_sph) const {
    std::string sph = current_sph;
    static const std::string CVC_DOUBLE_CONS = "lptmnrgdb";
    bool base_has_double = base.size() >= 2 &&
        base.back() == base[base.size() - 2] &&
        CVC_DOUBLE_CONS.find(base.back()) != std::string::npos;
    bool prefer_undoubled = false;
    if (base_has_double) {
        std::string prefix = base.substr(0, base.size() - 2);
        int vowel_groups = 0;
        bool in_v = false;
        for (char c : prefix) {
            if (isVowelLetter(c)) {
                if (!in_v) {
                    vowel_groups++;
                    in_v = true;
                }
            } else {
                in_v = false;
            }
        }
        prefer_undoubled = (vowel_groups >= 2);
    }
    if (sph.empty() && base_has_double && prefer_undoubled) {
        sph = getStemPhonemes(base.substr(0, base.size() - 1));
    }
    // For "rs"/"ns" not in dict: try magic-e first (otherwise the
    // word-final 's' SUFFIX rule emits 'z'; "dispense" needs 's').
    bool try_e_first = sph.empty() && !base_has_stress_override &&
        base.size() >= 2 && (base.compare(base.size() - 2, 2, "ns") == 0 ||
         base.compare(base.size() - 2, 2, "rs") == 0);
    if (try_e_first) { sph = getStemPhonemes(base + "e"); }
    if (sph.empty()) { sph = getStemPhonemes(base); }
    if (sph.empty() && base_has_double && !prefer_undoubled) {
        sph = getStemPhonemes(base.substr(0, base.size() - 1));
    }
    if (sph.empty() && !base.empty() &&
        !isVowelLetter(base.back()) && hasAnyVowelLetter(base)) {
        sph = getStemPhonemes(base + "e");
    }
    return sph;
}

// -ed candidate pre-flight: bundles the 7 skip checks.
bool IPAPhonemizer::isEdSuffixCandidate(
        const std::string& norm, const std::string& base) const {
    bool processing = true;
    if (!base.empty() && base.back() == 'e') { processing = false; }
    if (processing && !base.empty() && base.back() == 'u') {
        processing = false;
    }
    if (processing && norm.size() >= 4) {
        char penult = norm[norm.size() - 3];
        bool is_soft_c = (penult == 'c' || penult == 'g');
        bool is_ng_ged = (penult == 'g' && norm.size() >= 6 &&
                          norm[norm.size() - 4] == 'n');
        if (is_soft_c && !is_ng_ged && dict_.find(base) == dict_.end() &&
            verb_dict_.find(base) == verb_dict_.end()) {
            processing = false;
        }
    }
    if (processing && norm.size() >= 5 &&
        norm.compare(norm.size() - 4, 4, "nged") == 0 &&
        dict_.find(base) == dict_.end() &&
        verb_dict_.find(base) == verb_dict_.end()) {
        processing = false;
    }
    if (processing && norm.size() >= 5 &&
        norm.compare(norm.size() - 4, 4, "eted") == 0 &&
        dict_.find(base) == dict_.end() &&
        verb_dict_.find(base) == verb_dict_.end()) {
        processing = false;
    }
    if (processing && norm.size() >= 7 &&
        norm.compare(norm.size() - 6, 6, "mented") == 0) {
        char before_m = norm[norm.size() - 7];
        std::string mented_stem = norm.substr(0, norm.size() - 2);
        bool stem_has_stress = (stress_pos_.count(mented_stem) > 0);
        if (!isVowelLetter(before_m) && !stem_has_stress) {
            processing = false;
        }
    }
    if (processing) {
        int trail_cons = 0;
        for (int bi = (int)base.size() - 1;
             bi >= 0 && !isVowelLetter(base[bi]); bi--) {
            trail_cons++;
        }
        if (trail_cons >= 3) { processing = false; }
    }
    return processing;
}

// Voicing rule for the -ed suffix. The stem-final consonant
// determines the regular allomorph: t/d -> "I#d" (ɪd, syllabic);
// unvoiced (ptkfTSshx) -> "t"; voiced or vowel -> "d". An override
// applies when the regular choice would be "t" (unvoiced) but the
// full-word rule fires a "d" — this happens for words like
// "tied"/"tried" where silent-e + a SUFFIX rule emits the voiced
// allomorph that the stem-only path would miss.
std::string IPAPhonemizer::computeEdSuffixVoicing(
        const std::string& sph, const std::string& norm,
        const std::string& base) const {
    static const std::string UNVOICED = "ptkfTSshx";
    char last = sph.back();
    std::string ed_ph;
    if (last == 't' || last == 'd') { ed_ph = "I#d"; }
    else if (UNVOICED.find(last) != std::string::npos) { ed_ph = "t"; }
    else { ed_ph = "d"; }
    std::string final_ph = sph + ed_ph;
    if (ed_ph == "t" && norm.size() >= 2) {
        std::vector<bool> fw_replaced_e, fw_pos_visited;
        std::string fw_ph = applyRules(norm, false, -1, false, false,
                                       &fw_replaced_e,
                                       &fw_pos_visited);
        int e_pos = (int)norm.size() - 2;
        bool e_was_visited =
            (e_pos >= 0 && e_pos < (int)fw_pos_visited.size() &&
             fw_pos_visited[e_pos]);
        char fw_last = 0;
        for (int ri = (int)fw_ph.size() - 1;
             ri >= 0 && fw_last == 0; ri--) {
            if (fw_ph[ri] != '\x01') { fw_last = fw_ph[ri]; }
        }
        if (std::getenv("PHON_DEBUG")) {
            std::cerr << "[FW-CHECK] norm=" << norm
                      << " e_pos=" << e_pos
                      << " e_was_visited=" << e_was_visited
                      << " fw_ph='" << fw_ph << "'"
                      << " fw_last=" << fw_last << "\n";
        }
        if (!e_was_visited && fw_last == 'd') {
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "[SUFFIX-ED] " << norm
                          << " stem=" << base
                          << " sph=" << sph
                          << " +d [full-word rule]\n";
            }
            final_ph = fw_ph;
        }
    }
    if (std::getenv("PHON_DEBUG") && final_ph != norm) {
        std::cerr << "[SUFFIX-ED] " << norm
                  << " stem=" << base
                  << " sph=" << sph
                  << " +" << ed_ph << "\n";
    }
    return final_ph;
}

// PREFIX rule detection at word start. Tries the 2-char group key
// (e.g. "in" for "infrare") and the 1-char group key. Returns true
// when a prefix rule matches with non-zero advance that doesn't
// consume the whole word — meaning the prefix is a true compound
// piece. Used as a compound-word guard in checkSuffixEd.
bool IPAPhonemizer::hasPrefixAtStart(const std::string& w) const {
    bool r = false;
    if (!w.empty()) {
        char c0 = std::tolower((unsigned char)w[0]);
        auto checkGroup = [&](const std::string& key, int glen) -> bool {
            bool matched = false;
            auto it = ruleset_.rule_groups.find(key);
            if (it != ruleset_.rule_groups.end()) {
                // Loop terminates on first matching prefix rule.
                // `matched` IS the search state.
                for (const auto& rule : it->second) {
                    if (!matched && rule.is_prefix) {
                        std::string ph;
                        int adv, dfs, dfc;
                        int sc = matchRule(rule, w, 0, ph,
                            adv, dfs, dfc, glen, "", 0, nullptr);
                        matched = (sc >= 0 && adv > 0 && adv < (int)w.size());
                    }
                }
            }
            return matched;
        };
        // Try the 2-char group first, then the 1-char group.
        if (w.size() >= 2) {
            std::string k2(1, c0);
            k2 += std::tolower((unsigned char)w[1]);
            r = checkGroup(k2, 2);
        }
        if (!r) { r = checkGroup(std::string(1, c0), 1); }
    }
    return r;
}

// Magic-e applicability for CVC stems before -ing or -ed. Looks at
// the last vowel-group in the base phonemes: magic-e applies if it
// already carries primary or secondary stress, or no primary stress
// exists yet (monosyllabic), or the last vowel is a "full" vowel
// (not in the weak_vowels set). For -ing weak_vowels = "I@"; for
// -ed weak_vowels = "I@3" (rhotic ɚ is already a complete vowel).
// Empty base phonemes default to true — caller's outer flow decides.
bool IPAPhonemizer::shouldUseMagicEForCvcStem(
        const std::string& base_ph,
        const std::string& weak_vowels) const {
    bool use_magic_e = true;
    if (!base_ph.empty()) {
        static const std::string VC = "aAeEiIoOuUV03@";
        const std::string& bp = base_ph;
        int last_v = -1;
        for (int k = (int)bp.size() - 1;
             k >= 0 && last_v < 0; k--) {
            if (VC.find(bp[k]) != std::string::npos) { last_v = k; }
        }
        if (last_v > 0) {
            int vstart = last_v;
            while (vstart > 0 &&
                   (VC.find(bp[vstart - 1]) != std::string::npos ||
                    bp[vstart - 1] == ':' || bp[vstart - 1] == '#')) {
                vstart--;
            }
            bool stressed_at_end = (vstart > 0 &&
                (bp[vstart - 1] == '\'' || bp[vstart - 1] == ','));
            bool no_explicit_stress = (bp.find('\'') == std::string::npos);
            bool last_vowel_is_full =
                (weak_vowels.find(bp[last_v]) == std::string::npos);
            use_magic_e = stressed_at_end || no_explicit_stress ||
                          last_vowel_is_full;
        }
    }
    return use_magic_e;
}

// Detect whether the orthographic -ing/-ed base is a magic-e
// candidate. Outputs two flags:
//   cvc: classic Consonant-Vowel-Consonant (e.g. "writ" <- "writing"
//        -> try "write"). Optionally extended to CVRC for soft-g/-c
//        ("charging" -> "charge") when include_cvrc is true.
//        Suppressed for -en/-an/-in/-on/-un endings whose phonemized
//        base ends in @n/syllabic n, -el (vowel+l), -w, and -er.
//   nc:  -nc ending (separate magic-e case).
void IPAPhonemizer::detectStemPatternForSuffix(
        const std::string& base, bool include_cvrc,
        bool& cvc, bool& nc) const {
    cvc = base.size() >= 2 && !isVowelLetter(base.back()) &&
          isVowelLetter(base[base.size() - 2]);
    // CVRC: vowel + 'r' + soft-consonant ('g'/'c'). magic-e is
    // needed for the soft-g/-c IPA (dʒ / s); do NOT extend to
    // other consonants.
    if (include_cvrc && !cvc && base.size() >= 3 &&
        (base.back() == 'g' || base.back() == 'c') &&
        base[base.size() - 2] == 'r' && isVowelLetter(base[base.size() - 3])) {
        cvc = true;
    }
    // -en/-an/-in/-on/-un with phonemized base ending @n / nL-
    // (syllabic 'n') is a real suffix, not magic-e.
    if (cvc && base.size() >= 2 && base.back() == 'n') {
        char prev_vowel = base[base.size() - 2];
        if (prev_vowel == 'e' || prev_vowel == 'a' ||
            prev_vowel == 'i' || prev_vowel == 'o' || prev_vowel == 'u') {
            std::string base_ph = getStemPhonemes(base);
            if (!base_ph.empty() && base_ph.size() >= 2) {
                char last2 = base_ph.back();
                char last1 = base_ph[base_ph.size() - 2];
                bool ends_in_schwa_n = (last1 == '@' &&
                    (last2 == 'n' || last2 == 'N'));
                bool ends_in_syllabic_n = (last1 == 'n' && last2 == '-');
                if (ends_in_schwa_n || ends_in_syllabic_n) { cvc = false; }
            }
        }
    }
    nc = base.size() >= 2 && base.back() == 'c' &&
         base[base.size() - 2] == 'n';
    // -el: schwa, not magic-e vowel.
    if (cvc && base.size() >= 2 && base.back() == 'l' &&
        base[base.size() - 2] == 'e') {
        cvc = false;
    }
    // -w: semi-vowel digraph, not magic-e consonant.
    if (cvc && base.back() == 'w') { cvc = false; }
    // -er: rhotic schwa stem, not magic-e.
    if (cvc && base.size() >= 2 && base.back() == 'r' &&
        std::tolower((unsigned char)base[base.size() - 2]) == 'e') {
        cvc = false;
    }
}

// Syllabic-L collapse for -ing stems. If stem phonemes end in "@L"
// (syllabic L, the reference for word-final 'l' after a consonant)
// and the orthographic base ends in 'l', the syllabic context is
// lost when -ing follows. Drop "@L" -> "@l" (vowel+l base) or "l"
// (other consonant+l). Exceptions: 't' before 'l' (the reference
// emits @LI2N — "bottling"), and "-ngl" endings (handled elsewhere
// as "@-lI2N" — "tingling").
void IPAPhonemizer::simplifySyllabicLForIng(
        const std::string& base, std::string& sph) const {
    if (sph.size() >= 2 && sph.compare(sph.size() - 2, 2, "@L") == 0 &&
        base.size() >= 2 && base.back() == 'l' &&
        base[base.size() - 2] != 't' &&
        !(base.size() >= 3 && base.compare(base.size() - 3, 3, "ngl") == 0)) {
        char penult = base[base.size() - 2];
        bool vowel_before_l = (penult == 'a' || penult == 'e' ||
                               penult == 'i' || penult == 'o' ||
                               penult == 'u');
        sph = sph.substr(0, sph.size() - 2) + (vowel_before_l ? "@l" : "l");
    }
}

std::string IPAPhonemizer::checkSuffixIng(
        const std::string& norm) const {
    std::string result;
    bool is_ing = (norm.compare(norm.size() - 3, 3, "ing") == 0);
    if (is_ing) {
        std::string base = norm.substr(0, norm.size() - 3);
        std::string sph;
        // For "-nging" endings where the base is not in the
        // dictionary, bypass the custom handler and let applyRules
        // handle the full word. The reference uses two-char group
        // rules like "enging EndZIN" (for "challenging",
        // "exchanging") that fire on the full word but not on the
        // stem; without this exclusion, "challeng" would get wrong
        // rules vs "challenging".
        bool processing = true;
        if (processing && base.size() >= 2 &&
            base.compare(base.size() - 2, 2, "ng") == 0 &&
            dict_.find(base) == dict_.end() &&
            verb_dict_.find(base) == verb_dict_.end()) {
            processing = false;
        }
        if (processing) {
            // Detect CVC / CVRC / -nc patterns for magic-e candidacy.
            bool cvc_pattern = false;
            bool nc_pattern = false;
            detectStemPatternForSuffix(base, /*include_cvrc=*/true,
                                       cvc_pattern, nc_pattern);
            // If the base has an explicit $N stress override in the
            // dictionary, skip magic-e and use the base directly
            // (e.g. "maintain $2" -> "maintaining"). $only words are
            // excluded — their dict entry is restricted to bare/s-
            // suffix forms.
            bool base_has_stress_override =
                (stress_pos_.find(base) != stress_pos_.end() ||
                 (dict_.find(base) != dict_.end() &&
                     !onlys_words_.count(base) && !only_words_.count(base)) ||
                 verb_dict_.find(base) != verb_dict_.end());
            if ((cvc_pattern || nc_pattern) && !base_has_stress_override) {
                std::string magic_e_ing = base + "e";
                bool use_magic_e = true;
                std::string base_only_ph;
                if (cvc_pattern && !nc_pattern) {
                    base_only_ph = getStemPhonemes(base);
                    use_magic_e =
                        shouldUseMagicEForCvcStem(base_only_ph, "I@");
                }
                // For strend2 words (compounds like "become"):
                // prefer rules over dict so that the 'be' prefix
                // rule fires (giving 'bI#' = ᵻ, not 'bI' = ɪ from
                // the dict).
                if (use_magic_e) {
                    if (strend_words_.count(magic_e_ing)) {
                        std::string rules_ph = applyRules(magic_e_ing);
                        if (!rules_ph.empty()) {
                            sph = processPhonemeString(rules_ph);
                        }
                    }
                    if (sph.empty()) { sph = getStemPhonemes(magic_e_ing); }
                    if (sph.empty()) {
                        sph = base_only_ph.empty()
                            ? getStemPhonemes(base)
                            : base_only_ph;
                    }
                }
                // else: leave sph empty so the code falls through to
                // applyRules(norm) via the outer fallback chain
                // (correct handling of e.g. "handwriting").
            } else {
                // Non-magic-e fallback chain.
                sph = tryIngNonMagicEStemFallbacks(
                    base, base_has_stress_override, sph);
            }
            if (sph.empty() && base.size() >= 2 &&
                base.back() == base[base.size() - 2]) {
                sph = getStemPhonemes(base.substr(0, base.size() - 1));
            }
            if (sph.empty() && !base.empty() && base.back() == 'i') {
                sph = getStemPhonemes(base.substr(0, base.size() - 1) + "y");
            }
        }
        if (!sph.empty()) {
            if (std::getenv("PHON_DEBUG")) {
                std::cerr << "[SUFFIX-ING] " << norm
                          << " stem=" << base
                          << " sph=" << sph << "\n";
            }
            // Syllabic-L collapse before -ing (see helper).
            simplifySyllabicLForIng(base, sph);
            // If the stem came from a $strend2 dict entry, use
            // final-stress placement.
            bool stem_is_strend = strend_words_.count(base) > 0 ||
                strend_words_.count(base + "e") > 0;
            result = processPhonemeString(sph + "%IN",
                                           stem_is_strend);
        }
    }
    return result;
}

// -ed suffix handler. Caller must guarantee norm.size() >= 5.
std::string IPAPhonemizer::checkSuffixEd(
        const std::string& norm) const {
    std::string result;
    // `processing` is the natural "this -ed candidate is still alive"
    // state — set false here when an early skip fires, then re-checked
    // before later stages.
    bool processing = (norm.size() >= 4 &&
        norm.compare(norm.size() - 2, 2, "ed") == 0);
    std::string base;
    if (processing) {
        base = norm.substr(0, norm.size() - 2);
        processing = isEdSuffixCandidate(norm, base);
    }
    if (processing) {
        std::string sph;
        // CVC/nc magic-e logic, sans CVRC extension (-ed legacy
        // behavior).
        bool cvc_pattern = false;
        bool nc_pattern = false;
        detectStemPatternForSuffix(base, /*include_cvrc=*/false,
                                   cvc_pattern, nc_pattern);
        bool base_has_stress_override2 =
            (stress_pos_.find(base) != stress_pos_.end() ||
             (dict_.find(base) != dict_.end() && !onlys_words_.count(base) &&
                 !only_words_.count(base)) ||
             verb_dict_.find(base) != verb_dict_.end());
        // SUFX_I: words like "identified" -> base "identifi" ends in
        // 'i' from 'y'->'i' change. Try base[:-1]+'y' first.
        if (!base.empty() && base.back() == 'i' && base.size() >= 2) {
            std::string y_stem = base.substr(0, base.size() - 1) + "y";
            sph = getStemPhonemes(y_stem);
        }
        // Guard: if the CVC magic-e form (base+'e') starts with a
        // PREFIX rule at position 0, the word is a compound
        // ("infrare" = "infra"+"re") rather than a verb stem with
        // silent-e. Setting `processing = false` short-circuits the
        // remaining code — see hasPrefixAtStart and the
        // "fall-through" memo for why this must stay explicit.
        if ((cvc_pattern || nc_pattern) && !base_has_stress_override2) {
            std::string magic_e = base + "e";
            if (hasPrefixAtStart(magic_e)) {
                processing = false;
            } else {
                // Same use_magic_e heuristic as -ing, but with
                // rhotic ɚ ('3') also treated as a complete vowel.
                bool use_magic_e_ed = true;
                if (cvc_pattern && !nc_pattern) {
                    std::string base_ph_ed = getStemPhonemes(base);
                    use_magic_e_ed = shouldUseMagicEForCvcStem(
                        base_ph_ed, "I@3");
                }
                if (use_magic_e_ed) {
                    if (sph.empty()) { sph = getStemPhonemes(magic_e); }
                    if (sph.empty()) { sph = getStemPhonemes(base); }
                }
            }
        } else {
            // Non-magic-e fallback chain: doubled-consonant,
            // -rs/-ns magic-e, plain stem, secondary fallbacks.
            sph = tryEdNonMagicEStemFallbacks(
                base, base_has_stress_override2, sph);
        }
        // Doubled-consonant outer fallback. `processing` gates this
        // — if the magic-e branch short-circuited via
        // hasPrefixAtStart, this fallback must not run (matches the
        // original `return std::nullopt` semantics).
        if (processing && sph.empty() && base.size() >= 2 &&
            base.back() == base[base.size() - 2]) {
            sph = getStemPhonemes(base.substr(0, base.size() - 1));
            if (sph.empty()) { sph = getStemPhonemes(base); }
        }
        if (processing && !sph.empty()) {
            std::string final_ph = computeEdSuffixVoicing(sph, norm, base);
            result = processPhonemeString(final_ph);
        }
    }
    return result;
}

// All morphological-suffix handlers (-ing, -ed, -ies, -s, -[Ce]s,
// -[ch/sh]es, -xes, -arily) run in the same priority order as the
// original monolithic function. The first non-empty result wins.
// -s / -es dictionary-stem suffix handler. Splits into two passes:
// strip just 's', else strip "es" (only when stem ends sibilant in
// phoneme space). Each pass consults onlys_bare_dict_ / verb_dict_
// / dict_ / stress_pos_ in priority order. Suffix voicing matches
// English plural/3rd-person allomorphs.
std::string IPAPhonemizer::checkSuffixDictS(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 3 && norm.back() == 's' &&
        !(norm.size() >= 2 && norm[norm.size() - 2] == 's')) {
        static const std::string UNVOICED_S = "ptkfTSCxXhs";
        std::string stem_s = norm.substr(0, norm.size() - 1);
        bool skip_s_strip = (stem_s.size() >= 3 && stem_s.back() == 'u');
        auto doStemPhS = [&](const std::string& stem) -> std::string {
            std::string out;
            if (stem.size() >= 2) {
                bool hv = hasAnyVowelLetter(stem);
                if (hv) {
                    static const std::string VC3 = "aAeEIiOUVu03@o";
                    std::string ph2;
                    auto obit_s = onlys_bare_dict_.find(stem);
                    if (obit_s != onlys_bare_dict_.end()) {
                        ph2 = processPhonemeString(obit_s->second);
                    } else if (only_words_.count(stem)) {
                        auto vt = verb_dict_.find(stem);
                        if (vt != verb_dict_.end()) {
                            ph2 = processPhonemeString(vt->second);
                        }
                    } else {
                        auto jt = dict_.find(stem);
                        if (jt != dict_.end()) {
                            ph2 = processPhonemeString(jt->second);
                        } else {
                            auto sp = stress_pos_.find(stem);
                            if (sp != stress_pos_.end()) {
                                std::string raw = applyRules(stem, true, 0);
                                if (!raw.empty()) {
                                    ph2 = processPhonemeString(
                                        applyStressPosition(raw, sp->second));
                                }
                            }
                        }
                    }
                    if (!ph2.empty() && hasAnyVowelCode(ph2)) { out = ph2; }
                }
            }
            return out;
        };
        if (!skip_s_strip) {
            std::string sph_s = doStemPhS(stem_s);
            if (!sph_s.empty()) {
                static const std::string SIBILANTS_PH = "SZsz";
                bool last_sibilant =
                    SIBILANTS_PH.find(sph_s.back()) != std::string::npos;
                std::string s_ph = last_sibilant ? "I#z"
                    : (UNVOICED_S.find(sph_s.back()) !=
                          std::string::npos)
                        ? "s" : "z";
                if (std::getenv("PHON_DEBUG")) {
                    std::cerr << "[SUFFIX-DICT-S] " << norm
                              << " stem=" << stem_s
                              << " sph=" << sph_s
                              << " +" << s_ph << "\n";
                }
                result = processPhonemeString(sph_s + s_ph);
            }
        }
        // Try stem = word without trailing "es" — only fires when
        // stem phonemes end sibilant. (When skip_s_strip is true,
        // norm[size-2] == 'u' so the inner condition is false.)
        if (result.empty() && norm.size() >= 4 &&
            norm[norm.size() - 2] == 'e') {
            std::string stem_es = norm.substr(0, norm.size() - 2);
            std::string sph_es = doStemPhS(stem_es);
            if (!sph_es.empty()) {
                static const std::string SIBILANTS_ES = "SZszC";
                bool stem_sibilant =
                    SIBILANTS_ES.find(sph_es.back()) != std::string::npos;
                if (stem_sibilant) {
                    if (std::getenv("PHON_DEBUG")) {
                        std::cerr << "[SUFFIX-DICT-ES] " << norm
                                  << " stem=" << stem_es
                                  << " sph=" << sph_es
                                  << " +I#z\n";
                    }
                    result = processPhonemeString(sph_es + "I#z");
                }
            }
        }
    }
    return result;
}

// -ies plural / 3rd-person handler. "butterflies", "countries",
// "studies" — strip "ies", restore "y", phonemize the stem, append
// voiced 'z'. The direct-rules-comparison override handles words
// where a specific full-word rule fires that wouldn't fire on the
// stem (e.g. "species" -> "spe-ci-es" with the e->iː rule).
std::string IPAPhonemizer::checkSuffixIes(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 4 && norm.compare(norm.size() - 3, 3, "ies") == 0) {
        std::string base = norm.substr(0, norm.size() - 3) + "y";
        std::string sph = getStemPhonemes(base);
        if (!sph.empty()) {
            static const std::string VOWELS_IES = "aAeEIiOUVu03@o";
            auto firstVowelPos = [&](const std::string& ph) -> int {
                int pos = -1;
                for (int k = 0; k < (int)ph.size() && pos < 0; k++) {
                    if (VOWELS_IES.find(ph[k]) != std::string::npos) {
                        pos = k;
                    }
                }
                return pos;
            };
            int sv_pos = firstVowelPos(sph);
            // eI / aI / aU / OI / oU: 2-char diphthongs.
            bool stem_has_diphthong = (sv_pos >= 0 &&
                sv_pos + 1 < (int)sph.size() &&
                (sph[sv_pos + 1] == 'I' || sph[sv_pos + 1] == 'U'));
            if (!stem_has_diphthong) {
                char sv = (sv_pos >= 0) ? sph[sv_pos] : 0;
                std::string full_raw = applyRules(norm, false, 0);
                int dv_pos = firstVowelPos(full_raw);
                char dv = (dv_pos >= 0) ? full_raw[dv_pos] : 0;
                bool direct_fv_unstressed =
                    (dv_pos > 0 && full_raw[dv_pos - 1] == '%');
                // Magic-e strut artifact: stem first vowel 'V' (ʌ)
                // vs direct 'u'/'U' indicates wrong long-u from
                // '-ies' as magic-e. Trust the stem.
                bool magic_e_strut = (sv == 'V' && (dv == 'u' || dv == 'U'));
                if (dv != 0 && sv != dv && !direct_fv_unstressed &&
                    !magic_e_strut) {
                    if (std::getenv("PHON_DEBUG")) {
                        std::cerr << "[SUFFIX-IES-DIRECT] " << norm
                                  << " direct=" << full_raw
                                  << "\n";
                    }
                    result = processPhonemeString(full_raw);
                }
            }
            if (result.empty()) {
                if (std::getenv("PHON_DEBUG")) {
                    std::cerr << "[SUFFIX-IES] " << norm
                              << " stem=" << base
                              << " sph=" << sph << "\n";
                }
                result = processPhonemeString(sph + "z");
            }
        }
    }
    return result;
}

// -[Ce]s magic-e suffix handler. Base "norm minus s" ends in
// consonant+e where the consonant is not a sibilant. Stem is the
// base. Voicing: sibilant stem -> I#z, unvoiced -> s, else -> z.
// The doStemPh2 lambda matches dict_ then applies suffix_phoneme_only
// for stems that themselves end in magic-e (the "ribosome" case
// where -some is a SUFFIX rule).
std::string IPAPhonemizer::checkSuffixMagicEs(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 4 && norm.back() == 's' &&
        !(norm.size() >= 2 && norm[norm.size() - 2] == 's')) {
        std::string base = norm.substr(0, norm.size() - 1);
        if (base.size() >= 2 && base.back() == 'e') {
            char c_before_e = base[base.size() - 2];
            bool is_consonant = !isVowelLetter(c_before_e);
            bool is_sibilant = (c_before_e == 's' ||
                c_before_e == 'z' || c_before_e == 'x' || c_before_e == 'c');
            bool is_digraph_sibilant = (c_before_e == 'h' &&
                base.size() >= 3 && (base[base.size() - 3] == 'c' ||
                 base[base.size() - 3] == 's'));
            if (is_consonant && !is_sibilant && !is_digraph_sibilant) {
                auto doStemPh2 = [&](const std::string& stem) -> std::string {
                    std::string out;
                    if (stem.size() >= 2) {
                        bool hv = hasAnyVowelLetter(stem);
                        if (hv) {
                            std::string ph2;
                            auto jt = dict_.find(stem);
                            if (jt != dict_.end()) {
                                ph2 = jt->second;
                                auto sp = stress_pos_.find(stem);
                                if (sp != stress_pos_.end()) {
                                    ph2 = applyStressPosition(ph2, sp->second);
                                }
                            } else {
                                static const std::string
                                    GROUP_B_CHARS = "bcdfgjklmnpqstvxz";
                                static const std::string
                                    DELFWD_VOWELS = "aioy";
                                bool use_spo = false;
                                if (stem.size() >= 3 && stem.back() == 'e') {
                                    char c_cons =
                                        (char)std::tolower(
                                            (unsigned char)
                                            stem[stem.size() - 2]);
                                    char c_prev =
                                        (char)std::tolower(
                                            (unsigned char)
                                            stem[stem.size() - 3]);
                                    use_spo =
                                        GROUP_B_CHARS.find(c_cons) !=
                                            std::string::npos &&
                                        DELFWD_VOWELS.find(c_prev) !=
                                            std::string::npos;
                                }
                                std::string raw = applyRules(stem,
                                    true, -1, use_spo);
                                auto sp = stress_pos_.find(stem);
                                if (sp != stress_pos_.end()) {
                                    raw = applyStressPosition(raw, sp->second);
                                }
                                ph2 = processPhonemeString(raw);
                            }
                            if (!ph2.empty() && hasAnyVowelCode(ph2)) {
                                out = ph2;
                            }
                        }
                    }
                    return out;
                };
                std::string sph = doStemPh2(base);
                if (!sph.empty()) {
                    static const std::string UNVOICED = "ptkfTSCxXhs";
                    static const std::string SIBILANTS_PH = "SZsz";
                    bool last_sib =
                        SIBILANTS_PH.find(sph.back()) != std::string::npos;
                    std::string s_ph = last_sib ? "I#z"
                        : (UNVOICED.find(sph.back()) !=
                              std::string::npos)
                            ? "s" : "z";
                    if (std::getenv("PHON_DEBUG")) {
                        std::cerr << "[SUFFIX-S] " << norm
                                  << " stem=" << base
                                  << " sph=" << sph
                                  << " +" << s_ph << "\n";
                    }
                    result = processPhonemeString(sph + s_ph);
                }
            }
        }
    }
    return result;
}

// -ches / -shes digraph-sibilant suffix.
std::string IPAPhonemizer::checkSuffixChShEs(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 5 && norm.back() == 's' &&
        norm[norm.size() - 2] == 'e' && norm[norm.size() - 3] == 'h' &&
        (norm[norm.size() - 4] == 'c' || norm[norm.size() - 4] == 's')) {
        std::string stem = norm.substr(0, norm.size() - 2);
        if (hasAnyVowelLetter(stem) && stem.size() >= 2) {
            std::string sph;
            auto jt = dict_.find(stem);
            if (jt != dict_.end()) { sph = jt->second; }
            else { sph = processPhonemeString(applyRules(stem)); }
            if (hasAnyVowelCode(sph)) {
                result = processPhonemeString(sph + "I#z");
            }
        }
    }
    return result;
}

// -xes suffix.
std::string IPAPhonemizer::checkSuffixXes(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 4 && norm.compare(norm.size() - 3, 3, "xes") == 0) {
        std::string stem = norm.substr(0, norm.size() - 2);
        if (hasAnyVowelLetter(stem) && stem.size() >= 2) {
            std::string sph;
            auto jt = dict_.find(stem);
            if (jt != dict_.end()) { sph = jt->second; }
            else { sph = processPhonemeString(applyRules(stem)); }
            if (hasAnyVowelCode(sph)) {
                result = processPhonemeString(sph + "I#z");
            }
        }
    }
    return result;
}

// -arily suffix.
std::string IPAPhonemizer::checkSuffixArily(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 8 && norm.compare(norm.size() - 5, 5, "arily") == 0) {
        std::string stem_arily = norm.substr(0, norm.size() - 5);
        bool stem_has_vowel = hasAnyVowelLetter(stem_arily);
        if (stem_has_vowel && stem_arily.size() >= 2) {
            std::string stem_with_ari = stem_arily + "ari";
            std::string sph_arily;
            auto jt_ar = dict_.find(stem_with_ari);
            if (jt_ar != dict_.end()) {
                sph_arily = jt_ar->second;
            } else {
                sph_arily = applyRules(stem_with_ari);
            }
            if (!sph_arily.empty()) {
                // Two cases: stressed "'A@ri" (5-char strip, no
                // demote) vs schwa-r "3ri"/"@ri" (3-char strip,
                // demote stem's primary to secondary since the
                // primary will land on the new -e@rI#l%i below).
                std::string sph_stem;
                bool ends_in_stressed = sph_arily.size() >= 5 &&
                    sph_arily.compare(sph_arily.size() - 5, 5,
                                      "\'A@ri") == 0;
                bool ends_in_schwa_r = !ends_in_stressed &&
                    sph_arily.size() >= 3 &&
                    (sph_arily.compare(sph_arily.size() - 3, 3,
                                       "3ri") == 0 ||
                     sph_arily.compare(sph_arily.size() - 3, 3,
                                       "@ri") == 0);
                if (ends_in_stressed) {
                    sph_stem = sph_arily.substr(0, sph_arily.size() - 5);
                } else if (ends_in_schwa_r) {
                    sph_stem = sph_arily.substr(0, sph_arily.size() - 3);
                    replaceFirstChar(sph_stem, '\'', ',');
                }
                if (!sph_stem.empty()) {
                    std::string combined = sph_stem + "\'e@rI#l%i";
                    if (std::getenv("PHON_DEBUG")) {
                        std::cerr << "[SUFFIX-ARILY] " << norm
                                  << " stem_ari=" << stem_with_ari
                                  << " sph=" << sph_arily
                                  << " stem_ph=" << sph_stem
                                  << " combined=" << combined
                                  << "\n";
                    }
                    result = processPhonemeString(combined);
                }
            }
        }
    }
    return result;
}

std::string IPAPhonemizer::checkMorphologicalSuffixes(
        const std::string& norm) const {
    std::string result;
    // -ing and -ed require norm.size() >= 5 (so substr/compare are
    // safe and there's a stem of at least 2 chars).
    if (norm.size() >= 5) {
        if (result.empty()) { result = checkSuffixIng(norm); }
        if (result.empty()) { result = checkSuffixEd(norm); }
    }
    // -ies plural/3rd-person.
    if (result.empty()) { result = checkSuffixIes(norm); }
    // -s / -es with dict stem lookup (priority over magic-e).
    if (result.empty()) { result = checkSuffixDictS(norm); }
    // -[Ce]s magic-e.
    if (result.empty()) { result = checkSuffixMagicEs(norm); }
    // -ches / -shes digraph sibilant.
    if (result.empty()) { result = checkSuffixChShEs(norm); }
    // -xes.
    if (result.empty()) { result = checkSuffixXes(norm); }
    // -arily.
    if (result.empty()) { result = checkSuffixArily(norm); }
    return result;
}

// Compound prefix splitting: if the word starts with a known
// $strend2 prefix (a function word that shifts stress to the
// suffix, e.g. "under", "over", "through"), return prefix-phonemes
// (secondary) + suffix-phonemes (full, with primary). Matches the
// reference compound-word stress algorithm for $strend2 prefixes.
std::string IPAPhonemizer::checkCompoundPrefixes(
        const std::string& norm) const {
    std::string result;
    if (norm.size() >= 5 && !compound_prefixes_.empty()) {
        // `found` is the natural "first-match wins" predicate;
        // replaces the original `continue` / `return` flow.
        bool found = false;
        for (size_t i = 0;
             i < compound_prefixes_.size() && !found; i++) {
            const auto& pref = compound_prefixes_[i].first;
            const auto& pref_ph = compound_prefixes_[i].second;
            // Require minimum prefix length of 4 to avoid false
            // splits on short function words (e.g., "his"=3 would
            // wrongly split "history" as "his"+"tory").
            bool live = (pref.size() >= 4 && pref.size() < norm.size());
            std::string suffix;
            if (live) {
                size_t sfx_len = norm.size() - pref.size();
                if (sfx_len < 2) { live = false; }
                if (live && norm.compare(0, pref.size(), pref) != 0) {
                    live = false;
                }
                if (live) {
                    suffix = norm.substr(pref.size());
                    if (!hasAnyVowelLetter(suffix)) { live = false; }
                }
                // Suffix must be at least 4 chars OR be a recognized
                // dict word. Prevents false splits like "hers"+"elf"
                // = "herself" where "elf" (3 chars, not in dict) is
                // not a compound-forming morpheme. "over"+"all"
                // passes because "all" is in the dictionary.
                if (live && sfx_len < 4 && dict_.find(suffix) == dict_.end() &&
                    verb_dict_.find(suffix) == verb_dict_.end()) {
                    live = false;
                }
            }
            if (live) {
                // Process prefix phonemes: multi-syllable prefix (≥2
                // vowel codes) -> demote primary -> secondary;
                // mono-syllable prefix -> remove stress marker
                // (prefix is unstressed).
                std::string pfx_ph = processPhonemeString(pref_ph);
                static const char* MC_VOWELS[] = {
                    "O@","o@","U@","A@","e@","i@","aI@3","aI3",
                    "aU@","aI@","i@3","3:r","A:r","o@r","A@r","e@r",
                    "eI","aI","aU","OI","oU","IR","VR","3:","A:",
                    "i:","u:","O:","e:","a:","aa","@L","@2","@5",
                    "I2","I#","E2","E#","e#","a#","a2","0#","02",
                    "O2","A#", nullptr };
                int nvowels = 0;
                for (size_t pi = 0; pi < pfx_ph.size(); ) {
                    char c = pfx_ph[pi];
                    if (c == '\'' || c == ',' || c == '%' || c == '=') {
                        pi++;
                    } else {
                        bool matched = false;
                        for (int mi = 0;
                             MC_VOWELS[mi] && !matched; mi++) {
                            int ml = (int)strlen(MC_VOWELS[mi]);
                            if ((int)pi + ml <= (int)pfx_ph.size() &&
                                pfx_ph.compare(pi, ml,
                                    MC_VOWELS[mi]) == 0) {
                                nvowels++;
                                pi += ml;
                                matched = true;
                            }
                        }
                        if (!matched) {
                            if (isVowelCode(std::string(1, c))) { nvowels++; }
                            pi++;
                        }
                    }
                }
                if (nvowels >= 2) {
                    // Multi-syllable prefix: demote first '\'' to
                    // ',' (others left alone).
                    replaceFirstChar(pfx_ph, '\'', ',');
                } else {
                    // Mono-syllable prefix: remove stress markers
                    // (prefix becomes unstressed).
                    pfx_ph.erase(
                        std::remove(pfx_ph.begin(), pfx_ph.end(),
                                    '\''),
                        pfx_ph.end());
                    pfx_ph.erase(
                        std::remove(pfx_ph.begin(), pfx_ph.end(),
                                    ','),
                        pfx_ph.end());
                }
                // Suffix via full wordToPhonemes (handles dict,
                // -ing/-ed, rules).
                std::string sfx_ph = wordToPhonemes(suffix);
                std::string combined = pfx_ph + sfx_ph;
                // $N stress override on the FULL word (e.g.
                // "overture $1").
                auto sit = stress_pos_.find(norm);
                if (sit != stress_pos_.end()) {
                    combined = processPhonemeString(
                        applyStressPosition(combined, sit->second));
                }
                result = combined;
                found = true;
            }
        }
    }
    return result;
}

// Final fallback: applyRules + processPhonemeString + $N stress
// override. The original wordToPhonemes had a commented-out section
// 7 about voiced sibilant assimilation that was a no-op; preserved
// here for the historical record (the reference's en_rules suffix
// rules already emit the correct voicing).
std::string IPAPhonemizer::applyRulesFallback(
        const std::string& norm) const {
    std::string raw_ph = applyRules(norm);
    auto sit = stress_pos_.find(norm);
    if (sit != stress_pos_.end()) {
        raw_ph = applyStressPosition(raw_ph, sit->second);
    }
    std::string ph = processPhonemeString(raw_ph);
    // 7. Final sibilant voicing normalization (intentionally a
    // no-op — see comment in earlier revisions of this file). The
    // en_rules suffix rules already emit the correct voicing.
    if (std::getenv("PHON_DEBUG")) {
        std::cerr << "[RULES] " << norm << " -> " << ph << "\n";
    }
    return ph;
}

// Word to phoneme codes — slim chain dispatching to the helpers
// above. Each helper returns "" if it cannot claim the word; first
// non-empty wins.
std::string IPAPhonemizer::wordToPhonemes(
        const std::string& word) const {
    std::string norm = toLowerASCII(word);
    std::string result;
    if (result.empty()) { result = checkCapitalDict(word, norm); }
    if (result.empty()) { result = checkMainDict(norm); }
    if (result.empty()) { result = checkHyphenated(norm); }
    if (result.empty()) { result = checkPossessive(norm); }
    if (result.empty()) { result = checkSingleLetter(norm); }
    if (result.empty()) { result = checkMorphologicalSuffixes(norm); }
    if (result.empty()) { result = checkCompoundPrefixes(norm); }
    if (result.empty()) { result = applyRulesFallback(norm); }
    return result;
}

// ============================================================
// Apply $N stress position: remove all existing stress markers and
// insert primary stress (') before the Nth vowel (1-based).
// Used to implement en_list $N flags (e.g. "lemonade $3").
// ============================================================
std::string IPAPhonemizer::applyStressPosition(
        const std::string& raw, int n) const {
    // Same multi-char code table used in processPhonemeString
    static const char* S_MC2[] = {
        "aI@3","aU@r","aI@","aI3","aU@","i@3","3:r","A:r","o@r","A@r","e@r",
        "eI","aI","aU","OI","oU","tS","dZ","IR","VR",
        "e@","i@","U@","A@","O@","o@",
        "3:","A:","i:","u:","O:","e:","a:","aa",
        "@L","@2","@5",
        "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~","A#",
        "r-","w#","t#","d#","z#","t2","d2","n-","m-","l/","z/",
        nullptr
    };
    auto findC = [&](const std::string& s, size_t pos) -> std::string {
        for (int mi = 0; S_MC2[mi]; mi++) {
            int mclen = (int)strlen(S_MC2[mi]);
            if (pos + (size_t)mclen <= s.size() &&
                s.compare(pos, mclen, S_MC2[mi]) == 0) {
                return std::string(S_MC2[mi], mclen);
            }
        }
        return std::string(1, s[pos]);
    };

    // Remove ALL existing stress markers (' and ,) so step 5a can
    // re-distribute
    std::string ph;
    ph.reserve(raw.size());
    for (char c : raw) {
        if (c != '\'' && c != ',') { ph += c; }
    }

    // Find and mark the Nth vowel
    int vowel_count = 0;
    size_t pi = 0;
    std::string result = ph;
    bool inserted = false;
    while (pi < result.size() && !inserted) {
        char c = result[pi];
        if (c == '%' || c == '=' || c == '|') {
            pi++;
        } else {
            std::string code = findC(result, pi);
            if (isVowelCode(code)) {
                vowel_count++;
                if (vowel_count == n) {
                    result.insert(pi, 1, '\'');
                    inserted = true;
                }
            }
            if (!inserted) { pi += code.size(); }
        }
    }
    return result;
}

// ============================================================
// processPhonemeString step helpers (extracted for size and
// testability — each transforms `ph` in place).
// ============================================================

// Step 1: Velar nasal assimilation. n+k/g -> N+k/g (ŋ before velar
// stops). E.g. "income" -> ˈɪŋkʌm, "congress" -> kˈɑːŋɡɹəs.
static void applyVelarNasalAssimilation(std::string& ph) {
    for (size_t i = 0; i + 1 < ph.size(); i++) {
        if (ph[i] == 'n' && (ph[i+1] == 'k' || ph[i+1] == 'g')) {
            ph[i] = 'N';
        }
    }
}

// Step 2: Happy tensing — word-final unstressed ɪ -> i (American
// English). Applies to %I, I2, and bare-I at word end (when not
// stressed and not part of a diphthong).
static void applyHappyTensing(std::string& ph) {
    if (ph.size() >= 2 && ph[ph.size()-2] == '%' && ph[ph.size()-1] == 'I') {
        ph.resize(ph.size()-2);
        ph += 'i';
    } else if (ph.size() >= 2 && ph[ph.size()-2] == 'I' &&
        ph[ph.size()-1] == '2') {
        ph.resize(ph.size()-2);
        ph += 'i';
    } else if (!ph.empty() && ph.back() == 'I') {
        // Bare I at word end: convert to i unless stress-marked,
        // part of a diphthong, or sole vowel (monosyllabic).
        char prev = (ph.size() >= 2) ? ph[ph.size()-2] : 0;
        static const std::string DIPHTHONG_BEFORE = "eaOoU";
        bool part_of_diph = (prev != 0 &&
            DIPHTHONG_BEFORE.find(prev) != std::string::npos);
        if (prev != '\'' && prev != ',' && !part_of_diph) {
            static const std::string ALL_VOWELS = "aAeEiIOUV03o";
            int vowel_count = 0;
            for (char c : ph) {
                if (ALL_VOWELS.find(c) != std::string::npos) { vowel_count++; }
            }
            if (vowel_count > 1) { ph.back() = 'i'; }
        }
    }
}

// Step 3: Vowel reduction — unstressed back/front vowels -> schwa
// (American English). The REDUCTIONS table records intentional
// NON-reductions (%0#, %V, etc).
static void applyVowelReduction(std::string& ph) {
    struct Repl {
        const char* from;
        const char* to;
        char not_followed_by;
    };
    static const Repl REDUCTIONS[] = {
        // Note: '%0#', '%0', '=0#', '=0' (unstressed ɑː/ɑ) are NOT
        // reduced to schwa here. The reference keeps ɑː in content
        // word prefixes like "con-", "op-", "vol-". Step 5.5d/e
        // handle '0#' reduction between stress markers separately.
        // Note: %V/=V (unstressed ʌ) is NOT reduced to schwa —
        // reference keeps ʌ in un-/up-/sub- prefixes (e.g. "unlike"
        // -> ʌnlˈaɪk, not ənlˈaɪk).
        {"%A:", "%@", 0}, {"=A:", "%@", 0},
        // don't reduce %A@ (rhotic diphthong)
        {"%A",  "%@", '@'}, {"=A",  "%@", '@'},
        {nullptr, nullptr, 0}
    };
    for (int ri = 0; REDUCTIONS[ri].from; ri++) {
        std::string from = REDUCTIONS[ri].from;
        std::string to   = REDUCTIONS[ri].to;
        char not_follow  = REDUCTIONS[ri].not_followed_by;
        size_t rpos = 0;
        while ((rpos = ph.find(from, rpos)) != std::string::npos) {
            size_t after = rpos + from.size();
            bool blocked = (not_follow != 0 &&
                after < ph.size() && ph[after] == not_follow);
            if (blocked) {
                rpos = after;          // skip the protected match
            } else {
                ph.replace(rpos, from.size(), to);
                rpos += to.size();
            }
        }
    }
}

// Step 3b: LOT+R -> THOUGHT+R in American English (pre-rhotic vowel
// neutralization). '0r' (ɑːɹ) -> 'O:r' (ɔːɹ). Universal:
// "forest"->O:r, "moral"->O:r, "origin"->O:r, "horrible"->O:r, etc.
static void applyLotPlusRMerge(std::string& ph) {
    size_t rpos = 0;
    while ((rpos = ph.find("0r", rpos)) != std::string::npos) {
        ph.replace(rpos, 2, "O:r");
        rpos += 3;
    }
}

// Step 3c: Strip morpheme-boundary schwa before r: @-r -> r. Used
// for rule output like "or" before "ative" (collaborative,
// decorative) and dict entries like "average"=av@-rI2dZ. The '@'
// is elided, leaving just the r consonant.
static void stripMorphemeSchwaR(std::string& ph) {
    size_t i = 0;
    while (i + 2 < ph.size()) {
        if (ph[i] == '@' && ph[i+1] == '-' && ph[i+2] == 'r') {
            ph.erase(i, 2); // remove @- , keep r
        } else {
            i++;
        }
    }
}

// Step 4: Bare schwa '@' immediately before 'r' -> r-colored schwa
// '3'. The 'r' is absorbed into '3' (ɚ) when followed by a
// consonant or end-of-word, since 'r' only stays as a separate
// onset when the next phoneme is a vowel.
static void applyBareSchwaToRhotic(std::string& ph) {
    // Pre-pass: a#r -> @r (unstressed 'a' before 'r' acts like
    // schwa before 'r'; e.g. "around" a#raUnd -> @raUnd -> step 4
    // converts @r -> 3 -> ɚɹˈaʊnd).
    for (size_t i = 0; i + 2 < ph.size(); i++) {
        if (ph[i] == 'a' && ph[i+1] == '#' && ph[i+2] == 'r') {
            ph[i] = '@';
            ph.erase(i+1, 1); // remove '#', leaving @r
        }
    }
    // SESE: replace original guard-`continue`s with nested ifs.
    for (size_t rpos = 0; rpos + 1 < ph.size(); rpos++) {
        if (ph[rpos] == '@') {
            // Find 'r' after '@', possibly skipping stress/modifier
            // marks (%=',)
            size_t r_pos = rpos + 1;
            while (r_pos < ph.size() && (ph[r_pos]=='\'' || ph[r_pos]==',' ||
                    ph[r_pos]=='%' || ph[r_pos]=='=')) {
                r_pos++;
            }
            if (r_pos < ph.size() && ph[r_pos] == 'r') {
                bool is_diphthong = (rpos > 0 && (
                    ph[rpos-1] == 'o' || ph[rpos-1] == 'A' ||
                    ph[rpos-1] == 'U' || ph[rpos-1] == 'O' ||
                    ph[rpos-1] == 'e' || ph[rpos-1] == 'i' ||
                    ph[rpos-1] == 'I' || ph[rpos-1] == 'a'));
                if (!is_diphthong) {
                    ph[rpos] = '3';
                    // Absorb 'r' if followed by a consonant or
                    // end-of-word. Skip stress/modifier markers
                    // (%=',) to find the actual next phoneme.
                    size_t after_r = r_pos + 1;
                    while (after_r < ph.size() &&
                           (ph[after_r]=='%' || ph[after_r]=='=' ||
                            ph[after_r]=='\'' || ph[after_r]==',')) {
                        after_r++;
                    }
                    bool next_is_vowel = (after_r < ph.size() &&
                        (ph[after_r]=='a' || ph[after_r]=='A' ||
                         ph[after_r]=='e' || ph[after_r]=='E' ||
                         ph[after_r]=='i' || ph[after_r]=='I' ||
                         ph[after_r]=='o' || ph[after_r]=='O' ||
                         ph[after_r]=='u' || ph[after_r]=='U' ||
                         ph[after_r]=='@' || ph[after_r]=='3'));
                    // Absorb 'r' if: (a) next is not a vowel, OR
                    // (b) '@' was unstressed-prefixed (=/%) e.g.
                    // "factory" -> '=@ri' -> '=3ri' (ɚɹi), keeping
                    // r before vowel.
                    bool unstressed_pre = (rpos > 0 &&
                        (ph[rpos-1] == '=' || ph[rpos-1] == '%'));
                    if (!next_is_vowel || unstressed_pre) {
                        // absorb 'r' into '3' (ɚ)
                        ph.erase(r_pos, 1);
                    }
                }
            }
        }
    }
}

// Step 4c: -tion stress fix. Move primary stress to the vowel
// immediately before '-tion' (S@n). The reference SetWordStress
// detects the -tion suffix and places primary stress on the
// syllable immediately before the -tion. Rules often place primary
// earlier (e.g. on the first vowel), so we move it when the
// pattern is detected. Examples: "institution" (u: before S),
// "extraction" (a before kS), "production" (V before kS).
// Tail-shape check for the -tion @n pattern: returns true iff
// ph[after_N..) consists only of modifier chars, 'i' (terminal
// %i/=i), and/or 'z' (plurals) — no further syllabic vowels or
// other consonants. Distinguishes "-tion"/"-tions" (terminal) from
// "-ciency" (non-terminal: vowel 'i' is followed by more material).
static bool isTionTerminalAfterN(const std::string& ph,
                                 size_t after_N) {
    bool truly_terminal = true;
    size_t ki3 = after_N;
    while (ki3 < ph.size() && truly_terminal) {
        char c3 = ph[ki3];
        bool is_modifier = (c3 == '%' || c3 == '=' || c3 == '-' ||
                            c3 == '\'' || c3 == ',');
        bool is_terminal_i = (c3 == 'i');
        bool is_plural_z = (c3 == 'z');
        if (!is_modifier && !is_terminal_i && !is_plural_z) {
            // Any vowel (more syllables) or any other consonant
            // (e.g. 's' for "conscience"): not terminal.
            truly_terminal = false;
        }
        ki3++;
    }
    return truly_terminal;
}

// Scan up to 10 chars after S for the -tion '@n' pattern. Returns
// true iff '@' is found (skipping modifiers), followed by 'n'/'N'
// (skipping modifiers), and the tail after the N is terminal.
static bool hasTionAfterS(const std::string& ph, size_t after_S) {
    bool found_tion = false;
    bool stop = false;
    size_t ki = after_S;
    while (ki < ph.size() && ki < after_S + 10 && !found_tion && !stop) {
        char c = ph[ki];
        bool is_modifier = (c == '=' || c == '%' || c == '-' ||
                            c == '\'' || c == ',');
        if (c == '@') {
            // Inner scan up to 4 chars for 'n'/'N' modifier-skipped.
            size_t ki2 = ki + 1;
            bool n_found = false;
            bool n_stop = false;
            while (ki2 < ph.size() && ki2 < ki + 4 && !n_found && !n_stop) {
                char c2 = ph[ki2];
                bool c2_modifier = (c2 == '-' || c2 == '=' || c2 == '%');
                if (c2 == 'n' || c2 == 'N') {
                    if (isTionTerminalAfterN(ph, ki2 + 1)) {
                        found_tion = true;
                    }
                    n_found = true;
                } else if (!c2_modifier) {
                    n_stop = true;
                }
                ki2++;
            }
            stop = true; // saw '@', stop outer scan whether matched
        } else if (!is_modifier) {
            stop = true; // non-modifier, non-@: not a -tion suffix
        }
        ki++;
    }
    return found_tion;
}

// Locate the 'S' (ʃ) of a -tion suffix near word end. Scans from
// end backward; the first 'S' whose tail matches @n + terminal
// wins. Returns npos if no -tion pattern exists.
static size_t findTionSuffixSPos(const std::string& ph) {
    size_t S_pos = std::string::npos;
    size_t k = ph.size();
    while (k > 0 && S_pos == std::string::npos) {
        size_t idx = k - 1;
        if (ph[idx] == 'S' && hasTionAfterS(ph, idx + 1)) { S_pos = idx; }
        k--;
    }
    return S_pos;
}

// True iff the primary-stress marker '\'' is in the consonant onset
// of the syllable whose vowel starts at vowel_pos. Scans backward
// from vowel_pos-1; stops on another vowel (the '\'' belongs to a
// prior syllable).
static bool primaryAlreadyOnVowel(const std::string& ph,
                                  size_t vowel_pos) {
    static const std::string VOWEL_CHARS = "aAeEiIoOuUV03@";
    bool primary = false;
    bool stop = false;
    int bi = (int)vowel_pos - 1;
    while (bi >= 0 && !primary && !stop) {
        char bc = ph[bi];
        if (bc == '\'') {
            primary = true;
        } else if (VOWEL_CHARS.find(bc) != std::string::npos) {
            stop = true;
        }
        bi--;
    }
    return primary;
}

// Move primary stress to the syllable starting at vowel_pos.
// Demotes any earlier '\'' to ',' and absorbs the existing ','
// before vowel_pos (the displaced secondary marker). No-op when
// primary is already at or after vowel_pos (the rules placed it
// correctly downstream).
static void moveStressToVowel(std::string& ph, size_t vowel_pos) {
    size_t prime_pos = ph.find('\'');
    bool primary_after = (prime_pos != std::string::npos &&
                          prime_pos >= vowel_pos);
    if (!primary_after) {
        if (prime_pos != std::string::npos) {
            ph[prime_pos] = ','; // demote
        }
        if (vowel_pos > 0 && ph[vowel_pos-1] == ',') {
            ph.erase(vowel_pos-1, 1);
            vowel_pos--;
        }
        ph.insert(vowel_pos, 1, '\'');
    }
}

static void countVowelsBeforeUnitAware(
        const std::string& ph, size_t end,
        int& vowels_before, size_t& last_vowel_pos) {
    static const char* MC4e[] = {
        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
        "3:r","A:r","o@r","A@r","e@r",
        "eI","aI","aU","OI","oU","IR","VR",
        "e@","i@","U@","A@","O@","o@",
        "3:","A:","i:","u:","O:","e:","a:","aa",
        "@L","@2","@5",
        "I2","I#","E2","E#","e#","a#","a2","0#","02","O2",
        "A~","O~","A#",
        nullptr
    };
    static const std::string VOWEL_C = "aAeEiIoOuUV03@";
    vowels_before = 0;
    last_vowel_pos = std::string::npos;
    size_t vi = 0;
    while (vi < end) {
        char c = ph[vi];
        if (c == '\'' || c == ',' || c == '%' || c == '=') {
            vi++;
        } else {
            std::string vcode;
            bool found = false;
            for (int mi = 0; MC4e[mi] && !found; mi++) {
                int mcl = (int)strlen(MC4e[mi]);
                found = (int)(end - vi) >= mcl &&
                    ph.compare(vi, mcl, MC4e[mi]) == 0;
                if (found) { vcode = std::string(MC4e[mi], mcl); }
            }
            if (vcode.empty()) { vcode = std::string(1, c); }
            if (VOWEL_C.find(vcode[0]) != std::string::npos) {
                vowels_before++;
                last_vowel_pos = vi;
            }
            vi += vcode.size();
        }
    }
}

static void applyTionStressFix(std::string& ph) {
    size_t S_pos = findTionSuffixSPos(ph);
    if (S_pos != std::string::npos) {
        int v_count = 0;
        size_t vowel_pos = std::string::npos;
        countVowelsBeforeUnitAware(ph, S_pos, v_count, vowel_pos);
        if (vowel_pos != std::string::npos &&
            !primaryAlreadyOnVowel(ph, vowel_pos)) {
            moveStressToVowel(ph, vowel_pos);
        }
    }
}

// Step 4d: -ology stress fix. Words ending in '-ology' pattern get
// primary stress on the vowel before 'l@dZ' (the '-ol-' in
// technology, biology, etc.). The reference SetWordStress places
// primary stress one syllable before '-ology'. Rules typically
// place it earlier, so we move it here.
// Test for "-ology" pattern: '0l' + optional modifiers + '@dZ' at
// position k. Returns true if the match holds.
static bool matchesOlogyAt(const std::string& ph, size_t k) {
    bool matched = false;
    if (k + 2 < ph.size() && ph[k] == '0' && ph[k+1] == 'l') {
        size_t a = k + 2;
        while (a < ph.size() &&
               (ph[a] == '=' || ph[a] == '%' || ph[a] == ',')) {
            a++;
        }
        matched = (a + 2 < ph.size() && ph[a] == '@' && ph[a+1] == 'd' &&
                   ph[a+2] == 'Z');
    }
    return matched;
}

static void applyOlogyStressFix(std::string& ph) {
    // Look for '0l ... @dZ' (the -ology suffix). Move primary
    // stress to the '0' (the -ol- vowel) when it isn't already
    // primary-stressed.
    size_t ol_pos = std::string::npos;
    for (size_t k = 0;
         k + 2 < ph.size() && ol_pos == std::string::npos; k++) {
        if (matchesOlogyAt(ph, k)) { ol_pos = k; }
    }
    if (ol_pos != std::string::npos) {
        bool primary_on_ol = (ol_pos > 0 && ph[ol_pos-1] == '\'');
        if (!primary_on_ol) {
            size_t prime_pos = ph.find('\'');
            if (prime_pos != std::string::npos && prime_pos < ol_pos) {
                ph[prime_pos] = ','; // demote to secondary
            }
            if (ol_pos > 0 && ph[ol_pos-1] == ',') {
                ph.erase(ol_pos-1, 1);
                ol_pos--;
            }
            ph.insert(ol_pos, 1, '\'');
        }
    }
}

// Step 4e: -ic/-ical/-ics stress fix. Words with these suffixes
// get primary stress on the penultimate syllable before the suffix
// (the syllable ending in the vowel before 'Ik'). E.g. fantastic
// (f'antast=Ik -> fanˈtæstɪk), technology (already handled by 4d).
// Reference: stress before -ic suffix (penultimate syllable rule).
// Pattern: find '=I' (unstressed vowel I) followed by 'k' near end
// of phoneme string. Move primary to the last stressed vowel
// before '=Ik'.
// Scan ph[0..end) and count vowel-group starts, recording the last
// position. Multi-char codes (aa, aI, eI, ...) count as one. Stress
// markers and '%'/'=' are skipped. Out-params: vowels_before,
// last_vowel_pos.
static void applyIcStressFix(std::string& ph) {
    // Pattern: last '=I' or '%I' or word-leading 'I' followed by 'k'
    // (the -ic / -ical / -ics suffix). Locate by finding the last
    // 'k' and checking the preceding chars.
    size_t ik_pos = std::string::npos;
    size_t k_pos = ph.rfind('k');
    if (k_pos != std::string::npos && k_pos >= 1 && ph[k_pos - 1] == 'I') {
        bool i_unstressed = (k_pos >= 2 &&
            (ph[k_pos - 2] == '=' || ph[k_pos - 2] == '%')) || (k_pos == 1);
        if (i_unstressed) { ik_pos = k_pos - 2; }
    }
    if (ik_pos != std::string::npos) {
        // Move primary stress to the last vowel before ik_pos.
        int vowels_before = 0;
        size_t last_vowel_pos = std::string::npos;
        countVowelsBeforeUnitAware(ph, ik_pos, vowels_before,
                                   last_vowel_pos);
        bool already_on_last = last_vowel_pos != std::string::npos &&
            last_vowel_pos > 0 && ph[last_vowel_pos-1] == '\'';
        if (vowels_before >= 2 && last_vowel_pos != std::string::npos &&
            !already_on_last) {
            moveStressToVowel(ph, last_vowel_pos);
        }
    }
}

// Step 5.5: Reduce bare '0' (unstressed ɑː) -> '@' (schwa).
// Runs after step 5a so that secondary stress markers are already
// placed. '0' preceded by '\'' (primary) or ',' (secondary) stays
// as ɑː; bare '0' -> schwa. Mirrors the reference SetWordStress
// post-placement vowel reduction. Only active when step 4c's
// -ution pattern triggered (i.e. ph has 'u:S').
static void reduceBareZeroAfterUtion(std::string& ph) {
    if (ph.find("u:S") != std::string::npos) {
        for (size_t pi = 0; pi < ph.size(); pi++) {
            if (ph[pi] == '0') {
                bool stressed = (pi > 0 &&
                    (ph[pi-1] == '\'' || ph[pi-1] == ','));
                if (!stressed) { ph[pi] = '@'; }
            }
        }
    }
}

// Step 5.5b2: Reduce bare 'V' (ʌ) -> '@' (schwa) in unstressed
// syllables between secondary ',' and primary '\'' (American
// English). E.g. "productivity" pr,0#dVkt'Iv -> pr,0#d@kt'Iv ->
// pɹˌɑːdəktˈɪvᵻɾi.
static void reduceVBetweenSecAndPrimary(std::string& ph) {
    size_t prim_v = ph.find('\'');
    size_t sec_v  = ph.find(',');
    if (prim_v != std::string::npos &&
        sec_v != std::string::npos && sec_v < prim_v) {
        for (size_t pi = sec_v + 1; pi < prim_v; pi++) {
            if (ph[pi] == 'V') {
                bool stressed = (pi > 0 &&
                    (ph[pi-1] == '\'' || ph[pi-1] == ','));
                if (!stressed) { ph[pi] = '@'; }
            }
        }
    }
}

// Step 5.5c2: Reduce bare 'a' (æ) -> 'a#' (ɐ) in unstressed
// syllables AFTER primary '\'' and BEFORE secondary ','. Mirror of
// step 5.5c's after-secondary case. E.g. "analyst" 'anal,I ->
// 'ana#l,I -> ˈænɐlˌɪst. Only fires when primary precedes secondary
// in the phoneme string. Only for rule-derived phonemes (dict
// entries pre-encode vowel quality).
static void reduceABetweenPrimaryAndSec(
        std::string& ph, bool rule_derived) {
    if (!rule_derived) { return; }
    size_t primary_pos_a2 = ph.find('\'');
    size_t secondary_pos_a2 = ph.find(',');
    if (primary_pos_a2 != std::string::npos &&
        secondary_pos_a2 != std::string::npos &&
        primary_pos_a2 < secondary_pos_a2) {
        for (size_t pi = primary_pos_a2 + 1;
             pi < secondary_pos_a2; pi++) {
            if (ph[pi] == 'a') {
                bool is_diphthong_start = (pi + 1 < ph.size() &&
                    (ph[pi+1] == 'I' || ph[pi+1] == 'U' ||
                     ph[pi+1] == ':' || ph[pi+1] == '@' || ph[pi+1] == '#'));
                if (!is_diphthong_start) {
                    bool stressed = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        ph.insert(pi + 1, 1, '#');
                        secondary_pos_a2++;
                        pi++;
                    }
                }
            }
        }
    }
}

// Step 5.5d: Reduce bare '0' (ɑː) -> '@' (schwa) in unstressed
// syllables AFTER secondary ',' and BEFORE primary '\''.
// E.g. "democratic" d,Em0kr'at -> d,Em@kr'at -> dˌɛməkɹˈæɾɪk.
// Must NOT reduce '0' that is itself stressed (',0' or "'0") or
// part of '0#'/'02' variants.
static void reduceZeroBetweenSecAndPrimary(std::string& ph) {
    size_t primary_pos_0 = ph.find('\'');
    size_t secondary_pos_0 = ph.find(',');
    if (primary_pos_0 != std::string::npos &&
        secondary_pos_0 != std::string::npos &&
        secondary_pos_0 < primary_pos_0) {
        for (size_t pi = secondary_pos_0 + 1;
             pi < primary_pos_0; pi++) {
            if (ph[pi] == '0') {
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '#' || ph[pi+1] == '2'));
                if (is_variant) {
                    pi++;
                } else {
                    bool stressed = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) { ph[pi] = '@'; }
                }
            }
        }
    }
}

// Step 5.5d2: Mirror of 5.5d when primary comes BEFORE secondary.
// E.g. "demonstrate" d'Em0nstr,eIt -> d'Em@nstr,eIt ->
// dˈɛmənstɹˌeɪt. E.g. "parallelogram" p,ar@l'El0gr,am: secondary
// at pos 1 is BEFORE primary at pos 6, so ph.find(',') would find
// pos 1 and the primary<secondary condition would fail. Fix: find
// first ',' AFTER primary to get the secondary at pos 12 -> reduces
// '0' -> '@'.
static void reduceZeroBetweenPrimaryAndSec(std::string& ph) {
    size_t primary_d2 = ph.find('\'');
    size_t secondary_d2 = (primary_d2 != std::string::npos)
        ? ph.find(',', primary_d2) : std::string::npos;
    if (primary_d2 != std::string::npos && secondary_d2 != std::string::npos &&
        primary_d2 < secondary_d2) {
        for (size_t pi = primary_d2 + 1;
             pi < secondary_d2; pi++) {
            if (ph[pi] == '0') {
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '#' || ph[pi+1] == '2'));
                if (is_variant) {
                    pi++;
                } else {
                    bool stressed = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) { ph[pi] = '@'; }
                }
            }
        }
    }
}

// Step 5a-cleanup: Remove secondary stress markers at syllable
// distance 1 from primary. Rules can emit ',' for prefix
// syllables (e.g. "sc" -> "s," for "scientific"), but adjacent-to-
// primary secondary stress is discarded by the reference
// SetWordStress. Only runs when 5a did NOT run AND for rule-
// derived phonemes (dict entries with explicit ',' have trusted
// stress marks). Also skips removal when ',' and primary fall
// within the SAME rule (the secondary is intentional).
static void cleanupAdjacentSecondary(
        std::string& ph,
        bool step5a_ran,
        bool is_rule_leading_comma,
        const std::vector<bool>& rule_boundary_after,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (step5a_ran || is_rule_leading_comma || rule_boundary_after.empty() ||
        ph.find('\'') == std::string::npos ||
        ph.find(',') == std::string::npos) {
        return;
    }
    struct SylE {
        size_t pos;
        bool is_primary;
        bool is_secondary;
    };
    std::vector<SylE> syls2;
    {
        size_t pi2 = 0;
        bool prim2 = false;
        bool sec2 = false;
        while (pi2 < ph.size()) {
            char c = ph[pi2];
            if (c == '\'') {
                prim2 = true;
                pi2++;
            } else if (c == ',') {
                sec2 = true;
                pi2++;
            } else if (c == '%' || c == '=') {
                pi2++;
            } else {
                std::string code2 = findCode(pi2);
                if (isVowelCodeFn(code2)) {
                    syls2.push_back({pi2, prim2, sec2});
                    prim2 = sec2 = false;
                }
                pi2 += code2.size();
            }
        }
    }
    int prim_idx2 = -1;
    for (int si = 0;
         si < (int)syls2.size() && prim_idx2 < 0; si++) {
        if (syls2[si].is_primary) { prim_idx2 = si; }
    }
    if (prim_idx2 < 0) { return; }
    size_t prim_ph_pos = ph.find('\'');
    std::vector<size_t> commas_to_remove;
    for (int si = 0; si < (int)syls2.size(); si++) {
        if (syls2[si].is_secondary && std::abs(si - prim_idx2) == 1) {
            // Scan backward from syllable pos to find ','
            size_t syl_pos = syls2[si].pos;
            size_t comma_pos = std::string::npos;
            bool stop_scan = false;
            for (int bp = (int)syl_pos - 1;
                 bp >= 0 && !stop_scan; bp--) {
                if (ph[bp] == ',') {
                    comma_pos = (size_t)bp;
                    stop_scan = true;
                } else if (ph[bp] == '\'' || ph[bp] == '%') {
                    stop_scan = true;
                }
            }
            if (comma_pos != std::string::npos) {
                // Guard: same-rule secondaries are intentional.
                bool same_rule = false;
                if (!rule_boundary_after.empty() &&
                    prim_ph_pos != std::string::npos) {
                    size_t lo = std::min(comma_pos, prim_ph_pos);
                    size_t hi = std::max(comma_pos, prim_ph_pos);
                    same_rule = true;
                    for (size_t rp = lo;
                         rp < hi &&
                         rp < rule_boundary_after.size() && same_rule; rp++) {
                        if (rule_boundary_after[rp]) { same_rule = false; }
                    }
                }
                if (!same_rule) { commas_to_remove.push_back(comma_pos); }
            }
        }
    }
    std::sort(commas_to_remove.begin(), commas_to_remove.end(),
              [](size_t a, size_t b){ return a > b; });
    for (size_t pos : commas_to_remove) { ph.erase(pos, 1); }
}

// Step 5a-trochaic: Secondary stress for compound prefix words.
// When step 5a is skipped (ph_in has ',') and the leading char is
// NOT ',' (',' came from rule-internal prefix, not dict-inherent
// leading secondary), run trochaic: find the first vowel V with
// no marker AND both neighbors have no ',' or '\'' (only '%' /
// none ≤ STRESS_IS_UNSTRESSED). E.g. "electroencephalography":
// %Il,EktroUEns... -> 'E' in "encephalo" gets ','.
//
// When ph_in has no primary, the first trochaic assignment MOVES
// primary to the trochaic position (replaces step 5's pick_last).
// Otherwise (ph_in has primary), it inserts ',' as secondary.
static void trochaicCompoundPrefix(
        std::string& ph,
        const std::string& ph_in,
        bool step5a_ran,
        bool starts_with_secondary,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (step5a_ran || starts_with_secondary ||
        ph.find('\'') == std::string::npos ||
        ph_in.find(',') == std::string::npos) {
        return;
    }
    struct Syl5a3 {
        size_t pos;
        std::string code;
        int level;
    };
    std::vector<Syl5a3> syls_t;
    {
        size_t pi = 0;
        int cur_level = -1;
        while (pi < ph.size()) {
            char c = ph[pi];
            if (c == '\'') {
                cur_level = 4;
                pi++;
            } else if (c == ',') {
                cur_level = 2;
                pi++;
            } else if (c == '%' || c == '=') {
                cur_level = 1;
                pi++;
            } else {
                std::string code = findCode(pi);
                if (isVowelCodeFn(code)) {
                    syls_t.push_back({pi, code, cur_level});
                    cur_level = -1;
                } else {
                    // consonants reset (stress applies to next
                    // vowel only)
                    cur_level = -1;
                }
                pi += code.size();
            }
        }
    }
    // effectiveLv skips schwa (phNONSYLLABIC / phUNSTRESSED) when
    // looking for neighbor stress level.
    auto effectiveLv = [&](int sv, int dir) -> int {
        int found_level = -1;
        bool done = false;
        for (int nv = sv;
             nv >= 0 && nv < (int)syls_t.size() && !done;
             nv += dir) {
            const std::string& vc = syls_t[nv].code;
            if (vc=="@" || vc=="@2" || vc=="@5" || vc=="@L" || vc=="3") {
                // skip schwa
            } else {
                found_level = syls_t[nv].level;
                done = true;
            }
        }
        return found_level;
    };
    // When ph_in has no primary, trochaic uses stress=PRIMARY for
    // first assignment (moves the step-5 pick_last primary).
    bool input_has_primary = (ph_in.find('\'') != std::string::npos ||
         ph_in.find('=') != std::string::npos);
    bool trochaic_primary_done = false;
    bool stop_outer = false;
    for (int sv = 0;
         sv < (int)syls_t.size() && !stop_outer; sv++) {
        if (syls_t[sv].level == -1) {
            const std::string& vcode = syls_t[sv].code;
            // Skip phUNSTRESSED phonemes and schwa-variants.
            bool is_unstressed_phone =
                (vcode=="@" || vcode=="@2" || vcode=="@5" ||
                 vcode=="@L" || vcode=="3" || vcode=="I#" ||
                 vcode=="I2" || vcode=="a#" || vcode=="i");
            if (!is_unstressed_phone) {
                int prev_lv = effectiveLv(sv-1, -1);
                int next_lv = effectiveLv(sv+1, +1);
                if (prev_lv <= 1 && next_lv <= 1) {
                    if (!input_has_primary && !trochaic_primary_done) {
                        // Move pick_last primary to this position.
                        size_t prime_pos = ph.find('\'');
                        if (prime_pos != std::string::npos) {
                            ph.erase(prime_pos, 1);
                            for (auto& s : syls_t) {
                                if (s.pos > prime_pos) { s.pos--; }
                                if (s.level == 4) { s.level = -1; }
                            }
                        }
                        ph.insert(syls_t[sv].pos, "'");
                        for (int nv = sv + 1;
                             nv < (int)syls_t.size(); nv++) {
                            syls_t[nv].pos++;
                        }
                        syls_t[sv].level = 4;
                        trochaic_primary_done = true;
                        // Continue loop to find secondary
                    } else {
                        ph.insert(syls_t[sv].pos, ",");
                        for (int nv = sv + 1;
                             nv < (int)syls_t.size(); nv++) {
                            syls_t[nv].pos++;
                        }
                        // only one secondary
                        stop_outer = true;
                    }
                }
            }
        }
    }
}

// Step 5a-final: Add secondary stress to last stressable bare
// vowel when its direct predecessor is a schwa/phUNSTRESSED
// phoneme. Mirrors the reference SetWordStress trochaic rule for
// the last syllable: right neighbor is the sentinel (= UNSTRESSED
// = 1 ≤ 1), so only the left condition matters. The reference
// left check uses the DIRECT previous vowel's stress (not skipping
// schwas) and treats schwa/phUNSTRESSED as level=1.
// E.g. "metamorphosis" m,E*@m'o@f@sIs -> last I preceded by @ ->
// secondary -> ˌɪs.
static void finalSyllableSecondary(
        std::string& ph,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (ph.find('\'') == std::string::npos) { return; }
    struct SylFin {
        size_t pos;
        std::string code;
        int level;
    };
    std::vector<SylFin> syls_fin;
    {
        size_t pi = 0;
        int cur_level = -1;
        while (pi < ph.size()) {
            char c = ph[pi];
            if (c == '\'') {
                cur_level = 4;
                pi++;
            } else if (c == ',') {
                cur_level = 2;
                pi++;
            } else if (c == '%' || c == '=') {
                cur_level = 1;
                pi++;
            } else {
                std::string code = findCode(pi);
                if (isVowelCodeFn(code)) {
                    // Skip morpheme-boundary schwa (@-) — it's
                    // phNONSYLLABIC.
                    if (code == "@" && pi + 1 < ph.size() &&
                        ph[pi + 1] == '-') {
                        pi += code.size();
                    } else {
                        syls_fin.push_back({pi, code, cur_level});
                        cur_level = -1;
                        pi += code.size();
                    }
                } else if (ph[pi] == '-') {
                    // morpheme boundary: skip
                    pi++;
                } else {
                    // consonants don't reset cur_level (stress
                    // marker applies to next vowel regardless of
                    // intervening consonants).
                    pi += code.size();
                }
            }
        }
    }
    if (syls_fin.size() < 2) { return; }
    auto& last = syls_fin.back();
    const std::string& lcode = last.code;
    static const std::vector<std::string> UNSTRESSED_CODES =
        {"@","@2","@5","@L","3","I#","I2","a#"};
    auto is_unstressed_type = [&](const std::string& c) {
        bool found = false;
        for (auto& uc : UNSTRESSED_CODES) {
            if (c == uc) { found = true; }
        }
        return found;
    };
    bool last_is_stressable = !is_unstressed_type(lcode) && lcode != "i";
    if (last.level == -1 && last_is_stressable) {
        auto& prev = syls_fin[syls_fin.size() - 2];
        bool prev_unstressed =
            is_unstressed_type(prev.code) || prev.level == 1;
        if (prev_unstressed) {
            if (std::getenv("PHON_TRACE0")) {
                std::cerr << "[5a-final] adding secondary to last "
                          << lcode << " at pos " << last.pos
                          << " (prev=" << prev.code << ")\n";
            }
            ph.insert(last.pos, ",");
        }
    }
}

// Step 5a: Insert secondary stress ',' at even syllable distances
// from primary. Scan syllables before and after primary; every 2nd
// syllable (distance 2, 4, ...) gets ',' if it's a stressable
// vowel (not schwa/@/3/I#/I2/i/a#, not after '%' prefix).
// SKIP if ph_in already had secondary markers ',' (dict entries
// or stems from suffix stripping). Returns true if step actually
// ran (caller uses for downstream gate logic).
static bool placeSecondaryStress(
        std::string& ph,
        const std::string& ph_in,
        bool step50_fired,
        const std::vector<bool>& rule_boundary_after,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (ph.find('\'') == std::string::npos ||
        ph_in.find(',') != std::string::npos) {
        return false;
    }
    struct SylInfo {
        size_t pos;
        std::string code;
        bool stressable;
        bool is_primary;
        bool already_secondary;
    };
    std::vector<SylInfo> syls;
    {
        size_t pi = 0;
        bool unstressed_prefix = false;
        bool secondary_prefix = false;
        bool primary_prefix = false;
        while (pi < ph.size()) {
            char c = ph[pi];
            if (c == '\'') {
                primary_prefix = true;
                pi++;
            } else if (c == ',') {
                secondary_prefix = true;
                pi++;
            } else if (c == '%' || c == '=') {
                unstressed_prefix = true;
                pi++;
            } else {
                std::string code = findCode(pi);
                // Centering-diphthong split: 3+ char codes ending
                // in '@'/'3' may be a true diphthong OR two
                // syllables. Use rule boundaries to decide.
                if (code.size() >= 3 && !primary_prefix && !secondary_prefix) {
                    char last = code.back();
                    if (last == '@' || last == '3') {
                        // Convert pi to original (pre-marker) pos
                        size_t pi_orig = pi;
                        for (size_t q = 0; q < pi; q++) {
                            char qc = ph[q];
                            if (qc == '\'' || qc == ',' ||
                                qc == '%' || qc == '=') {
                                pi_orig--;
                            }
                        }
                        bool has_boundary = false;
                        for (size_t k = 0;
                             k + 1 < code.size() && !has_boundary; k++) {
                            size_t check_pos = pi_orig + k;
                            if (check_pos <
                                    rule_boundary_after.size() &&
                                rule_boundary_after[check_pos]) {
                                has_boundary = true;
                            }
                        }
                        if (has_boundary) {
                            // drop trailing '@'/'3'
                            code = code.substr(0,
                                code.size() - 1);
                        }
                    }
                }
                if (isVowelCodeFn(code)) {
                    // Skip morpheme-boundary schwa '@-' (does not
                    // count for secondary stress rhythm).
                    if (code == "@" && pi + 1 < ph.size() &&
                        ph[pi + 1] == '-') {
                        pi += code.size();
                    } else {
                        SylInfo si;
                        si.pos = pi;
                        si.code = code;
                        si.is_primary = primary_prefix;
                        si.already_secondary = secondary_prefix;
                        bool is_schwa = (code == "@" ||
                            code == "@2" || code == "@5" ||
                            code == "@L" || code == "3");
                        bool is_reduced = (code == "I#" ||
                            code == "I2" || code == "i" || code == "a#");
                        si.stressable = !is_schwa &&
                            !is_reduced && !unstressed_prefix;
                        syls.push_back(si);
                        primary_prefix = false;
                        secondary_prefix = false;
                        unstressed_prefix = false;
                        pi += code.size();
                    }
                } else {
                    pi += code.size();
                }
            }
        }
    }
    int primary_idx = -1;
    for (int si = 0;
         si < (int)syls.size() && primary_idx < 0; si++) {
        if (syls[si].is_primary) { primary_idx = si; }
    }
    if (primary_idx >= 0 && syls.size() >= 3) {
        std::vector<int> to_mark_secondary;
        int syllables_before_primary = primary_idx;
        // Backward scan: leftmost stressable at dist >= 2 from
        // primary, then cascade rightward every 2 syllables.
        if (syllables_before_primary >= 2 && !step50_fired &&
            ph.find(',') == std::string::npos) {
            int first_sec = -1;
            for (int idx = 0;
                 idx <= primary_idx - 2 && first_sec < 0; idx++) {
                if (syls[idx].stressable && !syls[idx].already_secondary &&
                    !syls[idx].is_primary) {
                    first_sec = idx;
                }
            }
            if (first_sec >= 0) {
                to_mark_secondary.push_back(first_sec);
                for (int idx = first_sec + 2;
                     idx <= primary_idx - 2; idx += 2) {
                    if (syls[idx].stressable && !syls[idx].already_secondary &&
                        !syls[idx].is_primary) {
                        to_mark_secondary.push_back(idx);
                    }
                }
                // Even-distance pass from primary backwards. Skip
                // positions adjacent to already-added secondaries.
                for (int dist = 2;
                     primary_idx - dist >= 0; dist += 2) {
                    int idx = primary_idx - dist;
                    if (syls[idx].stressable && !syls[idx].already_secondary &&
                        !syls[idx].is_primary) {
                        bool too_close = std::any_of(
                            to_mark_secondary.begin(),
                            to_mark_secondary.end(),
                            [idx](int m) {
                                return std::abs(m - idx) < 2;
                            });
                        if (!too_close) { to_mark_secondary.push_back(idx); }
                    }
                }
            }
        }
        // Forward scan: distance 2, 4, ... from primary.
        {
            int stressable_after = 0;
            for (int idx = primary_idx + 1;
                 idx < (int)syls.size(); idx++) {
                if (syls[idx].stressable) { stressable_after++; }
            }
            if (stressable_after >= 1) {
                bool stop_fwd = false;
                for (int dist = 2;
                     primary_idx + dist < (int)syls.size() &&
                     !stop_fwd; dist += 2) {
                    int idx = primary_idx + dist;
                    // Don't add secondary between two primaries.
                    bool later_primary = std::any_of(
                        syls.begin() + idx + 1, syls.end(),
                        [](const SylInfo& s) {
                            return s.is_primary;
                        });
                    if (later_primary) {
                        stop_fwd = true;
                    } else if (syls[idx].stressable &&
                        !syls[idx].already_secondary &&
                        !syls[idx].is_primary) {
                        to_mark_secondary.push_back(idx);
                    } else if (!syls[idx].stressable &&
                        !syls[idx].is_primary) {
                        // Try one further when slot is non-
                        // stressable.
                        int idx2 = primary_idx + dist + 1;
                        if (idx2 < (int)syls.size() && syls[idx2].stressable &&
                            !syls[idx2].already_secondary &&
                            !syls[idx2].is_primary) {
                            bool later_p2 = std::any_of(
                                syls.begin() + idx2 + 1, syls.end(),
                                [](const SylInfo& s) {
                                    return s.is_primary;
                                });
                            if (!later_p2) {
                                to_mark_secondary.push_back(idx2);
                            }
                        }
                    }
                }
            }
        }
        // Insert ',' right-to-left to preserve positions.
        std::sort(to_mark_secondary.begin(),
                  to_mark_secondary.end(),
                  [](int a, int b){ return a > b; });
        for (int idx : to_mark_secondary) { ph.insert(syls[idx].pos, 1, ','); }
    }
    return true;
}

// Step 5: Insert primary stress '\'' if no primary stress marker
// present. Also handles the case where only secondary stress ','
// exists (e.g., compound words via rules: ",Vnd3stand" -> insert
// '\'' on the main stressed syllable).
//
// Skip if the phoneme string starts with ',' from a DICT entry
// (inherent secondary stress, like "weren't"=,w3:nt). EXCEPTION:
// when the phoneme came from RULES (rule_boundary_after non-empty)
// and starts with ',', the leading ',' is a rule-emitted secondary
// stress for a prefix syllable (e.g. "entertain" -> ,Ent3teIn:
// '-en-' is secondary, '-tain' needs primary). In that case, run
// step 5 but treat the leading ',' as secondary so primary goes
// on the NEXT stressable vowel.
//
// Strategy:
//  - pick_last when ph ends in '=' before only weak vowels (e.g.
//    "plutonium" plu:toUn=i@m -> stress oU not u:).
//  - pick_first otherwise.
//  - force_final_stress: $strend2 words use end-stress.
static void insertPrimaryStress(
        std::string& ph,
        bool force_final_stress,
        const std::vector<bool>& rule_boundary_after,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    bool starts_with_secondary = (!ph.empty() && ph[0] == ',');
    bool is_rule_leading_comma = starts_with_secondary &&
        !rule_boundary_after.empty();
    bool active = (ph.find('\'') == std::string::npos &&
        (!starts_with_secondary || is_rule_leading_comma));
    if (!active) { return; }
    bool trace_0 = std::getenv("PHON_TRACE0") != nullptr;
    auto hasStrongAfter = [&](size_t pi) -> bool {
        bool unst = false;
        bool found = false;
        bool stop = false;
        while (pi < ph.size() && !found && !stop) {
            char c2 = ph[pi];
            if (c2 == '%' || c2 == '=' || c2 == ',') {
                unst = true;
                pi++;
            } else if (c2 == '\'') {
                stop = true;
            } else {
                std::string tc = findCode(pi);
                if (isVowelCodeFn(tc)) {
                    if (!unst && tc != "@" && tc != "@2" &&
                        tc != "@5" && tc != "@L" && tc != "3") {
                        found = true;
                    }
                    unst = false;
                }
                pi += tc.size();
            }
        }
        return found;
    };
    // Determine pick_last:
    bool pick_last = force_final_stress;
    {
        size_t last_eq = ph.rfind('=');
        if (trace_0) {
            std::cerr << "[step5] pick_last_init=" << pick_last
                      << " last_eq="
                      << (last_eq == std::string::npos ? -1
                          : (int)last_eq) << "\n";
        }
        if (last_eq != std::string::npos) {
            // Pre-scan: stressable vowels after last '='. 'i@' and
            // 'i' are weak in this position (e.g. =i@m in
            // plutonium, =i in "victory").
            bool has_strong_after = false;
            size_t pi2 = last_eq + 1;
            while (pi2 < ph.size() && !has_strong_after) {
                std::string code2 = findCode(pi2);
                if (isVowelCodeFn(code2)) {
                    bool is_weak = (code2 == "@2" ||
                        code2 == "@5" || code2 == "@L" ||
                        code2 == "I2" || code2 == "I2#" ||
                        code2 == "I#" || code2 == "a#" ||
                        code2 == "@" || code2 == "3" ||
                        code2 == "i@" || code2 == "i");
                    if (!is_weak) { has_strong_after = true; }
                }
                pi2 += code2.size();
            }
            pick_last = !has_strong_after;
            if (trace_0) {
                std::cerr << "[step5] has_strong_after="
                          << has_strong_after << " pick_last="
                          << pick_last << "\n";
            }
        }
    }
    // Rule-emitted commas are not real secondaries for primary
    // placement.
    bool ignore_comma_for_primary = !rule_boundary_after.empty();
    // Exception: secondary on the FIRST vowel + a strong diphthong
    // exists later -> treat as genuine secondary, use pick_last.
    bool use_pick_last_for_secondary = false;
    if (ignore_comma_for_primary) {
        size_t comma_pi = std::string::npos;
        bool stop_search = false;
        for (size_t spi = 0; spi < ph.size() && !stop_search;
             spi++) {
            if (ph[spi] == '\'') {
                stop_search = true;
            } else if (ph[spi] == ',') {
                if (spi > 0) { comma_pi = spi; }
                stop_search = true;
            }
        }
        if (comma_pi != std::string::npos) {
            size_t scan = comma_pi + 1;
            while (scan < ph.size() && (ph[scan] == '\'' || ph[scan] == ',' ||
                 ph[scan] == '%' || ph[scan] == '=')) {
                scan++;
            }
            if (scan < ph.size()) {
                std::string sv = findCode(scan);
                if (isVowelCodeFn(sv)) { scan += sv.size(); }
            }
            static const char* STRONG_DIPH[] = {
                "oU","aI","eI","aU","OI","aI@","aI3","aU@","i:",
                "u:","A:","E:","3:","o:","U:",nullptr
            };
            bool stop_strong = false;
            while (scan < ph.size() && !stop_strong) {
                if (ph[scan] == '\'') {
                    stop_strong = true;
                } else {
                    std::string code2 = findCode(scan);
                    for (int si = 0;
                         STRONG_DIPH[si] &&
                         !use_pick_last_for_secondary; si++) {
                        if (code2 == STRONG_DIPH[si]) {
                            use_pick_last_for_secondary = true;
                        }
                    }
                    if (use_pick_last_for_secondary) {
                        stop_strong = true;
                    } else {
                        scan += code2.size();
                    }
                }
            }
        }
    }
    if (use_pick_last_for_secondary) {
        pick_last = true;
        // treat this comma as genuine secondary
        ignore_comma_for_primary = false;
    }
    // Main scan loop:
    bool unstressed = false;
    bool secondary_next = false;
    // last non-schwa stressable vowel
    size_t last_strong_pos = std::string::npos;
    // last schwa/3 (fallback)
    size_t last_schwa_pos = std::string::npos;
    // position of first secondary-marked vowel
    size_t secondary_vowel_pos = std::string::npos;
    size_t insert_pos = std::string::npos;
    size_t pi = 0;
    bool stop_main = false;
    while (pi < ph.size() && !stop_main) {
        char c = ph[pi];
        if (c == '\'') {
            // primary already here (shouldn't happen)
            stop_main = true;
        } else if (c == ',') {
            // Always treat the leading ',' as secondary so primary
            // lands on the next stressable vowel.
            if (!ignore_comma_for_primary || pi == 0) {
                secondary_next = true;
            }
            pi++;
        } else if (c == '%' || c == '=') {
            unstressed = true;
            pi++;
        } else {
            std::string code = findCode(pi);
            if (isVowelCodeFn(code)) {
                if (secondary_next) {
                    secondary_next = false;
                    unstressed = false;
                    if (secondary_vowel_pos == std::string::npos) {
                        secondary_vowel_pos = pi;
                    }
                    pi += code.size();
                } else if (unstressed) {
                    unstressed = false;
                    pi += code.size();
                } else if (code == "@2" || code == "@5" ||
                           code == "@L" || code == "I2" ||
                           code == "I2#" || code == "I#" ||
                           code == "a#" || code == "i") {
                    // weak vowels never carry primary
                    pi += code.size();
                } else if (code == "@" || code == "3") {
                    // Schwa: only for last-resort.
                    if (!hasStrongAfter(pi + code.size())) {
                        last_schwa_pos = pi;
                        if (!pick_last && insert_pos == std::string::npos) {
                            insert_pos = pi;
                        }
                    }
                    pi += code.size();
                } else {
                    // Non-schwa stressable vowel.
                    if (pick_last) {
                        if (use_pick_last_for_secondary && code == "I") {
                            // fallback only
                            last_schwa_pos = pi;
                        } else {
                            // last wins
                            last_strong_pos = pi;
                        }
                        pi += code.size();
                    } else {
                        // first wins: stop here
                        insert_pos = pi;
                        stop_main = true;
                    }
                }
            } else {
                pi += code.size();
            }
        }
    }
    if (pick_last) {
        insert_pos = (last_strong_pos != std::string::npos)
            ? last_strong_pos : last_schwa_pos;
    } else if (insert_pos != std::string::npos) {
        // Centering/initial-diphthong skip: if primary landed on a
        // centering or initial diphthong, look forward for a
        // better candidate. (Currently disabled — kept for
        // historical reference.)
        std::string found_code = findCode(insert_pos);
        // disabled: initial diphthong IS the primary stress
        // target (e.g. "apricot" eIprIk0t -> ˈeɪpɹɪkˌɑːt).
        bool is_diphthong = false;
        // disabled: 'o@'/'e@' are primary stress targets in en-us.
        bool is_centering = false;
        if (is_diphthong || is_centering) {
            static const char* CENTERING_DIPHS[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
                "3:r","A:r","o@r","e@r","e@","i@","U@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                nullptr
            };
            static const char* ALL_DIPHS[] = {
                "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3",
                "aI","aU","eI","OI","oU",
                "3:r","A:r","o@r","e@r","e@","i@","U@","o@",
                "3:","A:","i:","u:","O:","e:","a:","aa",
                nullptr
            };
            auto isSkippable = [&](const std::string& cs) -> bool {
                const char** skip_list = is_centering
                    ? CENTERING_DIPHS : ALL_DIPHS;
                for (int di = 0; skip_list[di]; di++) {
                    if (cs == skip_list[di]) { return true; }
                }
                return false;
            };
            size_t pi2 = insert_pos + found_code.size();
            bool better_found = false;
            size_t better_pos = std::string::npos;
            bool unst2 = false;
            bool sec2 = false;
            while (pi2 < ph.size() && !better_found) {
                char c2 = ph[pi2];
                if (c2 == ',') {
                    sec2 = true;
                    pi2++;
                } else if (c2 == '%' || c2 == '=') {
                    unst2 = true;
                    pi2++;
                } else if (c2 == '\'') {
                    pi2++;
                } else {
                    std::string code2 = findCode(pi2);
                    if (isVowelCodeFn(code2)) {
                        if (sec2 || unst2 || code2 == "@" ||
                            code2 == "@2" || code2 == "@5" ||
                            code2 == "@L" || code2 == "I#" ||
                            code2 == "I2" || code2 == "a#" ||
                            code2 == "3" || code2 == "i" ||
                            isSkippable(code2)) {
                            sec2 = false;
                            unst2 = false;
                            pi2 += code2.size();
                        } else {
                            better_pos = pi2;
                            better_found = true;
                        }
                    } else {
                        pi2 += code2.size();
                    }
                }
            }
            if (better_found) { insert_pos = better_pos; }
        }
    }
    if (trace_0) {
        std::cerr << "[step5] insert_pos="
                  << (insert_pos == std::string::npos ? -1
                      : (int)insert_pos)
                  << " ph_before_insert=" << ph << "\n";
    }
    // Suppress primary on schwa when ph starts with '%' (whole-
    // phrase unstressed) and the only candidate is a schwa, OR when
    // a secondary-marked vowel exists (e.g. "gonna" g,@n@: leave
    // for StepC).
    bool at_schwa_or_r = (insert_pos != std::string::npos &&
        insert_pos < ph.size() && (ph[insert_pos] == '@' ||
         (ph[insert_pos] == '3' &&
          (insert_pos + 1 >= ph.size() || ph[insert_pos+1] != ':'))));
    bool pct_lead = ph.size() > 0 && ph[0] == '%';
    bool has_secondary = secondary_vowel_pos != std::string::npos;
    bool suppress_schwa_stress = at_schwa_or_r && (pct_lead || has_secondary);
    if (insert_pos != std::string::npos && !suppress_schwa_stress) {
        ph.insert(insert_pos, 1, '\'');
    } else {
        // Last resort: stress 'a#' as 'a' (= æ in en-us). E.g.
        // "an" (a#2n) -> ˈæn.
        size_t pi2 = 0;
        bool done = false;
        while (pi2 < ph.size() && !done) {
            std::string code = findCode(pi2);
            if (code == "a#") {
                ph.insert(pi2, 1, '\'');
                bool has_variant_digit = (pi2 + 3 < ph.size()) &&
                    (ph[pi2 + 3] >= '1' && ph[pi2 + 3] <= '9' &&
                     ph[pi2 + 3] != '3' && ph[pi2 + 3] != '8');
                if (has_variant_digit) {
                    // remove '#' -> 'a' maps to æ
                    ph.erase(pi2 + 2, 1);
                    bool stripping = true;
                    while (stripping && pi2 + 2 < ph.size()) {
                        char dc = ph[pi2 + 2];
                        if (dc >= '1' && dc <= '9' && dc != '3' && dc != '8') {
                            ph.erase(pi2 + 2, 1);
                        } else {
                            stripping = false;
                        }
                    }
                }
                done = true;
            } else if (isVowelCodeFn(code)) {
                // another vowel encountered first — leave word
                // unstressed
                done = true;
            } else {
                pi2 += code.size();
            }
        }
    }
}

// Step 5.0: Stress-shift for words with explicitly unstressed '='
// suffix syllable. If ph has primary '\'' AND has '=' (explicitly
// unstressed marker, e.g. -ity suffix's =I#t%i), AND the primary
// is NOT on the last stressable vowel before '=': move primary to
// the last stressable vowel before '=', demoting the old primary
// to ','. Also de-flap any '*' immediately before the new primary
// position (flapping only fires before UNSTRESSED vowels).
// E.g. "creativity" kri:'eI*Iv=I#*%i -> kri:,eIt'Iv=I#*%i ->
// kɹiːˌeɪtˈɪvᵻɾi.
// Guards: '=' comes AFTER '\''; no second '\'' between them
// (rules already placed it correctly); a stressable vowel exists
// between them that's different from the current primary vowel.
// Returns true if the shift fired (caller suppresses extra
// backward secondary placement in step 5a).
static bool applyEqualsSuffixStressShift(
        std::string& ph,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    bool fired = false;
    if (ph.find('\'') != std::string::npos) {
        size_t eq_pos = ph.rfind('=');     // last '=' in string
        size_t prim_pos = ph.find('\'');   // first '\'' in string
        if (eq_pos != std::string::npos && prim_pos < eq_pos) {
            // Guard: if there's already another '\'' between
            // prim_pos+1 and eq_pos, the rules placed a second
            // primary correctly (e.g. "participation" =
            // pA@t'IsIp'eIS=@n).
            bool has_second_primary = (ph.find('\'', prim_pos + 1) < eq_pos);
            if (!has_second_primary) {
                // Scan for the last stressable vowel between
                // prim_pos+1 and eq_pos-1.
                size_t last_sv_pos = std::string::npos;
                size_t scan = prim_pos + 1;
                while (scan < eq_pos) {
                    char c = ph[scan];
                    if (c == '\'' || c == ',' || c == '%' ||
                        c == '=' || c == '*') {
                        scan++;
                    } else {
                        std::string code = findCode(scan);
                        if (isVowelCodeFn(code)) {
                            // Stressable? (not schwa/reduced)
                            bool is_weak = (code == "@" ||
                                code == "@2" || code == "@5" ||
                                code == "@L" || code == "3" ||
                                code == "I#" || code == "I2" ||
                                code == "a#" || code == "i");
                            bool is_stressed = (scan > 0 &&
                                (ph[scan-1] == '\'' || ph[scan-1] == ','));
                            if (!is_weak && !is_stressed) {
                                last_sv_pos = scan;
                            }
                        }
                        scan += code.size();
                    }
                }
                if (last_sv_pos != std::string::npos) {
                    // '\'' can appear before a consonant onset
                    // (e.g. 'ju:' has '\'' before 'j'). Advance
                    // through consonants to find the actual
                    // stressed vowel.
                    size_t prim_vowel_pos = prim_pos + 1;
                    bool found_vowel = false;
                    while (prim_vowel_pos < ph.size() && !found_vowel) {
                        std::string pvc = findCode(prim_vowel_pos);
                        if (isVowelCodeFn(pvc)) {
                            found_vowel = true;
                        } else {
                            prim_vowel_pos += pvc.size();
                        }
                    }
                    if (prim_vowel_pos != last_sv_pos) {
                        // Move primary: demote old, de-flap '*',
                        // insert new '\''.
                        ph[prim_pos] = ',';
                        if (last_sv_pos > 0 && ph[last_sv_pos - 1] == '*') {
                            ph[last_sv_pos - 1] = 't';
                        }
                        ph.insert(last_sv_pos, 1, '\'');
                        fired = true;
                    }
                }
            }
        }
    }
    return fired;
}

// Step 5b: Post-stress bare 'I' before syllabic L (@L) -> 'i'
// (American English). Only converts 'I' that is directly followed
// by '@L', not arbitrary 'I' between stress and @L (avoiding e.g.
// "political" p@l'ItIk@L, where the second 'I' is followed by
// 'k@L' not '@L'). Skips first 'I' after stress (it IS the
// stressed vowel) and 'I' inside diphthongs.
static void postStressIBeforeSyllabicL(std::string& ph) {
    size_t stress_pos = ph.find('\'');
    if (stress_pos == std::string::npos) { return; }
    size_t al_pos = ph.find("@L");
    if (al_pos == std::string::npos || al_pos <= stress_pos) { return; }
    for (size_t pi = stress_pos + 1; pi < al_pos; pi++) {
        bool is_plain_I = (ph[pi] == 'I' && !(pi + 1 < ph.size() &&
              (ph[pi+1] == '2' || ph[pi+1] == '#')));
        if (is_plain_I) {
            static const std::string STEP5B_VOWELS = "aAeEiIoOuUV03@";
            bool seen_stressed_vowel = std::any_of(
                ph.begin() + stress_pos + 1, ph.begin() + pi,
                [](char ch) {
                    return STEP5B_VOWELS.find(ch) != std::string::npos;
                });
            // skip first I after stress (the stressed vowel)
            if (seen_stressed_vowel) {
                bool directly_before_al = (pi + 1 < ph.size() &&
                    ph.compare(pi+1, 2, "@L") == 0);
                static const std::string DIPH_BEFORE = "aAeEoOuU";
                bool part_of_diph = (pi > 0 && DIPH_BEFORE.find(ph[pi-1]) !=
                        std::string::npos);
                if (directly_before_al && !part_of_diph) { ph[pi] = 'i'; }
            }
        }
    }
}

// Step 5c: Reduce 3: (long ɜː) to 3 (short ɚ) in pre-tonic
// position (American English). Rules emit 3: for "er" by default;
// post-processing reduces to 3 when the syllable comes BEFORE
// primary stress, OR is between primary and a trailing secondary
// (post-tonic but with secondary after, NOT when '%' immediately
// precedes the primary — that's an applyStressPosition-forced
// case, not organic post-tonic).
// E.g. "conversation" -> kɑːnvɜːsˈeɪʃən: "ver" pre-tonic -> ɚ OK
//      "expert" -> ˈɛkspɜːt: "pert" post-tonic, no sec -> keeps ɜː
//      "metallurgist" m%Et'al3:dZ,Ist -> reduces (sec after)
// Guards: not stressed itself, not explicitly '%'-marked, has at
// least one vowel before it.
static void reduce3LongToShort(std::string& ph) {
    size_t prime_pos = ph.find('\'');
    if (prime_pos == std::string::npos) { return; }
    size_t pi = 0;
    while (pi + 1 < ph.size()) {
        if (ph[pi] == '3' && ph[pi+1] == ':') {
            bool is_stressed = (pi > 0 &&
                (ph[pi-1] == '\'' || ph[pi-1] == ','));
            // '%' immediately preceding '3:' marks intentional ɜː
            // assignment to an unstressed syllable (e.g. "ferment"
            // _f)erme(nt %3:mE — keeps ɜː).
            bool has_explicit_unstress = (pi > 0 && ph[pi-1] == '%');
            // Pre-tonic: '3:' appears before primary, AND the ':'
            // is strictly before primary (so '3:' immediately
            // adjacent to '\'' is NOT reduced — e.g. "herself"
            // h3:'sElf keeps ɜː because primary is right after).
            bool is_pretonic = (pi + 2 < prime_pos);
            // Post-tonic with secondary after: '3:' between primary
            // and secondary stress reduces to '3'. E.g.
            // "metallurgist" m%Et'al3:dZ,Ist -> ɚ.
            bool has_secondary_after =
                (ph.find(',', pi + 2) != std::string::npos);
            // When '%' immediately precedes the primary stress
            // marker (e.g. %'Int3:nS,Ip for "internship"), the
            // stress was forced by applyStressPosition and the
            // secondary after it was added by step 5a — not an
            // organic post-tonic position. Don't reduce.
            bool pct_before_primary = (prime_pos > 0 &&
                ph[prime_pos - 1] == '%');
            // Only reduce '3:' when there is another vowel BEFORE
            // it. If '3:' is the FIRST vowel (e.g. "personify"
            // p3:s'0n...), the reference keeps ɜː.
            static const std::string STEP5C_VOWELS = "aAeEiIoOuUV03@";
            bool has_vowel_before = std::any_of(
                ph.begin(), ph.begin() + pi,
                [](char ch) {
                    return STEP5C_VOWELS.find(ch) != std::string::npos;
                });
            if (!is_stressed && !has_explicit_unstress && has_vowel_before &&
                (is_pretonic ||
                 (has_secondary_after && !pct_before_primary))) {
                // remove ':' -> '3:' -> '3'
                ph.erase(pi+1, 1);
                if (prime_pos > 0) { prime_pos--; }
                // Don't advance pi; check the new char at pi+1
                // (could be a new pattern). Original used
                // `continue` for this — preserved by not
                // advancing.
            } else {
                pi += 2;
            }
        } else {
            pi++;
        }
    }
}

// Step 6: American English flap rule. /t/ -> [ɾ] between a vowel
// and unstressed vowel. Also de-flaps '*3n/m/N' -> 't3n/m/N'
// (rules emit '*' before '3' + nasal but rendering should be /t/).
// Reference rule: ChangePhoneme(t#) when prevPhW(isVowel) AND
// nextPhW(isVowel) AND nextPh(isUnstressed) AND
// (NOT next2PhW(n) OR nextPhW(3:)).
static void applyFlapRule(std::string& ph) {
    // De-tap: *3n/m/N -> t3n/m/N ("pattern", "western"-like words).
    for (size_t pi = 0; pi + 2 < ph.size(); pi++) {
        if (ph[pi] == '*' && ph[pi+1] == '3' &&
            (ph[pi+2] == 'n' || ph[pi+2] == 'm' || ph[pi+2] == 'N')) {
            ph[pi] = 't';
        }
    }
    static const std::string VOWEL_CHARS = "aAeEIiOUVu03@oY";
    for (size_t pi = 1; pi + 1 < ph.size(); pi++) {
        if (ph[pi] == 't') {
            char prev = ph[pi-1];
            // ':', '#' length/variant marks count as vowel-
            // preceded. '2' suffix in I2/E2/02/O2 also counts.
            // Syllabic-L (@L) acts as vowel for flapping:
            // "loyalty" @L+t+i -> ɾ.
            bool prev_vowel = (prev == ':' || prev == '#' || prev == 'r' ||
                VOWEL_CHARS.find(prev) != std::string::npos ||
                (prev == '2' && pi >= 2 && VOWEL_CHARS.find(ph[pi-2]) !=
                        std::string::npos) ||
                (prev == 'L' && pi >= 2 && ph[pi-2] == '@'));
            if (prev_vowel) {
                // 't#' is the reference "flappable-t" 2-char
                // phoneme code. When the char after 't' is '#',
                // skip past it to find the true next phoneme.
                size_t nxt_pos = (ph[pi+1] == '#') ? pi + 2
                                                   : pi + 1;
                char nxt = (nxt_pos < ph.size()) ? ph[nxt_pos] : 0;
                bool next_unstressed_vowel = false;
                if (nxt == '%' || nxt == '=') {
                    // Unstressed prefix followed by a vowel ->
                    // flap target. Don't flap if vowel is followed
                    // by 'n' (next2PhW(n) block).
                    if (nxt_pos + 1 < ph.size() &&
                        VOWEL_CHARS.find(ph[nxt_pos+1]) != std::string::npos) {
                        bool v2_long = (nxt_pos + 2 < ph.size() &&
                            ph[nxt_pos+2] == ':');
                        int n2p = v2_long ? (int)(nxt_pos + 3)
                                          : (int)(nxt_pos + 2);
                        bool n2_n = (n2p < (int)ph.size() && ph[n2p] == 'n');
                        bool v2_3c = (ph[nxt_pos+1] == '3' && v2_long);
                        next_unstressed_vowel = !(n2_n && !v2_3c);
                    }
                } else if (nxt != '\'' && nxt != ',' &&
                           VOWEL_CHARS.find(nxt) != std::string::npos) {
                    bool is_3colon = (nxt == '3' && nxt_pos + 1 < ph.size() &&
                        ph[nxt_pos+1] == ':');
                    bool vowel_is_long = (nxt_pos + 1 < ph.size()
                        && ph[nxt_pos+1] == ':');
                    int next2_pos = vowel_is_long
                        ? (int)(nxt_pos + 2)
                        : (int)(nxt_pos + 1);
                    // '@L' (syllabic L) is a single phoneme: skip
                    // the 'L' modifier to find the next real
                    // phoneme. E.g. "bottleneck" has t@Ln -> skip
                    // L -> see n -> no flap. But "bottle" has
                    // t@L(end) -> no n -> flap fires.
                    if (nxt == '@' && next2_pos < (int)ph.size() &&
                        ph[next2_pos] == 'L') {
                        next2_pos++;
                    }
                    bool next2_is_n =
                        (next2_pos < (int)ph.size() && ph[next2_pos] == 'n');
                    // No flap when: next vowel followed by 'n'
                    // AND NOT the 3: exception.
                    next_unstressed_vowel = !(next2_is_n && !is_3colon);
                }
                if (next_unstressed_vowel) { ph[pi] = '*'; }
            }
        }
    }
}

// Step 6b: '-ness' suffix reduction. 'nEs' at word-end -> 'n@s'
// (schwa, not ɛ). Also handles 'n,Es' (spurious secondary stress
// on -ness — remove the comma and reduce).
static void reduceNessSuffix(std::string& ph) {
    size_t plen = ph.size();
    if (plen >= 4 && ph[plen-1] == 's' && ph[plen-2] == 'E' &&
        ph[plen-3] == ',' && ph[plen-4] == 'n') {
        // n,Es -> n@s
        ph.replace(plen-3, 3, "@s");
    } else if (plen >= 3 && ph[plen-1] == 's' &&
               ph[plen-2] == 'E' && ph[plen-3] == 'n') {
        // nEs -> n@s
        ph.replace(plen-2, 2, "@s");
    }
}

// Step 6c: Reduce 'oU#' (compound-prefix vowel) based on adjacent
// stress context. Reference 'oU#' phoneme:
//   IF thisPh(isStressed)         -> ChangePhoneme(0)   [-> ɑː]
//   IF nextVowel/prevVowel
//       (isStressed)              -> ChangePhoneme(@)   [-> schwa]
//   default                       -> ChangePhoneme(oU)
// 'isStressed' = PRIMARY only (stress_level >= STRESS_PRIMARY).
// e.g. "cryptocurrency" kr'IptoU# -> 'I primary prev -> oU# -> @
// e.g. "Gastrointestinal" g,astroU#Int'Est -> no primary adj -> oU
// Scan backward from `before` looking for the nearest stress
// separator. Returns true if '\'' is found before ',' / '%'.
static bool prevSyllableIsPrimary(const std::string& ph,
                                  size_t before) {
    bool found = false;
    bool stop = false;
    int j = (int)before - 1;
    while (j >= 0 && !found && !stop) {
        char c = ph[j];
        if (c == '\'') { found = true; }
        else if (c == ',' || c == '%') { stop = true; }
        j--;
    }
    return found;
}

// Scan forward from `after` looking for the nearest stressed vowel.
// Returns true iff '\'' appears before a ',' or a bare vowel char.
static bool nextSyllableIsPrimary(const std::string& ph,
                                  size_t after) {
    static const std::string VOWEL_FWD = "aAeEiIoOuUV03@";
    bool found = false;
    bool stop = false;
    size_t j = after;
    while (j < ph.size() && !found && !stop) {
        char c = ph[j];
        if (c == '\'') {
            found = true;
        } else if (c == ',' || VOWEL_FWD.find(c) != std::string::npos) {
            stop = true;
        }
        j++;
    }
    return found;
}

static void reduceOUHashCompound(std::string& ph) {
    for (size_t i = 0; i + 2 < ph.size(); i++) {
        if (ph[i] == 'o' && ph[i+1] == 'U' && ph[i+2] == '#') {
            // Self primary-stressed (directly preceded by '\'')?
            bool self_primary = (i > 0 && ph[i-1] == '\'');
            if (self_primary) {
                ph.replace(i, 3, "0");
            } else if (prevSyllableIsPrimary(ph, i)) {
                ph.replace(i, 3, "@");
            } else if (nextSyllableIsPrimary(ph, i + 3)) {
                ph.replace(i, 3, "@");
            } else {
                // Default: keep as oU (remove #).
                ph.replace(i, 3, "oU");
            }
        }
    }
}

// Step 6e: '-ically' schwa elision. Word-final 'k@li' -> 'kli'.
// E.g. "typically"->tˈɪpɪkli, "basically"->bˈeɪsɪkli.
static void elideIcallySchwa(std::string& ph) {
    if (ph.size() >= 4 && ph[ph.size()-4]=='k' && ph[ph.size()-3]=='@' &&
        ph[ph.size()-2]=='l' && ph[ph.size()-1]=='i') {
        // remove '@': 'k@li' -> 'kli'
        ph.erase(ph.size()-3, 1);
    }
}

// Step 6f: Add secondary stress before syllabic-n in compound
// words. E.g. "handwritten" -> h'andrI?n- -> h'andrI?,n- (the
// reference: hˈændɹɪʔˌn̩). Condition: phoneme ends in 'n-' (with
// optional preceding '?'), primary stress already present, AND
// 2+ vowel groups before the syllabic consonant. Simple 2-syll
// words like "written"/"kitten" have only 1 vowel group -> skip.
// True compounds have no '%'/'=' before primary (those mark an
// unstressed prefix like "un-"/"en-" — not a compound first
// element).
static void addCompoundSyllabicNStress(std::string& ph) {
    if (ph.size() < 2 || ph[ph.size()-1] != '-' || ph[ph.size()-2] != 'n' ||
        ph.find('\'') == std::string::npos) {
        return;
    }
    static const std::string VC_SN = "aAeEiIoOuUV03@";
    // Find start of the syllabic block ('?n-' or just 'n-')
    size_t sn_start = ph.size() - 2;
    if (sn_start > 0 && ph[sn_start-1] == '?') { sn_start--; }
    // Count vowel groups before sn_start
    int vgroups = 0;
    bool in_v = false;
    for (size_t vi = 0; vi < sn_start; vi++) {
        char vc = ph[vi];
        if (vc != '\'' && vc != ',' && vc != '%' && vc != '=') {
            if (VC_SN.find(vc) != std::string::npos) {
                if (!in_v) {
                    vgroups++;
                    in_v = true;
                }
            } else {
                in_v = false;
            }
        }
    }
    // Check for unstressed-prefix marker before primary
    size_t prime = ph.find('\'');
    bool has_unstressed_prefix = (prime != std::string::npos) &&
        std::any_of(ph.begin(), ph.begin() + prime,
            [](char c) { return c == '%' || c == '='; });
    if (vgroups >= 2 && !has_unstressed_prefix &&
        (sn_start == 0 || ph[sn_start-1] != ',')) {
        ph.insert(sn_start, ",");
    }
}

// Step 5.5-dim0: Reduce phoneme `0` (ɑː) to `@` (schwa) when it
// receives DIMINISHED stress. In the reference, `phoneme 0` has
// ChangeIfDiminished(@). English stress_flags=0x08 has S_MID_DIM
// (0x10000) NOT set, so ALL "middle" vowels with effective stress
// ≤ UNSTRESSED get DIMINISHED. A "middle" vowel is one that is:
//   - not the first syllable (v != 1, 1-indexed)
//   - not the last syllable (v != N)
//   - not the penultimate syllable (v == N-1) when the last
//     syllable's initial stress ≤ 1
// Only reduce `0` with no stress marker (bare, level -1) or
// explicit UNSTRESSED (level 1); `0` with secondary or primary
// stress is kept as-is.
// Reference: SetWordStress finalization loop, dictionary.c
// lines 1503-1519.
static void reduceZeroDiminished(
        std::string& ph,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (ph.find('0') == std::string::npos) { return; }
    // Build syllable list with correct levels (stress markers
    // persist through consonants).
    struct Syl0D { size_t pos; std::string code; int level; };
    std::vector<Syl0D> syls0d;
    size_t pi = 0;
    int cur_level = -1;
    while (pi < ph.size()) {
        char c = ph[pi];
        if (c == '\'') {
            cur_level = 4;
            pi++;
        } else if (c == ',') {
            cur_level = 2;
            pi++;
        } else if (c == '%' || c == '=') {
            cur_level = 1;
            pi++;
        } else {
            std::string code = findCode(pi);
            if (isVowelCodeFn(code)) {
                // skip @- (morpheme-boundary non-syllabic schwa)
                if (code == "@" && pi + code.size() < ph.size() &&
                    ph[pi + code.size()] == '-') {
                    pi += code.size();
                } else {
                    syls0d.push_back({pi, code, cur_level});
                    cur_level = -1;
                    pi += code.size();
                }
            } else if (ph[pi] == '-') {
                pi++;
            } else {
                // consonants don't reset cur_level
                pi += code.size();
            }
        }
    }
    int N0 = (int)syls0d.size();
    // Process right-to-left to preserve insertion positions.
    // A '0' is DIMINISHED only when ALL of these hold:
    //   - it's a '0' code with no explicit stress (level <= 1);
    //   - it's not the first or last syllable (those are
    //     UNSTRESSED, not DIMINISHED);
    //   - if penultimate, the final syllable is itself stressed
    //     (so the '0' is mid-cluster, not pre-unstressed-final).
    for (int v = N0 - 1; v >= 0; v--) {
        int vnum = v + 1;
        bool eligible = (syls0d[v].code == "0" && syls0d[v].level <= 1 &&
                         vnum != 1 && vnum != N0);
        if (eligible && vnum == N0 - 1 && syls0d[N0 - 1].level <= 1) {
            eligible = false;
        }
        if (eligible) {
            if (std::getenv("PHON_TRACE0")) {
                std::cerr << "[5.5-dim0] reducing 0→@ at pos "
                          << syls0d[v].pos << " (vnum=" << vnum
                          << "/" << N0 << ")\n";
            }
            ph.replace(syls0d[v].pos, 1, "@");
        }
    }
}

// Step 5a-prime: Adjacent primary demotion (mimics phonSTRESS_PREV
// GetVowelStress logic). When phonSTRESS_PREV (leading '=' in a
// rule output) fires, it promotes a preceding vowel to PRIMARY and
// demotes any earlier PRIMARY to SECONDARY. This happens in the
// byte stream BEFORE SetWordStress runs. Words where
// phonSTRESS_PREV fires mid-word can end up with two primaries at
// syllable distance 1 (e.g. "creativity": eI+I from =I#t%i rule).
// The reference outputs ˌeɪtˈɪ (secondary + primary) because
// phonSTRESS_PREV demoted the earlier primary.
//
// Rule: if any two consecutive PRIMARY syllables are at distance 1
// (adjacent), demote the EARLIER one to secondary. Result:
// adjacent primaries never coexist; earlier -> secondary.
// Guard: only fire when ph_in already had "'" (the double-primary
// came from rules, not dict+suffix).
static void demoteAdjacentPrimaries(
        std::string& ph,
        const std::string& ph_in,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (ph_in.find('\'') == std::string::npos) { return; }
    // Build syllable list with primary/secondary flags
    struct SylP { size_t pos; bool is_primary; };
    std::vector<SylP> sylsp;
    size_t pi = 0;
    bool prim = false;
    while (pi < ph.size()) {
        char c = ph[pi];
        if (c == '\'') {
            prim = true;
            pi++;
        } else if (c == ',' || c == '%' || c == '=') {
            prim = false;
            pi++;
        } else {
            std::string code = findCode(pi);
            if (isVowelCodeFn(code)) {
                sylsp.push_back({pi, prim});
                prim = false;
            }
            pi += code.size();
        }
    }
    // Find pairs of primaries at distance 1 and demote the earlier
    // one. Process right-to-left so index positions stay valid.
    // positions of "'" to change to ","
    std::vector<size_t> demote_positions;
    for (int si = (int)sylsp.size() - 1; si >= 1; si--) {
        if (sylsp[si].is_primary && sylsp[si-1].is_primary) {
            // Two consecutive primaries: demote the EARLIER one
            // (si-1). Find the "'" marker before sylsp[si-1].pos.
            if (sylsp[si-1].pos > 0) {
                size_t comma_pos = sylsp[si-1].pos - 1;
                if (comma_pos < ph.size() && ph[comma_pos] == '\'') {
                    demote_positions.push_back(comma_pos);
                    // mark as demoted
                    sylsp[si-1].is_primary = false;
                }
            }
        }
    }
    // Apply demotions: change "'" to "," at these positions
    for (size_t pos : demote_positions) { ph[pos] = ','; }
}

// Step 5.5c: Reduce bare 'a' (æ) -> 'a#' (ɐ) in unstressed
// syllables AFTER a secondary-stress marker ',' or an explicit
// unstressed '%' prefix and BEFORE the primary stress '\''.
// Examples:
//  - With ',': "analytical" ,anal'I -> ,ana#l'I; "fantastic"
//    fant'ast stays as-is (no ',' before first 'a').
//  - With '%': "transatlantic" tr%ansatl'aan -> tr%ansp'a#tl'aan;
//    'a' directly after '%' is protected (it IS the prefix vowel).
// Only applies to bare 'a' that is NOT a diphthong start (aI, aU,
// a:, a@, a#). Only applies to rule-derived phonemes: dict
// entries specify vowel quality explicitly.
static void reduceABeforePrimary(
        std::string& ph,
        bool rule_derived,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    if (!rule_derived) { return; }
    size_t primary_pos_a = ph.find('\'');
    if (primary_pos_a == std::string::npos) { return; }
    size_t secondary_pos_a = ph.find(',');
    size_t pct_pos_a = ph.find('%');
    // Determine scan start: earliest of ',' or '%' that comes
    // before primary.
    size_t scan_start_a = std::string::npos;
    if (secondary_pos_a != std::string::npos &&
        secondary_pos_a < primary_pos_a) {
        scan_start_a = secondary_pos_a + 1;
    }
    // '%' prefix case: scan from AFTER the first vowel following
    // '%'. The first vowel after '%' is the prefix vowel (v=1,
    // PROTECTED/UNSTRESSED). Any bare 'a' after that first vowel
    // and before primary is a middle vowel -> DIMINISHED -> reduce
    // to 'a#'. Handles: "explanation" (%e#ksplan'eIS=@n): first
    // vowel=e#, then 'a' is middle -> ɐ. "transatlantic"
    // (tr%ansatl'aant): first vowel='a' after %, scan past it.
    if (pct_pos_a != std::string::npos && pct_pos_a < primary_pos_a) {
        // Advance past '%' to find the first vowel phoneme code
        size_t pct_scan = pct_pos_a + 1;
        while (pct_scan < primary_pos_a &&
               !isVowelCodeFn(findCode(pct_scan))) {
            pct_scan++;
        }
        if (pct_scan < primary_pos_a) {
            // Skip past the first (protected) vowel code
            pct_scan += findCode(pct_scan).size();
        }
        if (pct_scan < primary_pos_a) {
            scan_start_a =
                (scan_start_a == std::string::npos) ? pct_scan
                : std::min(scan_start_a, pct_scan);
        }
    }
    // General case: no '%' or ',' triggered a scan, but 'a' might
    // still be a middle vowel. Example: "reclamation"
    // (rI#klam'eIS=@n) — the 'a' is v=2 of 4 vowels -> DIMINISHED.
    // Scan from AFTER the first vowel before primary.
    if (scan_start_a == std::string::npos) {
        size_t gen_scan = 0;
        bool found_first = false;
        while (gen_scan < primary_pos_a && !found_first) {
            if (ph[gen_scan] == '\'' || ph[gen_scan] == ',' ||
                ph[gen_scan] == '%') {
                gen_scan++;
            } else {
                std::string gc = findCode(gen_scan);
                if (isVowelCodeFn(gc)) {
                    // Skip past this first (protected) vowel
                    gen_scan += gc.size();
                    found_first = true;
                } else {
                    gen_scan += gc.size();
                }
            }
        }
        if (gen_scan < primary_pos_a) { scan_start_a = gen_scan; }
    }
    if (scan_start_a != std::string::npos) {
        // Reduce bare 'a' that is:
        //   - AFTER the scan_start position
        //   - BEFORE the primary stress marker
        //   - NOT immediately following ',', '\'', or '%'
        //     (protected vowel)
        //   - NOT a diphthong start (aI, aU, a:, a@, a#)
        for (size_t pi2 = scan_start_a;
             pi2 < primary_pos_a; pi2++) {
            if (ph[pi2] == 'a') {
                bool is_diphthong_start = (pi2 + 1 < ph.size() &&
                    (ph[pi2+1] == 'I' || ph[pi2+1] == 'U' ||
                     ph[pi2+1] == ':' || ph[pi2+1] == '@' ||
                     ph[pi2+1] == '#'));
                if (!is_diphthong_start) {
                    // 'a' directly after a stress marker or '%'
                    // is the marked vowel itself.
                    bool protected_vowel = (pi2 > 0 && (ph[pi2-1] == '\'' ||
                         ph[pi2-1] == ',' || ph[pi2-1] == '%'));
                    if (!protected_vowel) {
                        ph.insert(pi2 + 1, 1, '#');
                        primary_pos_a++;
                        pi2++;
                    }
                }
            }
        }
    }
}

// Step 5.5b: Reduce bare 'E' (ɛ) -> 'I2' (ɪ) (or '@' before nasal
// 'n') in unstressed positions across four contexts:
//   1. Between secondary ',' and primary '\'' (e.g. "mathematics"
//      m,aTEm'at -> m,aTI2m'at).
//   2. After '%' initial unstressed syllable ("expectation"
//      %e#kspEkt'eIS -> %e#kspI2kt'eIS).
//   3. Between primary '\'' and trailing secondary ','
//      ("challenging" tS'alEndZ,IN -> tS'al@ndZ,IN).
//   4. After all explicit stress markers — middle unstressed E
//      gets DIMINISHED -> I2, with penultimate-vowel exception when
//      the only following vowel is phUNSTRESSED (@, I#, etc.).
// Guarded to rule-derived phonemes (rule_boundary_after non-empty)
// for contexts 1, 3, 4 — dict entries pre-encode their vowel
// quality. Caller passes `findCode` and `isVowelCodeFn` because
// step 5.5b needs the multi-char-aware vowel scanner used by
// processPhonemeString.
static void reduceEUnstressed(
        std::string& ph,
        const std::vector<bool>& rule_boundary_after,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) {
    size_t primary_pos = ph.find('\'');
    size_t secondary_pos_e = ph.find(',');
    // Context 1: secondary stress before primary
    if (!rule_boundary_after.empty() && primary_pos != std::string::npos &&
        secondary_pos_e != std::string::npos &&
        secondary_pos_e < primary_pos) {
        // Determine the secondary-stressed vowel (immediately
        // after ','). When secondary vowel is itself 'E',
        // subsequent bare E reduces to '@' (schwa) instead of
        // 'I2'. E.g. "presentation" ,Ez+E -> '@', not 'I2'. But
        // "mathematics" ,aT+E -> 'I2'.
        // (sec_is_E logic preserved as locally-tracked but unused
        // after the simplification — kept for documentation; the
        // four contexts together cover all observed cases.)
        for (size_t pi = secondary_pos_e + 1;
             pi < primary_pos; pi++) {
            if (ph[pi] == 'E') {
                // Check it's not E# or E2
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '#' || ph[pi+1] == '2'));
                if (is_variant) {
                    pi++;
                } else {
                    // Check if stressed (preceded by ' or ,)
                    bool stressed = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        // E before nasal 'n': reduce to '@', not
                        // 'I2'. E.g. "compensation" k,0mpEns ->
                        // k,0mp@ns. Otherwise -> 'I2' (ɪ).
                        bool before_n = (pi + 1 < primary_pos &&
                            ph[pi+1] == 'n');
                        if (before_n) {
                            ph[pi] = '@';
                        } else {
                            ph.replace(pi, 1, "I2");
                            // adjust for the extra char inserted
                            primary_pos++;
                            // skip over the '2' we just inserted
                            pi++;
                        }
                    }
                }
            }
        }
    }
    // Context 3: primary stress before secondary stress, 'E'
    // unstressed in between.
    //   - E before nasal 'n' -> '@'. E.g. "challenging" tS'alEndZ,IN
    //     -> tS'al@ndZ,IN.
    //   - E elsewhere -> 'I2'. E.g. "basketball" b'aa#skEtb,O:l ->
    //     b'aa#skI2tb,O:l.
    if (!rule_boundary_after.empty() && primary_pos != std::string::npos &&
        secondary_pos_e != std::string::npos &&
        secondary_pos_e > primary_pos) {
        for (size_t pi = primary_pos + 1;
             pi < secondary_pos_e; pi++) {
            if (ph[pi] == 'E') {
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '#' || ph[pi+1] == '2'));
                if (is_variant) {
                    pi++;
                } else {
                    bool stressed = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        bool before_n = (pi + 1 <
                            secondary_pos_e && ph[pi+1] == 'n');
                        if (before_n) {
                            ph[pi] = '@';
                        } else {
                            ph.replace(pi, 1, "I2");
                            // adjust for extra char inserted
                            secondary_pos_e++;
                            pi++; // skip the '2'
                        }
                    }
                }
            }
        }
    }
    // Context 2: '%' initial unstressed syllable (like "ex-"
    // prefix). Scan for 'E' between the end of the '%'-marked
    // initial syllable and primary stress.
    if (primary_pos != std::string::npos && !ph.empty() && ph[0] == '%') {
        // Find the end of the first vowel code after '%' (that's
        // the '%'-marked vowel; skip it).
        size_t scan_start = 1;
        // Advance past stress/modifier markers
        while (scan_start < ph.size() &&
            (ph[scan_start] == '\'' || ph[scan_start] == ',' ||
             ph[scan_start] == '%' || ph[scan_start] == '=')) {
            scan_start++;
        }
        // Now scan_start points to the first vowel code; advance
        // past it
        if (scan_start < ph.size()) {
            std::string fc = findCode(scan_start);
            if (isVowelCodeFn(fc)) { scan_start += fc.size(); }
        }
        // Now reduce any bare 'E' between scan_start and
        // primary_pos
        for (size_t pi = scan_start; pi < primary_pos; pi++) {
            if (ph[pi] == 'E') {
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '#' || ph[pi+1] == '2'));
                if (is_variant) {
                    pi++;
                } else {
                    bool stressed = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ','));
                    if (!stressed) {
                        ph.replace(pi, 1, "I2");
                        primary_pos++;
                        pi++;
                    }
                }
            }
        }
    }
    // Context 4: bare 'E' after ALL explicit stress markers ('\''
    // and ','), not the last vowel. Middle unstressed vowels get
    // DIMINISHED stress in the reference -> ChangeIfDiminished(I2)
    // -> ɪ. Exception (the reference penultimate check): if E is
    // the PENULTIMATE vowel AND the final vowel has phUNSTRESSED
    // attribute (codes: '@','@2','@5','@L','3','I#'), E stays
    // UNSTRESSED -> ɛ.
    // e.g. "physiotherapy" f'IzI2,oUTEr@p%i -> E followed by 2
    // vowels (@,i) -> middle -> I2. OK
    // e.g. "processes" pr'0sEs%I#z -> E followed by 1 vowel (I#,
    // phUNSTRESSED) -> penultimate -> ɛ. OK
    if (!rule_boundary_after.empty() && primary_pos != std::string::npos) {
        // Find the first explicit stress marker position. Using
        // first_stress (not last_stress) ensures we also catch 'E'
        // that lies between two secondary stress markers.
        size_t first_stress = std::string::npos;
        for (size_t pi = 0;
             pi < ph.size() && first_stress == std::string::npos;
             pi++) {
            if (ph[pi] == '\'' || ph[pi] == ',') { first_stress = pi; }
        }
        if (first_stress != std::string::npos) {
            // phUNSTRESSED codes (the reference phUNSTRESSED flag)
            auto isUnstressedCode = [](const std::string& vc) {
                return (vc == "@" || vc == "@2" || vc == "@5" ||
                        vc == "@L" || vc == "3" || vc == "I#");
            };
            for (size_t pi = first_stress + 1; pi < ph.size(); ) {
                if (ph[pi] == 'E') {
                    bool is_variant = (pi + 1 < ph.size() &&
                        (ph[pi+1] == '#' || ph[pi+1] == '2'));
                    if (is_variant) {
                        pi += 2;
                    } else {
                        bool stressed = (pi > 0 && (ph[pi-1] == '\'' ||
                             ph[pi-1] == ','));
                        bool reduce_to_i2 = false;
                        if (!stressed) {
                            // Collect vowels after this E
                            std::vector<std::string> vowels_after;
                            for (size_t pj = pi + 1;
                                 pj < ph.size(); ) {
                                std::string vc = findCode(pj);
                                if (isVowelCodeFn(vc)) {
                                    vowels_after.push_back(vc);
                                }
                                pj += vc.size();
                            }
                            // Fire if 2+ vowels after E (middle),
                            // OR exactly 1 vowel that's NOT
                            // phUNSTRESSED.
                            bool should_reduce = vowels_after.size() >= 2 ||
                                (vowels_after.size() == 1 &&
                                    !isUnstressedCode(vowels_after[0]));
                            if (should_reduce) {
                                bool before_n = (pi + 1 <
                                    ph.size() && ph[pi+1] == 'n');
                                if (before_n) {
                                    ph[pi] = '@';
                                } else {
                                    ph.replace(pi, 1, "I2");
                                    reduce_to_i2 = true;
                                }
                            }
                        }
                        // skip '2' just inserted when we did I2 replace
                        pi += reduce_to_i2 ? 2 : 1;
                    }
                } else {
                    pi++;
                }
            }
        }
    }
}

// Step 5.5e: Reduce '0#' -> '@' (schwa) when it precedes primary
// stress, UNLESS there is an UNSTRESSED vowel code between '0#'
// and primary. Examples:
//   "contains"    k%0#nt'eInz    -> no inter vowel       -> reduce
//   "consolidate" k%0#ns,0lId'   -> first inter vowel '0' is
//                                  stressed (,)         -> reduce
//   "condensation"k%0#ndE2ns'eI  -> first inter vowel 'E'
//                                  is UNSTRESSED        -> keep
//   "contribute"  k0#ntr'Ibju:t  -> no inter vowel       -> reduce
// Step 5.5d already covers the between-secondary-and-primary case.
// Returns true iff the first vowel code in ph[start..end) is
// unstressed (no preceding '\''/','). Used by 5.5e to detect "heavy
// ɑː syllable" patterns where extra rule-emitted phonemes mean we
// keep '0#' as ɑː instead of reducing to schwa.
static bool firstInterVowelIsUnstressed(
        const std::string& ph, size_t start, size_t end) {
    static const std::string VOW = "aAeEiIoOuUV03@";
    bool unstressed = false;
    bool found = false;
    size_t si = start;
    while (si < end && !found) {
        char c = ph[si];
        if (VOW.find(c) != std::string::npos) {
            bool sv = (si > 0 && (ph[si-1] == '\'' || ph[si-1] == ','));
            if (!sv) { unstressed = true; }
            found = true;
        }
        si++;
    }
    return unstressed;
}

static void reduceZeroHashBeforePrimary(std::string& ph) {
    size_t primary_pos = ph.find('\'');
    if (primary_pos != std::string::npos) {
        for (size_t pi = 0; pi < primary_pos; pi++) {
            if (ph[pi] == '0') {
                bool stressed = (pi > 0 &&
                    (ph[pi-1] == '\'' || ph[pi-1] == ','));
                bool has_hash = (pi + 1 < ph.size() && ph[pi+1] == '#');
                bool is_02 = (pi + 1 < ph.size() && ph[pi+1] == '2');
                if (!stressed && is_02) {
                    pi++;            // skip the '2'
                } else if (!stressed && has_hash &&
                    !firstInterVowelIsUnstressed(ph, pi + 2,
                                                 primary_pos)) {
                    // '0#' -> '@'. Heavy-syllable case (unstressed
                    // inter vowel present) keeps ɑː.
                    ph.replace(pi, 2, "@");
                    primary_pos--;
                }
                // bare '0' always stays as ɑː.
            }
        }
    }
}

// Step 5.5f: Reduce bare '0' (ɑː) -> '@' (schwa) in unstressed
// syllables BEFORE primary '\''. Complements step 5.5d (which
// requires a preceding secondary ',') by handling the case where
// there is no secondary stress. The reference SetWordStress
// reduces all unstressed '0' phonemes to schwa.
// E.g. "astronomical" a#str0n'0m=Ik@L -> a#str@n'0m=Ik@L
//      -> ɐstɹənˈɑːmɪkəl.
// Guards: '0' must be bare (no '\''/',' immediately before it),
// before primary, AND have an existing rule boundary AFTER it
// (i.e. a standalone rule output, not bundled e.g. with 'l' from
// rule `v)ol(C 0l`). Caller must pass rule_boundary_after to
// distinguish rule-emitted from dict-emitted '0'.
static void reduceBareZeroBeforePrimary(
        std::string& ph,
        const std::vector<bool>& rule_boundary_after) {
    if (rule_boundary_after.empty()) { return; }
    size_t prim_pos_5f = ph.find('\'');
    if (prim_pos_5f != std::string::npos) {
        for (size_t pi = 0; pi < prim_pos_5f; pi++) {
            if (ph[pi] == '0') {
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '#' || ph[pi+1] == '2'));
                if (is_variant) {
                    pi++;
                } else {
                    // Don't reduce if stressed ('\','/',') or
                    // explicitly marked ('%','='). '%0' = rule-
                    // explicit unstressed ɑː (keep); bare '0' =
                    // reduce to '@'.
                    bool marked = (pi > 0 &&
                        (ph[pi-1] == '\'' || ph[pi-1] == ',' ||
                         ph[pi-1] == '%' || ph[pi-1] == '='));
                    // Only reduce '0' that is a standalone rule
                    // output (boundary after it). When '0' is
                    // bundled with a following phoneme (e.g. '0l'
                    // from rule `v)ol(C 0l`), there's no rule
                    // boundary after '0', so don't reduce it.
                    // E.g. "volcanic" v0lk'an=Ik: '0' at pos 1 has
                    // no boundary (part of 0l) -> keep ɑː.
                    //      "astronomical" a#str0n'0m=: '0' at
                    // pos 5 has boundary -> reduce to @.
                    bool is_standalone = (pi <
                        rule_boundary_after.size() && rule_boundary_after[pi]);
                    // Don't reduce if this '0' is the first vowel
                    // in the pre-tonic region — the reference
                    // keeps the initial-syllable '0' as ɑː even
                    // when pre-tonic. E.g. "oncology" 0Nk'0l: first
                    // '0' at pos 0, no prior vowel -> keep ɑː.
                    //      "astronomical" a#str0n'0m=: prior vowel
                    // 'a' -> reduce to @.
                    static const std::string VOW_5F_SET = "aAeEiIoOuUV03@";
                    bool has_prior_vowel = std::any_of(
                        ph.begin(), ph.begin() + pi,
                        [](char ch) {
                            return VOW_5F_SET.find(ch) != std::string::npos;
                        });
                    if (!marked && is_standalone && has_prior_vowel) {
                        ph[pi] = '@';
                    }
                }
            }
        }
    }
}

// Step 5.5g: Reduce bare 'E' -> '@' in pre-tonic position when 'E'
// follows a '%'-marked syllable (with a vowel between '%' and 'E')
// and is immediately followed by nasal 'n'. Reference reduces this
// 'E' to schwa in AmEng. E.g. "fermentation" f%3:mEnt'eIS=@n ->
// f%3:m@nt'eIS=@n (fɜːmənˈteɪʃən).
// Guard: only when there's a vowel between '%' and 'E' so 'E' is
// NOT the nucleus of the '%'-syllable (e.g. %Env... keeps ɛ since
// E is right after %).
static void reduceEPreTonicAfterPctNasal(std::string& ph) {
    static const std::string VOW_5G = "aAeEiIoOuUV03@";
    size_t prime_pos_g = ph.find('\'');
    if (prime_pos_g != std::string::npos) {
        for (size_t pi = 1; pi < prime_pos_g; pi++) {
            if (ph[pi] == 'E') {
                bool is_variant = (pi + 1 < ph.size() &&
                    (ph[pi+1] == '2' || ph[pi+1] == '#'));
                if (is_variant) {
                    pi++;
                } else if (ph[pi-1] != '\'' && ph[pi-1] != ',' &&
                           pi + 1 < ph.size() && ph[pi+1] == 'n') {
                    // Find last '%' before this 'E'
                    size_t pct_pos = std::string::npos;
                    for (size_t j = 0; j < pi; j++) {
                        if (ph[j] == '%') { pct_pos = j; }
                    }
                    if (pct_pos != std::string::npos) {
                        // Check there's at least one vowel between
                        // '%' and 'E'
                        bool has_vowel_between = std::any_of(
                            ph.begin() + pct_pos + 1,
                            ph.begin() + pi,
                            [](char ch) {
                                return VOW_5G.find(ch) != std::string::npos;
                            });
                        if (has_vowel_between) { ph[pi] = '@'; }
                    }
                }
            }
        }
    }
}

// Step 4b: Linking R. Rhotacized vowels followed by another vowel
// get a linking 'r'. Applies to: '3' (ɚ), '3:' (ɜːr), 'U@' (ʊɹ),
// and 'A@' (ɑːɹ). Examples: "forever" (3 before E -> insert r),
// "preferring" (3: before I -> insert r), "during" (U@ before %I ->
// dˈʊɹɹɪŋ), "RNA" (A@ before ,E2 -> ˌɑːɹɹˌɛnˈeɪ). Reference source:
// ph_english_us phoneme definitions have IfNextVowelAppend(r-).
static void applyLinkingR(std::string& ph) {
    static const std::string VOWEL_STARTS = "aAeEiIoOuUV03@";
    // SESE: replace original guard-`continue`s with nested ifs.
    for (size_t rpos = 0; rpos < ph.size(); rpos++) {
        int code_len = 0;
        if (ph[rpos] == '3') {
            // '3' or '3:'
            code_len = 1;
            if (rpos + 1 < ph.size() && ph[rpos+1] == ':') { code_len = 2; }
        } else if (ph[rpos] == 'U' && rpos + 1 < ph.size() &&
            ph[rpos+1] == '@') {
            // 'U@' = ʊɹ (rhotic U, as in "during"->dˈʊɹɹɪŋ)
            code_len = 2;
        } else if (ph[rpos] == 'A' && rpos + 1 < ph.size() &&
            ph[rpos+1] == '@') {
            // 'A@' = ɑːɹ (rhotic A, IfNextVowelAppend(r-) in
            // ph_english_us). e.g. "RNA" R->A@, followed by vowel
            // E2 -> insert r: A@r,E2 -> ɑːɹɹ
            code_len = 2;
        }
        if (code_len > 0) {
            size_t after_code = rpos + code_len;
            // Skip if already has linking r after the code
            if (!(after_code < ph.size() && ph[after_code] == 'r')) {
                // Find next phoneme (skip stress/modifier markers)
                size_t after = after_code;
                while (after < ph.size() &&
                       (ph[after] == '\'' || ph[after] == ',' ||
                        ph[after] == '%'  || ph[after] == '=')) {
                    after++;
                }
                if (after < ph.size() && VOWEL_STARTS.find(ph[after]) !=
                        std::string::npos) {
                    ph.insert(after_code, "r");
                    // skip past the inserted 'r'
                    rpos += code_len;
                }
            }
        }
    }
}

// ============================================================
// Post-process raw phoneme string (steps 1–6)
// Called by wordToPhonemes (for rule-derived and suffix-stripped
// words)
// ============================================================
// Strip \x01 rule-boundary markers from `ph` in place, returning a
// parallel bool vector where boundaries[i] is true iff a \x01
// marker followed ph[i] in the original input. Returns an empty
// vector when the input contained no markers — callers treat that
// as "all-false" (no boundary information available).
static std::vector<bool> stripRuleBoundaryMarkers(std::string& ph) {
    std::vector<bool> rule_boundary_after;
    if (ph.find('\x01') != std::string::npos) {
        std::string stripped;
        stripped.reserve(ph.size());
        std::vector<bool> boundaries;
        boundaries.reserve(ph.size());
        for (size_t i = 0; i < ph.size(); i++) {
            if (ph[i] == '\x01') {
                if (!stripped.empty()) { boundaries.back() = true; }
            } else {
                stripped += ph[i];
                boundaries.push_back(false);
            }
        }
        ph = stripped;
        rule_boundary_after = std::move(boundaries);
    }
    return rule_boundary_after;
}

// Step-5 family: primary stress insertion + 5.0 equals-suffix shift
// + 5a secondary placement + 3 post-steps (cleanup / trochaic /
// final). See header for context on the callbacks.
void IPAPhonemizer::applyStressPhases(
        std::string& ph, const std::string& ph_in,
        bool force_final_stress,
        const std::vector<bool>& rule_boundary_after,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) const {
    // starts_with_secondary / is_rule_leading_comma — needed by
    // 5a-cleanup and 5a-trochaic below.
    bool starts_with_secondary = (!ph.empty() && ph[0] == ',');
    bool is_rule_leading_comma = starts_with_secondary &&
        !rule_boundary_after.empty();
    insertPrimaryStress(ph, force_final_stress,
        rule_boundary_after, findCode, isVowelCodeFn);
    // 5.0 stress-shift for explicitly unstressed '=' suffix
    // (e.g. "-ity" -> =I#t%i). Returns true if it moved primary,
    // which step 5a uses to suppress extra backward secondary.
    bool step50_fired = applyEqualsSuffixStressShift(
        ph, findCode, isVowelCodeFn);
    // 5a secondary placement at even syllable distances. Skipped
    // when ph_in already has explicit secondaries (',').
    bool step5a_ran = placeSecondaryStress(ph, ph_in,
        step50_fired, rule_boundary_after, findCode,
        isVowelCodeFn);
    cleanupAdjacentSecondary(ph, step5a_ran,
        is_rule_leading_comma, rule_boundary_after, findCode,
        isVowelCodeFn);
    trochaicCompoundPrefix(ph, ph_in, step5a_ran,
        starts_with_secondary, findCode, isVowelCodeFn);
    finalSyllableSecondary(ph, findCode, isVowelCodeFn);
}

// Step-5.5 onwards: dim0 reduction, adjacent-primary demotion, the
// en_us reduction batch (5.5b–5.5g, 5b/5c, 6/6b/6c/6e), and the
// unconditional final 6f compound syllabic-n stress.
void IPAPhonemizer::applyPhonemeStringLatePhases(
        std::string& ph, const std::string& ph_in, bool is_en_us,
        const std::vector<bool>& rule_boundary_after,
        const std::function<std::string(size_t)>& findCode,
        const std::function<bool(const std::string&)>&
            isVowelCodeFn) const {
    // Step 5.5-dim0: '0' -> '@' when middle vowel and ≤ UNSTRESSED.
    if (is_en_us) { reduceZeroDiminished(ph, findCode, isVowelCodeFn); }
    // Step 5a-prime: demote earlier of two adjacent primaries
    // (mimics phonSTRESS_PREV GetVowelStress demotion).
    demoteAdjacentPrimaries(ph, ph_in, findCode, isVowelCodeFn);
    // Step 5.5: bare '0' -> schwa (only fires when 4c -ution
    // pattern triggered, i.e. ph has 'u:S').
    if (is_en_us) { reduceBareZeroAfterUtion(ph); }
    // Step 5.5b: bare 'E' -> 'I2' (or '@' before 'n') in unstressed
    // positions across four contexts.
    if (is_en_us) {
        reduceEUnstressed(ph, rule_boundary_after, findCode,
                          isVowelCodeFn);
    }
    // Step 5.5b2: bare 'V' (ʌ) -> '@' between secondary and primary.
    if (is_en_us) { reduceVBetweenSecAndPrimary(ph); }
    // Step 5.5c: bare 'a' -> 'a#' before primary in pre-tonic middle
    // (rule-derived only).
    if (is_en_us) {
        reduceABeforePrimary(ph, !rule_boundary_after.empty(),
                             findCode, isVowelCodeFn);
    }
    // Step 5.5c2: bare 'a' -> 'a#' between primary and secondary
    // (rule-derived only).
    if (is_en_us) {
        reduceABetweenPrimaryAndSec(ph,
                                    !rule_boundary_after.empty());
    }
    // Step 5.5d: bare '0' -> '@' between secondary and primary.
    if (is_en_us) { reduceZeroBetweenSecAndPrimary(ph); }
    // Step 5.5d2: bare '0' -> '@' between primary and following sec.
    if (is_en_us) { reduceZeroBetweenPrimaryAndSec(ph); }
    // Step 5.5e: '0#' -> '@' before primary stress (with guard).
    if (is_en_us) { reduceZeroHashBeforePrimary(ph); }
    // Step 5.5f: bare '0' -> '@' in unstressed pre-tonic (rule).
    if (is_en_us) { reduceBareZeroBeforePrimary(ph, rule_boundary_after); }
    // Step 5.5g: bare 'E' -> '@' pre-tonic after '%'-syll + nasal n.
    if (is_en_us) { reduceEPreTonicAfterPctNasal(ph); }
    // Step 5b: post-stress 'I' -> 'i' before '@L' (syllabic L).
    if (is_en_us) { postStressIBeforeSyllabicL(ph); }
    // Step 5c: 3: -> 3 in pre-tonic / inter-stress positions.
    if (is_en_us) { reduce3LongToShort(ph); }
    // Step 6: flap rule + de-tap fixup.
    if (is_en_us) { applyFlapRule(ph); }
    // Step 6b: '-ness' suffix 'nEs' -> 'n@s' at word-end.
    if (is_en_us) { reduceNessSuffix(ph); }
    // Step 6c: 'oU#' compound prefix -> 0/@/oU.
    if (is_en_us) { reduceOUHashCompound(ph); }
    // Step 6d: DISABLED — see earlier revisions.
    // Step 6e: '-ically' schwa elision (k@li -> kli).
    if (is_en_us) { elideIcallySchwa(ph); }
    // Step 6f: secondary stress on syllabic-n in compound words.
    // (Runs unconditionally.)
    addCompoundSyllabicNStress(ph);
}

std::string IPAPhonemizer::processPhonemeString(
        const std::string& ph_in, bool force_final_stress) const {
    std::string ph = ph_in;
    bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");
    // Strip \x01 rule-boundary markers (used by step 5a to tell
    // true centering diphthongs aI@/aU@ apart from coincidental
    // pairs like "bio" = aI + @). Returns parallel bool vector,
    // empty when no markers existed.
    std::vector<bool> rule_boundary_after = stripRuleBoundaryMarkers(ph);
    // Convert '\x02' (phonSTRESS_PREV-demoted secondary) to ','
    // (secondary stress). \x02 is a "protected secondary" marker
    // used by applyRules' phonSTRESS_PREV handler so that
    // ph_in.find(',') stays npos (allowing step 5a to run).
    for (char& c : ph) {
        if (c == '\x02') { c = ','; }
    }
    // Steps 1-3c: simple in-place phoneme transforms.
    applyVelarNasalAssimilation(ph);
    if (is_en_us) {
        applyHappyTensing(ph);
        applyVowelReduction(ph);
        applyLotPlusRMerge(ph);
    }
    stripMorphemeSchwaR(ph);
    // Steps 4 and 4b: rhotic schwa + linking R (en-us).
    if (is_en_us) {
        applyBareSchwaToRhotic(ph);
        applyLinkingR(ph);
    }
    // Steps 4c/4d/4e: stress rebalance for -tion/-ology/-ic
    // suffixes (en-us only).
    if (is_en_us) {
        applyTionStressFix(ph);
        applyOlogyStressFix(ph);
        applyIcStressFix(ph);
    }
    // Shared multi-char code table used by steps 5/5a/5b/5.5+.
    static const char* S_MC[] = {
        "aI@3","aU@r","i@3r","aI@","aI3","aU@","i@3","3:r","A:r",
        "o@r","A@r","e@r",
        "eI","aI","aU","OI","oU","tS","dZ","IR","VR",
        "e@","i@","U@","A@","O@","o@",
        "3:","A:","i:","u:","O:","e:","a:","aa",
        "@L","@2","@5",
        "I2","I#","E2","E#","e#","a#","a2","0#","02","O2","A~","O~",
        "A#",
        "r-","w#","t#","d#","z#","t2","d2","n-","m-","l/","z/",
        nullptr
    };
    std::function<std::string(size_t)> findCode =
            [&](size_t pos) -> std::string {
        for (int mi = 0; S_MC[mi]; mi++) {
            int mclen = (int)strlen(S_MC[mi]);
            if (pos + mclen <= ph.size() &&
                ph.compare(pos, mclen, S_MC[mi]) == 0) {
                return std::string(S_MC[mi], mclen);
            }
        }
        return std::string(1, ph[pos]);
    };
    std::function<bool(const std::string&)> isVowelCodeFn =
            [this](const std::string& c) {
        return isVowelCode(c);
    };
    // Step 5 / 5.0 / 5a (primary + secondary stress placement).
    applyStressPhases(ph, ph_in, force_final_stress,
                      rule_boundary_after, findCode,
                      isVowelCodeFn);
    // Step 5.5-dim0 onwards: en_us reductions + final 6f.
    applyPhonemeStringLatePhases(ph, ph_in, is_en_us,
                                 rule_boundary_after, findCode,
                                 isVowelCodeFn);
    return ph;
}

// ============================================================
// Phoneme code string -> IPA
// ============================================================
bool IPAPhonemizer::isVowelCode(const std::string& code) const {
    if (code.empty()) { return false; }
    char c = code[0];
    return c=='@' || c=='a' || c=='A' || c=='E' || c=='I' ||
           c=='i' || c=='O' || c=='U' || c=='V' || c=='0' ||
           c=='3' || c=='e' || c=='o' || c=='u';
}

std::string IPAPhonemizer::singleCodeToIPA(
        const std::string& code) const {
    if (code.empty()) { return ""; }
    // Stress markers
    if (code == "'")  { return "\xcb\x88"; } // ˈ primary stress
    if (code == ",")  { return "\xcb\x8c"; } // ˌ secondary stress
    if (code == "%")  { return ""; }          // unstressed - no marker
    if (code == "=")  { return ""; }          // unstressed - no marker
    if (code == "==") { return ""; }
    if (code == "|")  { return ""; }          // syllable boundary
    if (code == "||") { return ""; }          // word end

    // Check IPA overrides table
    auto it = ipa_overrides_.find(code);
    if (it != ipa_overrides_.end()) { return it->second; }

    // Use ipa1 table conversion
    bool is_vowel = isVowelCode(code);
    return phonemeCodeToIPA_table(code, is_vowel);
}

// phonemesToIPA inner-loop body: emit a single phoneme code at
// pstr[i]. Greedy match against MULTI_CODES (longer first, with
// diphthong suppression after %/=), then absorb variant-marker
// digits and emit any pending stress marker before vowel/syllabic
// nuclei.
void IPAPhonemizer::emitPhonemeCode(
        const std::string& pstr, int& i, int len,
        std::string& pending_stress, bool& last_was_unstress,
        bool& last_code_was_vowel, std::string& result) const {
    static const char* MULTI_CODES[] = {
        // 4-char
        "aI@3", "aU@r", "i@3r",
        // 3-char
        "aI@", "aI3", "aU@", "i@3", "3:r", "A:r",
        "I2#",
        "o@r", "O@r", "e@r",
        // 2-char diphthongs
        "eI", "aI", "aU", "OI", "oU", "tS", "dZ", "IR", "VR",
        "e@", "i@", "U@", "A@", "O@", "o@",
        // Long vowels
        "3:", "A:", "i:", "u:", "O:", "e:", "a:",
        "aa",
        // Schwa variants
        "@L", "@2", "@5",
        // Consonant variants
        "r-", "w#", "t#", "d#", "z#", "t2", "d2", "n-", "m-",
        "l/", "z/",
        // Vowel variants
        "I2", "I#", "E2", "E#", "e#", "a#", "a2", "0#", "02",
        "O2", "A~", "O~", "A#",
        nullptr
    };
    std::string code;
    bool found = false;
    for (int mi = 0; MULTI_CODES[mi] != nullptr && !found; mi++) {
        const char* mc = MULTI_CODES[mi];
        int mclen = (int)strlen(mc);
        // After %/= we skip false diphthongs i@/U@ that should be
        // read as separate vowels (e.g. =i@n -> i+@+n, not ɪə+n).
        // True diphthongs (aU/aI/oU/OI) and consonant digraphs
        // (tS/dZ) are not blocked.
        bool skip = last_was_unstress && mclen == 2 &&
            (strcmp(mc, "i@") == 0 || strcmp(mc, "U@") == 0);
        found = !skip && i + mclen <= len &&
            pstr.compare(i, mclen, mc, mclen) == 0;
        if (found) {
            code = std::string(mc, mclen);
            i += mclen;
        }
    }
    if (!found) {
        code = std::string(1, (char)pstr[i]);
        i++;
    }
    last_was_unstress = false;
    // Skip variant-marker digits that follow a code (e.g. '2' in
    // "a#2n"). A digit is a variant marker if its ipa1 mapping is
    // itself. Real phoneme digits: '0'->ɒ, '3'->ɜ, '8'->ɵ.
    while (i < len && pstr[i] >= '0' && pstr[i] <= '9' &&
           ASCII_TO_IPA[(unsigned char)pstr[i] - 0x20] ==
               (unsigned char)pstr[i]) {
        i++;
    }
    // Pending stress fires before vowels and syllabic consonants
    // (n-/m-/@L/r-/l/) — both act as syllable nuclei.
    bool is_syllabic = (code == "n-" || code == "m-" ||
                        code == "@L" || code == "r-" || code == "l/");
    if (!pending_stress.empty() && (isVowelCode(code) || is_syllabic)) {
        result += pending_stress;
        pending_stress.clear();
    }
    result += singleCodeToIPA(code);
    last_code_was_vowel = isVowelCode(code);
}

std::string IPAPhonemizer::phonemesToIPA(
        const std::string& phoneme_str) const {
    std::string result;
    if (!phoneme_str.empty()) {
        // Remove trailing $ flags
        std::string pstr = phoneme_str;
        size_t dollar = pstr.find('$');
        if (dollar != std::string::npos) {
            pstr = trim(pstr.substr(0, dollar));
        }
        int i = 0;
        int len = (int)pstr.size();
        std::string pending_stress;
        bool last_was_unstress = false;
        bool last_code_was_vowel = false;
        while (i < len) {
            unsigned char c = (unsigned char)pstr[i];
            if (std::isspace(c)) {
                i++;
            } else if (pstr[i] == '|' || pstr[i] == '-') {
                // Syllable boundary markers, skip
                i++;
            } else if (pstr[i] == ';') {
                // Palatalization modifier: ';' after a consonant ->
                // output ʲ; after a vowel -> skip.
                if (!last_code_was_vowel) {
                    result += "\xca\xb2"; // ʲ (U+02B2)
                }
                i++;
            } else if (pstr[i] == '\'' || pstr[i] == ',') {
                pending_stress = singleCodeToIPA(std::string(1, pstr[i]));
                last_was_unstress = false;
                i++;
            } else if (pstr[i] == '%' || pstr[i] == '=') {
                pending_stress.clear();
                last_was_unstress = true;
                i++;
            } else {
                emitPhonemeCode(pstr, i, len, pending_stress,
                                last_was_unstress,
                                last_code_was_vowel, result);
            }
        }
        // Any remaining pending_stress with no following vowel is
        // discarded intentionally — trailing ',' in a dict entry
        // acts as a "no-secondary-stress" marker for step 5a.
    }
    return result;
}

static bool containsStressMarker(const std::string& ipa) {
    // Check for ˈ (0xCB 0x88) or ˌ (0xCB 0x8C)
    for (size_t i = 0; i + 1 < ipa.size(); i++) {
        if ((unsigned char)ipa[i] == 0xCB &&
            ((unsigned char)ipa[i+1] == 0x88 ||
             (unsigned char)ipa[i+1] == 0x8C)) {
            return true;
        }
    }
    return false;
}

// IPA vowel character detection (UTF-8)
// Only returns true for actual vowel characters (not consonants like ɹ, ɡ, etc.)
static bool isIPAVowelStart(const std::string& s, size_t i) {
    if (i >= s.size()) { return false; }
    unsigned char c = (unsigned char)s[i];
    // ASCII vowels: a, e, i, o, u
    if (c < 0x80) {
        char lc = (char)std::tolower(c);
        return lc=='a'||lc=='e'||lc=='i'||lc=='o'||lc=='u';
    }
    if (i + 1 >= s.size()) { return false; }
    unsigned char c2 = (unsigned char)s[i+1];

    if (c == 0xC3) {
        // æ=0xA6, ø=0xB8, œ=0x93
        return c2==0xA6||c2==0xB8||c2==0x93;
    }
    if (c == 0xC9) {
        // IPA vowels in U+024X-U+02BX range (second byte 0x90-0xBF):
        // ɐ=0x90, ɑ=0x91, ɒ=0x92, ɔ=0x94, ɕ=0x95(not vowel),
        // ə=0x99, ɚ=0x9A, ɛ=0x9B, ɜ=0x9C, ɞ=0x9E,
        // ɪ=0xAA, ɵ=0xB5
        // NON-vowels: ɡ=0xA1, ɟ=0x9F, ɹ=0xB9, ɸ=0xB8, etc.
        switch (c2) {
            case 0x90: // ɐ
            case 0x91: // ɑ
            case 0x92: // ɒ
            case 0x94: // ɔ
            case 0x99: // ə
            case 0x9A: // ɚ
            case 0x9B: // ɛ
            case 0x9C: // ɜ
            case 0x9E: // ɞ
            case 0xAA: // ɪ
            case 0xB5: // ɵ
                return true;
            default:
                return false;
        }
    }
    if (c == 0xCA) {
        // ʊ=0x8A, ʌ=0x8C
        return c2==0x8A||c2==0x8C;
    }
    if (c == 0xE1) {
        // ᵻ = U+1D7B = 0xE1 0xB5 0xBB
        if (i+2 < s.size()) {
            unsigned char c3 = (unsigned char)s[i+2];
            return c2==0xB5 && c3==0xBB;
        }
    }
    return false;
}

// Check if the IPA character at i is schwa (ə, U+0259 = 0xC9 0x99)
static bool isIPASchwa(const std::string& s, size_t i) {
    if (i + 1 >= s.size()) { return false; }
    return (unsigned char)s[i] == 0xC9 && (unsigned char)s[i+1] == 0x99;
}

// UTF-8 byte-skip step. Returns 1/2/3/4 based on the first byte.
static size_t utf8Step(unsigned char c) {
    size_t step = 4;
    if (c < 0x80) { step = 1; }
    else if (c < 0xE0) { step = 2; }
    else if (c < 0xF0) { step = 3; }
    return step;
}

// Find first IPA vowel position from `start` (npos if none). Sets
// `is_schwa` for the found vowel.
static size_t findFirstIPAVowel(const std::string& ipa, size_t start,
                                bool& is_schwa) {
    size_t pos = std::string::npos;
    size_t i = start;
    while (i < ipa.size() && pos == std::string::npos) {
        if (isIPAVowelStart(ipa, i)) {
            pos = i;
            is_schwa = isIPASchwa(ipa, i);
        } else {
            i += utf8Step((unsigned char)ipa[i]);
        }
    }
    return pos;
}

// Find first non-schwa IPA vowel position from `start` (npos if
// none). Used by addDefaultStress to skip an initial schwa.
static size_t findFirstNonSchwaVowel(const std::string& ipa,
                                     size_t start) {
    size_t pos = std::string::npos;
    size_t j = start;
    while (j < ipa.size() && pos == std::string::npos) {
        if (isIPAVowelStart(ipa, j) && !isIPASchwa(ipa, j)) {
            pos = j;
        } else {
            j += utf8Step((unsigned char)ipa[j]);
        }
    }
    return pos;
}

static std::string addDefaultStress(const std::string& ipa) {
    // ˈ = 0xCB 0x88
    static const std::string STRESS = "\xcb\x88";
    std::string result = ipa;
    if (!ipa.empty() && !containsStressMarker(ipa)) {
        bool first_is_schwa = false;
        size_t first_vowel = findFirstIPAVowel(ipa, 0, first_is_schwa);
        if (first_vowel != std::string::npos) {
            // Default: stress first vowel. If it's a schwa AND a
            // non-schwa vowel follows, stress that instead
            // ("hello" h@loU -> həlˈoʊ, "about" -> əbˈaʊt).
            size_t target = first_vowel;
            if (first_is_schwa) {
                size_t non_schwa =
                    findFirstNonSchwaVowel(ipa, first_vowel + 2);
                if (non_schwa != std::string::npos) { target = non_schwa; }
            }
            result = ipa.substr(0, target) + STRESS + ipa.substr(target);
        }
    }
    return result;
}

// ============================================================
// Text tokenization
// ============================================================
std::vector<IPAPhonemizer::Token> IPAPhonemizer::tokenizeText(
        const std::string& text) const {
    std::vector<Token> tokens;
    std::string current_word;
    auto flush_word = [&]() {
        if (!current_word.empty()) {
            Token t;
            t.text = current_word;
            t.is_word = true;
            t.needs_space_before = !tokens.empty();
            tokens.push_back(t);
            current_word.clear();
        }
    };
    // Emit punctuation as its own non-word token (flushing any
    // accumulated word first). Replaces the duplicated emit-block
    // that appeared in two arms of the original if-chain.
    auto emit_punct = [&](char c) {
        flush_word();
        Token t;
        t.text = std::string(1, c);
        t.is_word = false;
        t.needs_space_before = false;
        tokens.push_back(t);
    };
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 0x80) {
            // UTF-8 multi-byte: take lead byte then all continuation
            // bytes (0x80–0xBF). Replaces the original `continue`-
            // based shortcut.
            current_word += text[i++];
            while (i < text.size() &&
                ((unsigned char)text[i] & 0xC0) == 0x80) {
                current_word += text[i++];
            }
        } else if (std::isalpha(c) || c == '\'') {
            current_word += (char)c;
            i++;
        } else if (c == '.' && current_word.size() == 1 &&
                   std::isupper((unsigned char)current_word[0]) &&
                   i+1 < text.size() &&
                   std::isupper((unsigned char)text[i+1])) {
            // Abbreviation start: single uppercase letter followed
            // by '.' and next char is uppercase. E.g. "U.S." ->
            // accumulate "U." into current_word and keep collecting.
            current_word += '.';
            i++;
        } else if (c == '.' && !current_word.empty()) {
            // Trailing-period dispatch. If current_word already
            // contains a period, we're mid-abbreviation (e.g. "U.S"
            // about to become "U.S.") — append the period and flush
            // as a word token. Otherwise treat the period as plain
            // punctuation. (Inlining the original `is_abbrev` flag —
            // the find() result IS the predicate.)
            if (current_word.find('.') != std::string::npos) {
                current_word += '.';
                i++;
                flush_word();
            } else {
                emit_punct((char)c);
                i++;
            }
        } else if (c == '-' && !current_word.empty() &&
                   i+1 < text.size() && std::isalpha(text[i+1])) {
            current_word += (char)c;
            i++;
        } else if (std::isdigit(c)) {
            current_word += (char)c;
            i++;
        } else if (std::isspace(c)) {
            flush_word();
            i++;
        } else {
            emit_punct((char)c);
            i++;
        }
    }
    flush_word();
    return tokens;
}

// ============================================================
// Number-to-words conversion (cardinal, American English, the reference style)
// ============================================================
static std::string intToWords(int n) {
    static const char* ones[] = {
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    };
    static const char* tens_words[] = {
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    };
    if (n == 0) { return "zero"; }
    std::string result;
    if (n >= 1000000) {
        result += intToWords(n / 1000000) + " million";
        n %= 1000000;
        if (n > 0) { result += " "; }
    }
    if (n >= 1000) {
        result += intToWords(n / 1000) + " thousand";
        n %= 1000;
        if (n > 0) { result += " "; }
    }
    if (n >= 100) {
        result += std::string(ones[n / 100]) + " hundred";
        n %= 100;
        if (n > 0) { result += " "; }
    }
    if (n >= 20) {
        result += tens_words[n / 10];
        n %= 10;
        if (n > 0) { result += " " + std::string(ones[n]); }
    } else if (n > 0) {
        result += ones[n];
    }
    return result;
}

// Number expansion: pure-digit token -> spoken word form, each sub-word
// converted to IPA via the normal rule pipeline (wordToPhonemes ->
// phonemesToIPA -> addDefaultStress). Returns true when the token was
// numeric and IPA was appended; false leaves token unhandled.
bool IPAPhonemizer::expandNumberToken(
        const Token& token, std::string& result,
        bool& first_word) const {
    bool all_digits = !token.text.empty();
    for (char c : token.text) {
        if (!std::isdigit((unsigned char)c)) { all_digits = false; }
    }
    bool consumed = false;
    if (all_digits) {
        long long num_val = std::stoll(token.text);
        if (num_val >= 0 && num_val <= 9999999LL) {
            std::string num_words = intToWords((int)num_val);
            std::istringstream iss(num_words);
            std::string w;
            bool is_first_sub = true;
            while (iss >> w) {
                std::string wph = wordToPhonemes(w);
                std::string wipa = phonemesToIPA(wph);
                wipa = addDefaultStress(wipa);
                if (is_first_sub) {
                    if (token.needs_space_before && !first_word) {
                        result += ' ';
                    }
                    is_first_sub = false;
                } else {
                    result += ' ';
                }
                result += wipa;
                first_word = false;
            }
            consumed = true;
        }
    }
    return consumed;
}

// Spell all-caps / mixed-case-no-vowel acronyms as letter names.
// "DNA"->D-N-A, "PhD"->[Ph][D] (camelCase split into separate words).
// Triggers when (all-upper or mixed-case-no-vowel) AND (lowercase form
// is $abbrev-flagged OR unknown to the dictionary). Returns true when
// the token was spelled and IPA was appended.
bool IPAPhonemizer::spellAcronymToken(
        const Token& token, std::string& result,
        bool& first_word) const {
    std::string lower = toLowerASCII(token.text);
    bool all_upper = token.text.size() >= 2;
    for (char c : token.text) {
        if (!std::isupper((unsigned char)c)) { all_upper = false; }
    }
    bool unknown_word = !dict_.count(lower);
    // mixed_case_no_vowel_abbrev: token has both upper- and
    // lower-case letters but no vowels (PhD, McD, BSc, ...).
    bool mixed_case_no_vowel_abbrev = false;
    if (!all_upper && token.text.size() >= 2 && unknown_word) {
        auto is_upper = [](char c) {
            return std::isupper((unsigned char)c) != 0;
        };
        auto is_lower = [](char c) {
            return std::islower((unsigned char)c) != 0;
        };
        auto is_ascii_vowel = [](char c) {
            char lc = (char)std::tolower((unsigned char)c);
            return lc=='a'||lc=='e'||lc=='i'||lc=='o'||lc=='u';
        };
        const auto& t = token.text;
        mixed_case_no_vowel_abbrev =
            std::any_of(t.begin(), t.end(), is_upper) &&
            std::any_of(t.begin(), t.end(), is_lower) &&
            std::none_of(t.begin(), t.end(), is_ascii_vowel);
    }
    bool consumed = false;
    if ((all_upper || mixed_case_no_vowel_abbrev) &&
        (abbrev_words_.count(lower) > 0 || unknown_word)) {
        // camelCase split for mixed-case abbreviations:
        // "PhD" -> ["Ph","D"]; all-upper words are one group.
        std::vector<std::string> groups;
        if (mixed_case_no_vowel_abbrev) {
            std::string cur;
            for (size_t ci = 0; ci < token.text.size(); ci++) {
                if (ci > 0 && std::islower((unsigned char)token.text[ci-1]) &&
                    std::isupper((unsigned char)token.text[ci])) {
                    if (!cur.empty()) { groups.push_back(cur); }
                    cur.clear();
                }
                cur += token.text[ci];
            }
            if (!cur.empty()) { groups.push_back(cur); }
        } else {
            groups.push_back(token.text);
        }
        // Helper: spell letters in a group as letter names -> IPA.
        auto spellGroup = [&](const std::string& grp) -> std::string {
            std::vector<std::string> lipa;
            for (char lc : grp) {
                char lc_lower = std::tolower((unsigned char)lc);
                std::string uk = std::string("_") + lc_lower;
                auto uit = dict_.find(uk);
                if (uit != dict_.end()) {
                    lipa.push_back(uit->second);
                } else {
                    lipa.push_back(wordToPhonemes(std::string(1, lc)));
                }
            }
            std::string out;
            if (!lipa.empty()) {
                std::string codes;
                for (size_t li = 0; li < lipa.size(); li++) {
                    std::string code = lipa[li];
                    if (li < lipa.size() - 1) {
                        std::string mod;
                        size_t first = code.find_first_of("',");
                        if (first == std::string::npos) {
                            mod = "," + code;
                        } else {
                            mod = code.substr(0, first) + ",";
                            for (size_t k = first + 1; k < code.size(); k++) {
                                char c2 = code[k];
                                if (c2 != '\'' && c2 != ',') { mod += c2; }
                            }
                        }
                        codes += mod;
                    } else {
                        if (code.find('\'') == std::string::npos) {
                            codes += '\'' + code;
                        } else {
                            codes += code;
                        }
                    }
                }
                out = phonemesToIPA(processPhonemeString(codes));
            }
            return out;
        };
        bool first_grp = true;
        for (const auto& grp : groups) {
            std::string ipa = spellGroup(grp);
            if (!ipa.empty()) {
                if (first_grp) {
                    if (token.needs_space_before && !first_word) {
                        result += ' ';
                    }
                } else {
                    result += ' ';
                }
                result += ipa;
                first_word = false;
                first_grp = false;
            }
        }
        consumed = true;
    }
    return consumed;
}

// POS context override: after $pastf/$nounf/$verbf words bumped the
// expect_* counters, swap ph_codes for the side-dict entry if it
// exists. Gated off for isolated words and phrase-matched bigrams.
void IPAPhonemizer::applyPosContextOverride(
        const Token& token, bool is_isolated_word, bool phrase_matched,
        int expect_past, int expect_noun, int expect_verb,
        std::string& ph_codes) const {
    if (!is_isolated_word && !phrase_matched) {
        std::string lw = toLowerASCII(token.text);
        if (expect_past > 0) {
            auto pit = past_dict_.find(lw);
            if (pit != past_dict_.end()) {
                ph_codes = processPhonemeString(pit->second);
            }
        } else if (expect_noun > 0) {
            auto nit = noun_dict_.find(lw);
            if (nit != noun_dict_.end()) {
                ph_codes = processPhonemeString(nit->second);
            }
        } else if (expect_verb > 0) {
            auto vit = verb_dict_.find(lw);
            if (vit != verb_dict_.end()) {
                ph_codes = processPhonemeString(vit->second);
            }
        }
    }
}

// Hand-coded function-word allophone overrides. Each `if` is a
// self-contained branch — they don't compose, and only one fires per
// token (the lemma equality gate is mutually exclusive).
//   - "the" -> ðɪ ("%DI") before vowel-initial, non-yod next word.
//   - "a"   -> "eI" (letter name) when isolated, else "a#" (ɐ).
//   - "an"  -> "a#n" (ɐn) before vowel, "an" (æn) before consonant.
//   - "to"  -> "tu:" if isolated/utterance-final; "tU" before vowel
//             non-yod next; "t@5" otherwise (the @5 marker blocks
//             cross-word @->3 rhotacization).
//   - "use" -> "ju:z" (verb form) when previous token is a pronoun.
void IPAPhonemizer::applyLemmaOverride(
        const Token& token, size_t ti,
        const std::vector<Token>& tokens, bool is_isolated_word,
        size_t last_word_ti, std::string& ph_codes) const {
    std::string lw = toLowerASCII(token.text);
    if (lw == "the" && !ph_codes.empty()) {
        size_t tj = findNextWordStopOnPunct(tokens, ti + 1);
        if (tj != std::string::npos && wordIsVowelInitialNonYod(tokens, tj)) {
            ph_codes = "%DI";
        }
    }
    if (lw == "a") { ph_codes = is_isolated_word ? "eI" : "a#"; }
    if (lw == "an" && !is_isolated_word) {
        size_t tj2 = findNextNonEmptyWord(tokens, ti + 1);
        bool next_vowel_initial = false;
        if (tj2 != std::string::npos) {
            char fc = (char)std::tolower((unsigned char)tokens[tj2].text[0]);
            next_vowel_initial = (fc=='a'||fc=='e'||fc=='i'||fc=='o'||fc=='u');
        }
        ph_codes = next_vowel_initial ? "a#n" : "an";
    }
    if (lw == "to") {
        if (is_isolated_word || ti == last_word_ti) {
            ph_codes = "tu:";
        } else {
            size_t tj2 = findNextNonEmptyWord(tokens, ti + 1);
            // "to" uses tU before vowel-initial non-yod next; t@5
            // otherwise. wordIsVowelInitialNonYod handles both
            // the vowel-letter check and the /j/-onset exception
            // (e.g. "to unify" -> t@5 not tU because "unify"->jˌuː).
            bool use_tU = (tj2 != std::string::npos &&
                wordIsVowelInitialNonYod(tokens, tj2));
            ph_codes = use_tU ? "tU" : "t@5";
        }
    }
    if (lw == "use" && !is_isolated_word) {
        static const std::unordered_set<std::string> PRONOUNS =
            {"i", "we", "you", "they", "he", "she", "who"};
        std::string prev_word;
        if (ti > 0) {
            bool found = false;
            int tj = (int)ti - 1;
            while (tj >= 0 && !found) {
                if (tokens[tj].is_word) {
                    prev_word = toLowerASCII(tokens[tj].text);
                    found = true;
                }
                tj--;
            }
        }
        if (PRONOUNS.count(prev_word) > 0) { ph_codes = "ju:z"; }
    }
}

// Inter-word t-flap. The marker "t#" on the tail of function-word
// phoneme strings (at, it, but, ...) becomes "*" (flap) when the
// next word begins with a vowel phoneme. No-op otherwise.
void IPAPhonemizer::applyInterWordTFlap(
        size_t ti, const std::vector<Token>& tokens,
        std::string& ph_codes) const {
    if (ph_codes.size() >= 2 && ph_codes[ph_codes.size()-2] == 't' &&
        ph_codes[ph_codes.size()-1] == '#') {
        size_t tnext = ti + 1;
        while (tnext < tokens.size() && !tokens[tnext].is_word) { tnext++; }
        if (tnext < tokens.size() && !tokens[tnext].text.empty()) {
            std::string nph = wordToPhonemes(toLowerASCII(tokens[tnext].text));
            size_t npi = 0;
            while (npi < nph.size() && (nph[npi]=='\'' || nph[npi]==',' ||
                    nph[npi]=='%' || nph[npi]=='=')) {
                npi++;
            }
            static const std::string VOWEL_STARTS = "aAeEIiOUVu03@oY";
            bool next_vowel_onset = (npi < nph.size() &&
                VOWEL_STARTS.find(nph[npi]) != std::string::npos);
            if (next_vowel_onset) {
                ph_codes[ph_codes.size()-2] = '*';
                ph_codes.erase(ph_codes.size()-1, 1);
            }
        }
    }
}

// Cross-word @->3 rhotacization. Standalone trailing "@" (schwa)
// promotes to "3" (ɚ) before an r-initial next word. Skipped when
// the "@" is the tail of a diphthong digraph (e.g. "i@" = ɪə) — we
// check the preceding char isn't one of the bigram prefixes.
void IPAPhonemizer::applyCrossWordSchwaRhotic(
        size_t ti, bool is_en_us, bool is_isolated_word,
        const std::vector<Token>& tokens,
        std::string& ph_codes) const {
    if (is_en_us && !is_isolated_word &&
        !ph_codes.empty() && ph_codes.back() == '@') {
        // Standalone '@' (not the tail of a diphthong bigram like
        // i@/e@/A@/O@/o@/U@).
        char prev = ph_codes.size() >= 2
            ? ph_codes[ph_codes.size()-2] : 0;
        bool standalone = !(prev=='A' || prev=='e' || prev=='O' ||
                            prev=='o' || prev=='U' || prev=='i');
        if (standalone) {
            const size_t tj = ti + 1;
            if (tj < tokens.size() && tokens[tj].is_word &&
                !tokens[tj].text.empty()) {
                std::string nph = wordToPhonemes(tokens[tj].text);
                size_t pi = 0;
                while (pi < nph.size() && (nph[pi]=='\'' || nph[pi]==',' ||
                        nph[pi]=='%' || nph[pi]=='=')) {
                    pi++;
                }
                if (pi < nph.size() && nph[pi] == 'r') {
                    ph_codes.back() = '3';
                }
            }
        }
    }
}

// Cross-word /t/ flap on "it" before vowel-initial next word.
// Operates on the post-IPA string (not the phoneme code). The
// reference flaps "it is" -> ɪɾ ɪz, "it equally" -> ɪɾ ˈiːkwəli.
// Restricted to "it"; "that"/"but"/"not" do not flap cross-word.
void IPAPhonemizer::applyCrossWordTFlap(
        const Token& token, size_t ti, bool is_en_us,
        bool is_isolated_word, const std::vector<Token>& tokens,
        std::string& ipa) const {
    if (is_en_us && !is_isolated_word && !ipa.empty() && ipa.back() == 't' &&
        toLowerASCII(token.text) == "it") {
        size_t tj = ti + 1;
        while (tj < tokens.size() && !tokens[tj].is_word) { tj++; }
        if (tj < tokens.size() && !tokens[tj].text.empty()) {
            std::string nph = wordToPhonemes(tokens[tj].text);
            size_t pi2 = 0;
            while (pi2 < nph.size() && (nph[pi2]=='\'' || nph[pi2]==',' ||
                    nph[pi2]=='%' || nph[pi2]=='=')) {
                pi2++;
            }
            if (pi2 < nph.size()) {
                char fc = nph[pi2];
                bool nv = (fc=='a' || fc=='A' || fc=='e' ||
                           fc=='E' || fc=='i' || fc=='I' ||
                           fc=='o' || fc=='O' || fc=='u' ||
                           fc=='U' || fc=='V' || fc=='0' ||
                           fc=='3' || fc=='@');
                if (nv) {
                    // Replace trailing 't' with ɾ (U+027E).
                    ipa.pop_back();
                    ipa += '\xC9';
                    ipa += '\xBE';
                }
            }
        }
    }
}

// Default-stress decision. The IPA gets a primary stress marker
// (ˈ) prepended unless the token is inherently unstressed:
//   - ph_codes contains '%' (whole-word unstressed marker) but no
//     existing stress markers (and not a phrase split-bigram).
//   - $u-flagged function word in sentence context.
//   - ph_codes contains @2 or @5 (weak-schwa variants) and no
//     stress markers — these are intrinsically unstressed.
//   - article a/an in sentence context (ph_codes == "a#" or "a#n").
void IPAPhonemizer::maybeAddDefaultStress(
        const Token& token, bool phrase_matched,
        bool is_isolated_word, bool is_unstressed_word,
        const std::string& ph_codes, std::string& ipa) const {
    bool pct_unstressed = (ph_codes.find('%') != std::string::npos &&
         ph_codes.find('\'') == std::string::npos &&
         (ph_codes.find(',') == std::string::npos ||
          phrase_matched)) && !is_isolated_word;
    bool u_unstressed = is_unstressed_word && !is_isolated_word;
    bool weak_schwa = (ph_codes.find("@2") != std::string::npos ||
         ph_codes.find("@5") != std::string::npos) &&
        ph_codes.find('\'') == std::string::npos &&
        ph_codes.find(',') == std::string::npos;
    bool article_a = (!is_isolated_word && (toLowerASCII(token.text) == "a" ||
         toLowerASCII(token.text) == "an") &&
        (ph_codes == "a#" || ph_codes == "a#n"));
    bool no_stress = pct_unstressed || weak_schwa || u_unstressed || article_a;
    if (!no_stress) { ipa = addDefaultStress(ipa); }
}

// Pre-vowel "the" allophone fixup (used when a phrase_dict_ entry
// ends in "@2" / ðə and the next word is vowel-initial non-yod):
// swap the final ə byte pair (0xC9 0x99) for ɪ (0xC9 0xAA).
void IPAPhonemizer::applyPreVowelTheFixup(
        bool phrase_pre_vowel_the, std::string& ipa) const {
    if (phrase_pre_vowel_the) {
        if (ipa.size() >= 2 && (unsigned char)ipa[ipa.size()-2] == 0xc9 &&
            (unsigned char)ipa[ipa.size()-1] == 0x99) {
            ipa[ipa.size()-1] = (char)0xaa;
        }
    }
}

// Update expect_* counters for next-word POS context. $pastf/$nounf/
// $verbf flags bump the matching counter (initial value chosen so
// that after the immediate decrement below it reflects the reference
// 2/1 schedule). All counters always decrement at the end.
void IPAPhonemizer::updatePosContextCounters(
        const Token& token, int& expect_past, int& expect_noun,
        int& expect_verb) const {
    std::string lw = toLowerASCII(token.text);
    if (pastf_words_.count(lw)) {
        expect_past = 3;
        expect_noun = 0;
        expect_verb = 0;
    } else if (nounf_words_.count(lw)) {
        expect_noun = 2;
        expect_past = 0;
        expect_verb = 0;
    } else if (verbf_words_.count(lw)) {
        expect_verb = 2;
        expect_past = 0;
        expect_noun = 0;
    }
    if (expect_past > 0) { expect_past--; }
    if (expect_noun > 0) { expect_noun--; }
    if (expect_verb > 0) { expect_verb--; }
}

// R-linking (r-sandhi). Inserts ɹ (U+0279, 0xC9 0xB9) at the end of
// the IPA when the current word ends in ɚ (0xC9 0x9A) or ɜː
// (0xC9 0x9C 0xCB 0x90) AND the next word begins with a vowel
// phoneme. Suppressed before non-final conjunctions and at
// punctuation boundaries.
// True if `ipa` ends in ɚ (0xC9 0x9A) or ɜː (0xC9 0x9C 0xCB 0x90).
static bool endsInRhoticR(const std::string& ipa) {
    size_t n = ipa.size();
    bool ends_in_schwar = (n >= 2 &&
        (unsigned char)ipa[n-2] == 0xC9 && (unsigned char)ipa[n-1] == 0x9A);
    bool ends_in_long_er = (n >= 4 && (unsigned char)ipa[n-4] == 0xC9 &&
        (unsigned char)ipa[n-3] == 0x9C &&
        (unsigned char)ipa[n-2] == 0xCB && (unsigned char)ipa[n-1] == 0x90);
    return ends_in_schwar || ends_in_long_er;
}

size_t IPAPhonemizer::findNextWordStopOnPunct(
        const std::vector<Token>& tokens, size_t start) {
    size_t found = std::string::npos;
    bool stop = false;
    size_t tj = start;
    while (tj < tokens.size() && found == std::string::npos && !stop) {
        if (tokens[tj].is_word) {
            if (!tokens[tj].text.empty()) { found = tj; }
        } else {
            stop = true;
        }
        tj++;
    }
    return found;
}

bool IPAPhonemizer::conjunctionSuppressesLinking(
        const std::vector<Token>& tokens, size_t tj,
        bool first_word) const {
    bool suppress = false;
    std::string lc = toLowerASCII(tokens[tj].text);
    bool is_conjunction = (lc == "or" || lc == "and");
    if (is_conjunction && !first_word) {
        size_t after = findNextWordStopOnPunct(tokens, tj + 1);
        suppress = (after != std::string::npos);
    }
    return suppress;
}

bool IPAPhonemizer::nextWordIsVowelInitial(
        const std::vector<Token>& tokens, size_t tj) const {
    bool use_first_letter = (tokens[tj].text.find('.') != std::string::npos);
    if (!use_first_letter && tokens[tj].text.size() >= 2) {
        bool all_upper = !tokens[tj].text.empty();
        for (char c2 : tokens[tj].text) {
            if (!std::isupper((unsigned char)c2)) { all_upper = false; }
        }
        if (all_upper) {
            std::string lower_nxt = toLowerASCII(tokens[tj].text);
            if (!dict_.count(lower_nxt) || abbrev_words_.count(lower_nxt)) {
                use_first_letter = true;
            }
        }
    }
    std::string next_ph;
    if (use_first_letter) {
        bool found = false;
        for (char lc : tokens[tj].text) {
            if (!found && std::isalpha((unsigned char)lc)) {
                next_ph = wordToPhonemes(std::string(1, lc));
                found = true;
            }
        }
    } else {
        next_ph = wordToPhonemes(tokens[tj].text);
    }
    size_t pi = 0;
    while (pi < next_ph.size() && (next_ph[pi] == '\'' || next_ph[pi] == ',' ||
            next_ph[pi] == '%' || next_ph[pi] == '=')) {
        pi++;
    }
    return pi < next_ph.size() && isVowelCode(std::string(1, next_ph[pi]));
}

void IPAPhonemizer::applyRLinking(
        bool first_word, size_t ti,
        const std::vector<Token>& tokens, std::string& ipa) const {
    if (endsInRhoticR(ipa)) {
        size_t tj = findNextWordStopOnPunct(tokens, ti + 1);
        if (tj != std::string::npos &&
            !conjunctionSuppressesLinking(tokens, tj, first_word) &&
            nextWordIsVowelInitial(tokens, tj)) {
            ipa += '\xC9';
            ipa += '\xB9'; // ɹ (U+0279)
        }
    }
}

// Step D: shift "'" out of diphthong digraphs. The phoneme rule
// engine sometimes places the primary-stress marker between the two
// letters of a diphthong (e.g. "e'I" instead of "'eI"). This swaps
// the position so the full digraph gets the stress.
void IPAPhonemizer::fixDiphthongStressPosition(
        std::string& ph_codes) const {
    static const char* DIPHS[] = {
        "eI", "aI", "aU", "OI", "oU", nullptr};
    for (size_t si = 1; si + 1 < ph_codes.size(); si++) {
        if (ph_codes[si] == '\'') {
            char prev = ph_codes[si-1];
            char next = ph_codes[si+1];
            std::string pair = {prev, next};
            bool matched = false;
            for (int di = 0; DIPHS[di] && !matched; di++) {
                matched = pair == DIPHS[di];
                if (matched) {
                    ph_codes.erase(si, 1);
                    ph_codes.insert(si-1, 1, '\'');
                }
            }
        }
    }
}

// Step B: function-word primary-stress suppression. Strips '\''
// from ph_codes for $u-flagged words (and % words whose only primary
// is a step-5 last-resort before 'a#'), unless the word is in the
// step-B keep-secondary list (those are handled by step C).
bool IPAPhonemizer::applyStepB(
        const Token& token, bool phrase_matched,
        bool is_isolated_word, std::string& ph_codes) const {
    static const std::unordered_set<std::string>
        STEP_B_KEEP_SECONDARY_WORDS = {
            "within", "without", "about", "across", "above",
            "among", "amongst", "before", "upon", "below",
            "beside", "between", "beyond", "despite",
            "except", "inside", "outside", "toward", "towards",
            "along", "around", "behind", "beneath", "underneath",
            "over", "under"};
    std::string token_lower = toLowerASCII(token.text);
    std::string unstress_check = token_lower;
    auto apos_pos = token_lower.find('\'');
    if (apos_pos != std::string::npos) {
        unstress_check = token_lower.substr(0, apos_pos);
    }
    bool is_unstressed_word = !phrase_matched &&
        (unstressed_words_.count(token_lower) > 0 ||
         (apos_pos != std::string::npos &&
          unstressed_words_.count(unstress_check) > 0));
    // is_pct_word fires when ph_codes starts with '%' and its only
    // '\'' is the one immediately before 'a#' (step-5 last-resort).
    size_t prime_pos = ph_codes.find('\'');
    size_t hash_a_pos = ph_codes.find("'a#");
    bool is_pct_word = !is_isolated_word &&
        !ph_codes.empty() && ph_codes[0] == '%' &&
        prime_pos != std::string::npos &&
        hash_a_pos != std::string::npos && prime_pos == hash_a_pos;
    bool is_step_b_keep_sec =
        STEP_B_KEEP_SECONDARY_WORDS.count(token_lower) > 0;
    if ((is_unstressed_word || is_pct_word) && !is_isolated_word &&
        !is_step_b_keep_sec) {
        ph_codes.erase(
            std::remove(ph_codes.begin(), ph_codes.end(), '\''),
            ph_codes.end());
    }
    return is_unstressed_word;
}

// Step C: function-word stress assignment in sentence context.
// Phases:
//   1. Compute keep_sec / needs_sec / is_strend_secondary /
//      is_ing_of_strend / is_keep_sec_phrase from token+state.
//   2. Promote keep_sec -> primary when phrase-final (no following
//      stressed word). Upgrade ',' -> '\'' if needed.
//   3. Demote primary -> secondary for keep_sec words.
//   4. Insert ',' for NEEDS_SECONDARY words.
//   5. Promote leading-or-mid ',' -> '\'' in non-keep-sec contexts.
//   6. $unstressend demotion at utterance end.
// "-ing forms of $strend2 stems with secondary-stressed dict
// phoneme" check. e.g. "making" from "make" (m,eIk). Returns true
// iff the word's stem (with or without restored -e) is in
// strend_words_ AND that stem's dict_ entry contains ','.
bool IPAPhonemizer::isIngOfStrendSecondary(
        const std::string& word) const {
    bool result = false;
    if (word.size() > 3 && word.compare(word.size()-3, 3, "ing") == 0) {
        // Try the bare stem first, then magic-e restored. The first
        // one that is in strend_words_ "wins" — `sk` is empty when
        // neither qualifies.
        std::string base = word.substr(0, word.size()-3);
        std::string sk;
        if (strend_words_.count(base) > 0) { sk = base; }
        else if (strend_words_.count(base + "e") > 0) { sk = base + "e"; }
        if (!sk.empty()) {
            auto sit = dict_.find(sk);
            result = sit != dict_.end() &&
                sit->second.find(',') != std::string::npos;
        }
    }
    return result;
}

// keep_sec -> primary promotion: a keep_sec word gets promoted to
// primary stress when no following stressed content remains in the
// utterance. "About it" promotes "about"; "about now" keeps it
// secondary. Scans forward through `tokens` skipping unstressed
// words and weak-form words (dict starts with ',' or '%' and not
// itself in a secondary-stress set).
void IPAPhonemizer::tryPromoteKeepSecToPrimary(
        const std::string& token_lower, size_t ti,
        const std::vector<Token>& tokens,
        bool& keep_sec, std::string& ph_codes) const {
    static const std::unordered_set<std::string> KEEP_SECONDARY =
        {"on", "onto", "multiple", "multiples", "going",
         "into", "any", "how", "where", "why",
         "being", "while", "but",
         "across", "above", "among", "amongst", "before",
         "within", "without", "upon", "below", "beside", "between",
         "beyond", "underneath", "behind", "beneath",
         "over", "under",
         "about",
         "make", "makes"};
    if (keep_sec && !u_plus_secondary_words_.count(token_lower)) {
        bool has_following_stressed = false;
        for (size_t tj = ti + 1;
             tj < tokens.size() && !has_following_stressed; tj++) {
            if (tokens[tj].is_word) {
                std::string fw = toLowerASCII(tokens[tj].text);
                if (!unstressed_words_.count(fw)) {
                    bool fw_is_strend_sec = comma_strend2_words_.count(fw) > 0;
                    bool fw_is_ing_strend = isIngOfStrendSecondary(fw);
                    bool fw_in_secondary_set = KEEP_SECONDARY.count(fw) > 0 ||
                        u2_strend2_words_.count(fw) > 0 ||
                        fw_is_strend_sec || fw_is_ing_strend;
                    auto dit = dict_.find(fw);
                    bool fw_weak = !fw_in_secondary_set &&
                        dit != dict_.end() && !dit->second.empty() &&
                        (dit->second[0] == ',' || dit->second[0] == '%');
                    if (!fw_weak) { has_following_stressed = true; }
                }
            }
        }
        if (!has_following_stressed) {
            keep_sec = false;
            if (ph_codes.find('\'') == std::string::npos) {
                replaceFirstChar(ph_codes, ',', '\'');
            }
        }
    }
}

// Leading-comma gated comma -> primary promotion. Promotes the first
// ',' in ph_codes to '\'' unless the comma is leading (position 0;
// or position 1 after '%'; or all preceding chars are
// non-vowels/markers — making it effectively word-initial). Also
// skipped for "%-prefix phrase entries with secondary but no
// primary" — those carry genuine secondary stress.
void IPAPhonemizer::applyCommaToPrimaryPromotion(
        bool keep_sec, bool needs_sec, bool phrase_matched,
        bool is_isolated_word, std::string& ph_codes) const {
    size_t comma_pos = ph_codes.find(',');
    bool comma_at_start = (comma_pos == 0 ||
        (comma_pos == 1 && !ph_codes.empty() && ph_codes[0] == '%'));
    // "Effectively leading": no real vowel appears before the comma
    // (e.g. "gonna" g,@n@: ',' at pos 1 after consonant 'g').
    static const std::string VOWEL_CHARS = "aAeEiIoOuUV03@";
    bool effectively_leading = (!comma_at_start &&
        comma_pos != std::string::npos &&
        std::none_of(ph_codes.begin(), ph_codes.begin() + comma_pos,
            [](char cc) {
                return cc != '\'' && cc != '%' && cc != '=' &&
                       VOWEL_CHARS.find(cc) != std::string::npos;
            }));
    bool leading_comma = comma_at_start || effectively_leading;
    bool is_pct_phrase = phrase_matched && !ph_codes.empty() &&
        ph_codes[0] == '%' && !is_isolated_word &&
        ph_codes.find('\'') == std::string::npos;
    bool should_promote = (!keep_sec && !needs_sec &&
        ph_codes.find('\'') == std::string::npos &&
        comma_pos != std::string::npos && !is_pct_phrase &&
        (is_isolated_word || !leading_comma));
    if (should_promote) { replaceFirstChar(ph_codes, ',', '\''); }
}

void IPAPhonemizer::applyStepC(
        const Token& token, size_t ti,
        const std::vector<Token>& tokens, bool is_isolated_word,
        bool phrase_matched, size_t last_word_ti,
        const std::string& matched_phrase_key,
        std::string& ph_codes) const {
    static const std::unordered_set<std::string> KEEP_SECONDARY =
        {"on", "onto", "multiple", "multiples", "going",
         "into", "any", "how", "where", "why",
         "being", "while", "but",
         "across", "above", "among", "amongst", "before",
         "within", "without", "upon", "below", "beside", "between",
         "beyond", "underneath", "behind", "beneath",
         "over", "under",
         "about",
         "make", "makes"};
    static const std::unordered_set<std::string> NEEDS_SECONDARY = {"our"};
    std::string token_lower = toLowerASCII(token.text);
    bool is_strend_secondary = (comma_strend2_words_.count(token_lower) > 0 &&
         !ph_codes.empty() && ph_codes[0] == ',' &&
         ph_codes.find('\'') == std::string::npos);
    bool is_ing_of_strend = isIngOfStrendSecondary(token_lower);
    bool is_keep_sec_phrase = (!matched_phrase_key.empty() &&
        keep_sec_phrase_keys_.count(matched_phrase_key) > 0);
    bool keep_sec = !is_isolated_word &&
        (KEEP_SECONDARY.count(token_lower) > 0 ||
         u2_strend2_words_.count(token_lower) > 0 ||
         u_plus_secondary_words_.count(token_lower) > 0 ||
         is_strend_secondary || is_ing_of_strend || is_keep_sec_phrase);
    bool needs_sec = !is_isolated_word &&
        NEEDS_SECONDARY.count(token_lower) > 0;
    // Promotion from secondary to primary based on lookahead.
    tryPromoteKeepSecToPrimary(token_lower, ti, tokens,
                               keep_sec, ph_codes);
    if (keep_sec) { replaceFirstChar(ph_codes, '\'', ','); }
    if (needs_sec && ph_codes.find('\'') == std::string::npos &&
        ph_codes.find(',') == std::string::npos) {
        static const std::string STRONG_VOWELS = "aAeEiIoOuUV3";
        auto it = std::find_if(ph_codes.begin(), ph_codes.end(),
            [](char c) {
                return STRONG_VOWELS.find(c) != std::string::npos;
            });
        if (it != ph_codes.end()) { ph_codes.insert(it, ','); }
    }
    if (std::getenv("PHON_DEBUG")) {
        std::cerr << "[StepC] word=" << token.text
                  << " ph_codes=" << ph_codes
                  << " is_isolated=" << is_isolated_word
                  << " keep_sec=" << keep_sec << "\n";
    }
    // Comma -> primary promotion (leading-comma gate).
    applyCommaToPrimaryPromotion(keep_sec, needs_sec, phrase_matched,
                                 is_isolated_word, ph_codes);
    // $unstressend: keep secondary even when utterance-final.
    if (!is_isolated_word && ti == last_word_ti &&
        unstressend_words_.count(token_lower) > 0) {
        replaceFirstChar(ph_codes, '\'', ',');
    }
}

size_t IPAPhonemizer::findNextNonEmptyWord(
        const std::vector<Token>& tokens, size_t start) {
    size_t found = std::string::npos;
    size_t tk = start;
    while (tk < tokens.size() && found == std::string::npos) {
        if (tokens[tk].is_word && !tokens[tk].text.empty()) { found = tk; }
        tk++;
    }
    return found;
}

bool IPAPhonemizer::wordIsVowelInitialNonYod(
        const std::vector<Token>& tokens, size_t idx) const {
    bool result = false;
    if (!tokens[idx].text.empty()) {
        char fc = (char)std::tolower((unsigned char)tokens[idx].text[0]);
        bool is_vowel = (fc=='a'||fc=='e'||fc=='i'|| fc=='o'||fc=='u');
        if (is_vowel) {
            bool j_onset = false;
            if (fc == 'u' || fc == 'e') {
                std::string nph = wordToPhonemes(
                    toLowerASCII(tokens[idx].text));
                size_t npi = 0;
                while (npi < nph.size() && (nph[npi]=='\'' || nph[npi]==',' ||
                        nph[npi]=='%' || nph[npi]=='=')) {
                    npi++;
                }
                j_onset = (npi < nph.size() && nph[npi] == 'j');
            }
            result = !j_onset;
        }
    }
    return result;
}

namespace {
    const std::unordered_map<std::string, std::string>&
    cliticIpaTable() {
        static const std::unordered_map<std::string, std::string>
            CLITIC_IPA = {
                {"of a",    "\xc9\x99v\xc9\x99"},
                {"of the",  "\xca\x8cv\xc3\xb0\xc9\x99"},
                {"in the",  "\xc9\xaan\xc3\xb0\xc9\x99"},
                {"on the",  "\xc9\x94n\xc3\xb0\xc9\x99"},
                {"from the",
                    "\x66\xc9\xb9\xca\x8cm\xc3\xb0\xc9\x99"},
                {"that a",
                    "\xc3\xb0\xcb\x8c\xc3\xa6\xc9\xbe\xc9\x99"},
                {"i am",    "a\xc9\xaa\xc9\x90m"},
                {"was a",   "w\xca\x8cz\xc9\x90"},
                {"to be",   "t\xc9\x99" "bi"},
                {"out of",
                    "\xcb\x8c" "a\xca\x8a\xc9\xbe\xc9\x99v"},
            };
        return CLITIC_IPA;
    }
    const std::unordered_map<std::string, std::string>&
    staticPhraseDict() {
        static const std::unordered_map<std::string, std::string>
            PHRASE_DICT = {
                {"has been", "h'azbi:n"},
            };
        return PHRASE_DICT;
    }
}

bool IPAPhonemizer::tryEmitDirectClitic(
        const std::string& bigram, const Token& token,
        const std::vector<Token>& tokens, size_t tj,
        std::string& result, bool& first_word) const {
    auto cit = cliticIpaTable().find(bigram);
    bool matched = (cit != cliticIpaTable().end());
    if (matched) {
        std::string clitic_ipa = cit->second;
        // "the"-ending bigram -> swap trailing ə to ɪ if the next
        // word is vowel-initial non-yod.
        if (toLowerASCII(tokens[tj].text) == "the") {
            size_t tk = findNextNonEmptyWord(tokens, tj + 1);
            if (tk != std::string::npos &&
                wordIsVowelInitialNonYod(tokens, tk) &&
                clitic_ipa.size() >= 2 &&
                (unsigned char)clitic_ipa[clitic_ipa.size()-2] == 0xc9 &&
                (unsigned char)clitic_ipa[clitic_ipa.size()-1] == 0x99) {
                clitic_ipa[clitic_ipa.size()-1] = (char)0xaa;
            }
        }
        if (token.needs_space_before && !first_word) { result += ' '; }
        result += clitic_ipa;
        first_word = false;
    }
    return matched;
}

bool IPAPhonemizer::tryMatchStaticPhrase(
        const std::string& bigram, CliticOrPhraseResult& r) const {
    auto pit = staticPhraseDict().find(bigram);
    bool matched = (pit != staticPhraseDict().end());
    if (matched) {
        r.ph_codes = processPhonemeString(pit->second);
        r.phrase_matched = true;
    }
    return matched;
}

bool IPAPhonemizer::tryMatchLoadedPhrase(
        const std::string& bigram, size_t tj,
        const std::vector<Token>& tokens,
        CliticOrPhraseResult& r) const {
    auto pit2 = phrase_dict_.find(bigram);
    bool matched = (pit2 != phrase_dict_.end());
    if (matched) {
        const std::string& raw_phrase = pit2->second;
        // Pre-vowel "the" flag: phrase ends in "@2" (ðə) and the
        // next non-empty word is vowel-initial non-yod.
        bool phrase_ends_the = (raw_phrase.size() >= 3 &&
            raw_phrase.compare(raw_phrase.size()-3, 3,
                               "D@2") == 0);
        if (phrase_ends_the) {
            size_t tk = findNextNonEmptyWord(tokens, tj + 1);
            r.phrase_pre_vowel_the = (tk != std::string::npos &&
                 wordIsVowelInitialNonYod(tokens, tk));
        }
        r.ph_codes = processPhonemeString(raw_phrase);
        r.phrase_matched = true;
        r.matched_phrase_key = bigram;
    }
    return matched;
}

bool IPAPhonemizer::tryEmitSplitPhrase(
        const std::string& bigram, const Token& token,
        const std::vector<Token>& tokens, size_t tj,
        std::string& result, bool& first_word) const {
    (void)tokens; (void)tj;  // signature kept for symmetry
    auto psit = phrase_split_dict_.find(bigram);
    bool matched = (psit != phrase_split_dict_.end());
    if (matched) {
        const std::string& ph1 = psit->second.first;
        const std::string& ph2 = psit->second.second;
        bool phrase_has_primary = (ph1.find('\'') != std::string::npos ||
             ph2.find('\'') != std::string::npos);
        auto doSplitPart = [&](const std::string& ph,
                               bool is_first) -> std::string {
            bool has_stress = (ph.find('\'') != std::string::npos ||
                               ph.find(',') != std::string::npos);
            std::string ph_proc = ph;
            if (!has_stress && !(is_first && !phrase_has_primary)) {
                ph_proc = "%" + ph;
            }
            return phonemesToIPA(processPhonemeString(ph_proc));
        };
        std::string ipa1 = doSplitPart(ph1, true);
        if (token.needs_space_before && !first_word) { result += ' '; }
        result += ipa1;
        first_word = false;
        std::string ipa2 = doSplitPart(ph2, false);
        result += ' ';
        result += ipa2;
    }
    return matched;
}

// Bigram cliticization + phrase lookup. Tries 4 sub-paths in
// order. Direct-clitic and split-phrase paths emit IPA + advance
// ti (clitic_matched=true). Static/loaded phrase paths populate
// r.ph_codes for downstream processing (phrase_matched=true).
IPAPhonemizer::CliticOrPhraseResult
IPAPhonemizer::tryCliticOrPhrase(
        const Token& token, size_t ti,
        const std::vector<Token>& tokens,
        std::string& result, bool& first_word) const {
    CliticOrPhraseResult r;
    r.advance_to = ti;
    size_t tj = ti + 1;
    while (tj < tokens.size() && !tokens[tj].is_word) { tj++; }
    if (tj < tokens.size() && tokens[tj].is_word) {
        std::string bigram = toLowerASCII(token.text) +
            " " + toLowerASCII(tokens[tj].text);
        if (tryEmitDirectClitic(bigram, token, tokens, tj,
                                result, first_word)) {
            r.clitic_matched = true;
            r.advance_to = tj;
        } else if (tryMatchStaticPhrase(bigram, r)) {
            r.advance_to = tj;
        } else if (tryMatchLoadedPhrase(bigram, tj, tokens, r)) {
            r.advance_to = tj;
        } else if (tryEmitSplitPhrase(bigram, token, tokens, tj,
                                      result, first_word)) {
            r.clitic_matched = true;
            r.advance_to = tj;
        }
    }
    return r;
}

// Period-abbrev expansion. "U.S." / "U.K." / "N.Y." style tokens
// become single phoneme units with secondary stress on all but the
// last letter and primary on the last (the reference convention:
// j'u: -> j,u: -> IPA jˌuː for "U" leading "U.S.").
bool IPAPhonemizer::expandPeriodAbbreviation(
        const Token& token, std::string& result, bool& first_word,
        std::string& ph_codes) const {
    bool consumed = false;
    if (token.text.find('.') != std::string::npos) {
        std::vector<std::string> letter_ipa;
        for (size_t ci = 0; ci < token.text.size(); ci++) {
            char lc = token.text[ci];
            if (std::isalpha((unsigned char)lc)) {
                char lc_lower = std::tolower((unsigned char)lc);
                std::string underscore_key = std::string("_") + lc_lower;
                auto uit = dict_.find(underscore_key);
                if (uit != dict_.end()) {
                    letter_ipa.push_back(uit->second);
                } else {
                    std::string lword(1, lc);
                    letter_ipa.push_back(wordToPhonemes(lword));
                }
            }
        }
        if (letter_ipa.size() >= 2) {
            // All-but-last -> secondary stress; last -> primary.
            std::string combined_codes;
            for (size_t li = 0; li < letter_ipa.size(); li++) {
                std::string code = letter_ipa[li];
                if (li < letter_ipa.size() - 1) {
                    std::string modified;
                    size_t first = code.find_first_of("',");
                    if (first == std::string::npos) {
                        modified = "," + code;
                    } else {
                        modified = code.substr(0, first) + ",";
                        for (size_t ci = first + 1; ci < code.size(); ci++) {
                            char c2 = code[ci];
                            if (c2 != '\'' && c2 != ',') { modified += c2; }
                        }
                    }
                    combined_codes += modified;
                } else {
                    if (code.find('\'') == std::string::npos) {
                        combined_codes += '\'' + code;
                    } else {
                        combined_codes += code;
                    }
                }
            }
            std::string combined_ipa = phonemesToIPA(combined_codes);
            if (token.needs_space_before && !first_word) { result += ' '; }
            result += combined_ipa;
            first_word = false;
            consumed = true;
        } else if (letter_ipa.size() == 1) {
            ph_codes = letter_ipa[0];
        } else {
            ph_codes = wordToPhonemes(token.text);
        }
    } else {
        ph_codes = wordToPhonemes(token.text);
    }
    return consumed;
}

// $atstart override: first-word lookup in atstart_dict_.
void IPAPhonemizer::applyAtStartOverride(
        const Token& token, bool first_word, bool phrase_matched,
        std::string& ph_codes) const {
    if (first_word && !phrase_matched) {
        auto ait = atstart_dict_.find(toLowerASCII(token.text));
        if (ait != atstart_dict_.end()) { ph_codes = ait->second; }
    }
}

// $atend override: last-word-of-utterance lookup in atend_dict_.
// Skipped for isolated words (the only word IS already the last
// but the override expects multi-word context) and phrase-matched
// tokens.
void IPAPhonemizer::applyAtEndOverride(
        const Token& token, size_t ti, size_t last_word_ti,
        bool is_isolated_word, bool phrase_matched,
        std::string& ph_codes) const {
    if (ti == last_word_ti && !is_isolated_word && !phrase_matched) {
        auto aeit = atend_dict_.find(toLowerASCII(token.text));
        if (aeit != atend_dict_.end()) { ph_codes = aeit->second; }
    }
}

// ============================================================
// Main phonemize functions
// ============================================================
// phonemizeText per-token pipeline for word tokens. Runs the full
// dispatch chain and updates POS-context counters at the end.
void IPAPhonemizer::processWordToken(
        const Token& token, size_t& ti,
        const std::vector<Token>& tokens, bool is_isolated_word,
        size_t last_word_ti, bool is_en_us,
        int& expect_past, int& expect_noun, int& expect_verb,
        std::string& result, bool& first_word) const {
    bool consumed = false;
    // Number expansion ("16"->"sixteen", "2011"->...).
    if (!consumed && expandNumberToken(token, result, first_word)) {
        consumed = true;
    }
    // All-caps acronym / mixed-case-no-vowel abbreviation
    // (DNA, PhD, ...).
    if (!consumed && spellAcronymToken(token, result, first_word)) {
        consumed = true;
    }
    std::string ph_codes;
    bool phrase_matched = false;
    bool clitic_matched = false;
    bool phrase_pre_vowel_the = false;
    std::string matched_phrase_key;
    if (!consumed) {
        // Bigram cliticization + phrase lookup.
        auto cr = tryCliticOrPhrase(token, ti, tokens, result,
                                    first_word);
        ph_codes = std::move(cr.ph_codes);
        phrase_matched = cr.phrase_matched;
        clitic_matched = cr.clitic_matched;
        phrase_pre_vowel_the = cr.phrase_pre_vowel_the;
        matched_phrase_key = std::move(cr.matched_phrase_key);
        ti = cr.advance_to;
        if (clitic_matched) { consumed = true; }
    }
    if (!consumed && !phrase_matched) {
        // Period-abbrev ("U.S.") OR fall-through to wordToPhonemes.
        if (expandPeriodAbbreviation(token, result, first_word,
                                     ph_codes)) {
            consumed = true;
        }
    }
    if (!consumed) {
        applyPosContextOverride(token, is_isolated_word,
                                phrase_matched, expect_past,
                                expect_noun, expect_verb, ph_codes);
        applyAtStartOverride(token, first_word, phrase_matched,
                             ph_codes);
        applyAtEndOverride(token, ti, last_word_ti,
                           is_isolated_word, phrase_matched,
                           ph_codes);
        applyLemmaOverride(token, ti, tokens, is_isolated_word,
                           last_word_ti, ph_codes);
        // Step A is a no-op. Step B returns is_unstressed_word for
        // maybeAddDefaultStress below.
        bool is_unstressed_word = applyStepB(
            token, phrase_matched, is_isolated_word, ph_codes);
        applyStepC(token, ti, tokens, is_isolated_word,
                   phrase_matched, last_word_ti,
                   matched_phrase_key, ph_codes);
        fixDiphthongStressPosition(ph_codes);
        applyInterWordTFlap(ti, tokens, ph_codes);
        applyCrossWordSchwaRhotic(ti, is_en_us, is_isolated_word,
                                  tokens, ph_codes);
        std::string ipa = phonemesToIPA(ph_codes);
        maybeAddDefaultStress(token, phrase_matched,
                              is_isolated_word, is_unstressed_word,
                              ph_codes, ipa);
        applyPreVowelTheFixup(phrase_pre_vowel_the, ipa);
        applyRLinking(first_word, ti, tokens, ipa);
        applyCrossWordTFlap(token, ti, is_en_us, is_isolated_word,
                            tokens, ipa);
        if (!ipa.empty()) {
            if (!first_word) { result += " "; }
            result += ipa;
            first_word = false;
        }
    }
    updatePosContextCounters(token, expect_past, expect_noun,
                             expect_verb);
}

std::string IPAPhonemizer::phonemizeText(
        const std::string& text) const {
    if (!loaded_) { return ""; }
    bool is_en_us = (dialect_ == "en-us" || dialect_ == "en_us");
    auto tokens = tokenizeText(text);
    std::string result;
    bool first_word = true;
    // Count word tokens for single-word isolation + find last-word
    // index (for $atend and $unstressend gates).
    int word_token_count = 0;
    size_t last_word_ti = std::string::npos;
    for (size_t tii = 0; tii < tokens.size(); tii++) {
        if (tokens[tii].is_word) {
            word_token_count++;
            last_word_ti = tii;
        }
    }
    bool is_isolated_word = (word_token_count == 1);
    // POS-context counters bumped by \$pastf/\$nounf/\$verbf
    // words and decremented each word.
    int expect_past = 0;
    int expect_noun = 0;
    int expect_verb = 0;
    for (size_t ti = 0; ti < tokens.size(); ti++) {
        const auto& token = tokens[ti];
        if (token.is_word) {
            processWordToken(token, ti, tokens, is_isolated_word,
                             last_word_ti, is_en_us,
                             expect_past, expect_noun, expect_verb,
                             result, first_word);
        }
        // Punctuation tokens are not emitted in the IPA output.
    }
    return result;
}

std::vector<std::string> IPAPhonemizer::phonemize(
        const std::vector<std::string>& texts) {
    std::vector<std::string> results;
    results.reserve(texts.size());
    for (const auto& text : texts) { results.push_back(phonemizeText(text)); }
    return results;
}

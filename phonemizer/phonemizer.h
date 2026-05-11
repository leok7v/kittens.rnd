// Copyright 2024 - Apache 2.0 License
// Main IPA phonemizer interface
// Reads rule and dictionary files to phonemize English text to IPA.

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "rule_parser.h"
#include "ipa_table.h"

// ============================================================
// Phoneme token - represents one phoneme code from the rules
// ============================================================
struct PhToken {
    enum Type { STRESS_PRIMARY, STRESS_SECONDARY, PHONEME, PAUSE, SYLLABLE };
    Type type;
    std::string code;   // phoneme code (e.g., "eI", "@", "t")
    bool is_vowel;

    PhToken(Type t, const std::string& c, bool v = false)
        : type(t), code(c), is_vowel(v) {}
};

// ============================================================
// Main phonemizer class
// ============================================================
class IPAPhonemizer {
public:
    // dialect: "en-us" (American) or "en-gb" (British)
    // rules_path: path to en_rules file
    // list_path: path to en_list file
    explicit IPAPhonemizer(const std::string& rules_path,
                               const std::string& list_path,
                               const std::string& dialect = "en-us");

    // Phonemize a list of texts; returns IPA strings (one per input)
    // Matches the interface: phonemizer.phonemize([text])
    std::vector<std::string> phonemize(const std::vector<std::string>& texts);

    // Phonemize a single text string
    std::string phonemizeText(const std::string& text) const;

    bool isLoaded() const { return loaded_; }
    const std::string& getError() const { return error_; }

private:
    std::string dialect_;
    bool loaded_;
    std::string error_;

    // Dictionary: word -> raw phoneme code string
    std::unordered_map<std::string, std::string> dict_;

    // Verb-form dictionary: word -> phoneme string for the verb pronunciation ($verb entries).
    // Used for -ing/-ed suffix stripping where the verb form is needed (e.g. "live"->"lIv"
    // not "laIv", so "living"->"lˈɪvɪŋ" not "lˈaɪvɪŋ").
    std::unordered_map<std::string, std::string> verb_dict_;

    // Past-tense pronunciation: loaded from $past entries in en_list.
    // Only used when expect_past > 0 (i.e., after a $pastf word like "was", "were", "had").
    // e.g., "read $past" -> rEd = ɹˈɛd (past) vs ri:d = ɹˈiːd (present/default).
    std::unordered_map<std::string, std::string> past_dict_;

    // Noun-context pronunciation: loaded from $noun entries in en_list (not stored in dict_).
    // Only used when expect_noun > 0 (i.e., after a $nounf word like "a", "my", "the", etc.).
    // e.g., "elaborate $noun" -> I#lab3@t = ᵻlˈæbɚɹˌɪt; without noun context rules fire instead.
    std::unordered_map<std::string, std::string> noun_dict_;

    // Words with $pastf flag: trigger expect_past for the following word(s).
    // e.g., "was", "were", "is", "are", "been", "had", "have" (as auxiliaries).
    std::unordered_set<std::string> pastf_words_;

    // Words with $nounf flag: trigger expect_noun for the following word(s).
    // e.g., "a", "every", "my", "his", "her", "its", "our", "your", "their", "some", etc.
    std::unordered_set<std::string> nounf_words_;

    // Words with $verbf flag: trigger expect_verb for the following word(s).
    // e.g., "I", "we", "you", "they", "will", "would", "shall", "should", "to", etc.
    std::unordered_set<std::string> verbf_words_;

    // Rule set
    RuleSet ruleset_;

    // IPA override table
    std::unordered_map<std::string, std::string> ipa_overrides_;

    // Words that carry the $u (typically unstressed) flag in the dictionary
    std::unordered_set<std::string> unstressed_words_;

    // Words with $unstressend flag: stay at secondary stress even when utterance-final.
    // e.g. "ones w02nz $only $unstressend" — keeps ˌ at sentence end, not promoted to ˈ.
    std::unordered_set<std::string> unstressend_words_;

    // Words with $abbrev flag: always read as individual letter names when all-caps
    std::unordered_set<std::string> abbrev_words_;

    // Stressed syllable position from $N dict flags (1-based; 0 = not set)
    std::unordered_map<std::string, int> stress_pos_;

    // Word-level alt flags from $altN dict entries (bitmask: bit N-1 set when word has $altN)
    std::unordered_map<std::string, int> word_alt_flags_;

    // Dictionary for $atstart entries: used only when word is the first in the utterance.
    std::unordered_map<std::string, std::string> atstart_dict_;

    // Dictionary for $atend entries: used only when word is the last in the utterance.
    // e.g. "to tu: $u $atend" -> "to" at sentence end gets full form tuː (not reduced tə).
    std::unordered_map<std::string, std::string> atend_dict_;

    // Dictionary for $capital entries: used only when word starts with a capital letter.
    // e.g. "Bologna b@loUn;@ $capital" -> used for "Bologna" (city) not "bologna" (sausage).
    std::unordered_map<std::string, std::string> capital_dict_;

    // Words with $onlys flag: dict entry is only valid for the bare form or with 's' suffix.
    // When stripping non-s suffixes (e.g. -ed, -ing, -able), skip these dict entries.
    std::unordered_set<std::string> onlys_words_;

    // Bare-word override from $onlys entries that coexist with a plain (no-flag) entry.
    // e.g. "desert dEz3t $onlys" overrides "desert dI#z3:t" for bare-word lookup.
    // Suffix stripping still uses dict_ (the plain entry) for stems.
    std::unordered_map<std::string, std::string> onlys_bare_dict_;

    // Words with $only flag: dict entry is only valid for the isolated bare word form.
    // Should NOT be used as a stem for any suffix stripping (even -s).
    // E.g. "guid" has $only -> should not suppress magic-e when processing "guiding"/"guided".
    std::unordered_set<std::string> only_words_;

    // Words where stress_pos_ came from a flag-only $N $onlys entry (noun-form-only stress).
    // These stress overrides should NOT be applied when phonemizing verb-derived stems
    // (e.g. "construct $1 $onlys" -> noun has 1st-syll stress, but "constructing" uses verb rules).
    std::unordered_set<std::string> noun_form_stress_;

    // Words with a $verb flag-only en_list entry (no phoneme, just marks verb-context override).
    // These words have a separate verb pronunciation governed by rules, NOT by stress_pos_.
    // E.g. "conduct $verb" -> verb form uses rules (2nd-syllable stress), not $1 noun-form stress.
    std::unordered_set<std::string> verb_flag_words_;

    // Compound prefix words: entries with $strend2 and bare (unstressed) phoneme.
    // Sorted by word length descending for longest-match-first.
    // e.g., "under" -> "Vnd3", "over" -> "oUv3", "through" -> "Tru:"
    std::vector<std::pair<std::string, std::string>> compound_prefixes_;

    // Words with $strend2 and bare phoneme (no leading stress marker) that need
    // final-syllable (pick_last) stress placement in processPhonemeString step 5.
    // Mirrors compound_prefixes_ but as a set for O(1) lookup.
    // e.g., "become" (bIkVm), "within" (wIDIn), "without" (wIDaUt).
    std::unordered_set<std::string> strend_words_;

    // Words with BOTH $u2 AND $strend2 flags: function words that should carry
    // secondary stress (not primary) in sentence context (e.g. "together", "across").
    std::unordered_set<std::string> u2_strend2_words_;

    // Words with $strend2 whose dict phoneme starts with ',' (secondary-stressed).
    // These are NOT in strend_words_ (which requires bare/unstressed phoneme).
    // Like KEEP_SECONDARY, they stay secondary when followed by stressed content,
    // but get promoted to primary when phrase-final. E.g., "go" (,goU), "so" (,soU),
    // "up" (,Vp), "down" (,daUn), "doing" (,du:IN), "should" (,SUd), "might" (,maIt).
    std::unordered_set<std::string> comma_strend2_words_;

    // Words with $u+ flag whose dict phoneme has ',' (secondary) but no '\'' (primary).
    // These should keep secondary stress in sentence context (not promoted to primary).
    // e.g. "made" (m,eId $u+): keeps ˌ in sentence context.
    std::unordered_set<std::string> u_plus_secondary_words_;

    // Phrase dictionary: "word1 word2" -> phoneme string for multi-word phrases.
    // Loaded from parenthesized entries in en_list (e.g. "(for the) f3D@2 $nounf").
    // Used for bigram cliticization in sentence context.
    std::unordered_map<std::string, std::string> phrase_dict_;

    // Split-phrase dictionary: "word1 word2" -> (phoneme1, phoneme2) for phrases where
    // each word has its own phoneme separated by || in the en_list entry.
    // E.g., "(most of) moUst||@v" -> phrase_split_dict_["most of"] = {"moUst", "@v"}.
    // (too much) t'u:||mVtS -> "too much" -> {"t'u:", "mVtS"}.
    std::unordered_map<std::string, std::pair<std::string,std::string>> phrase_split_dict_;

    // Phrase keys that should behave like KEEP_SECONDARY: secondary in sentence context
    // (when followed by stressed content), primary when phrase-final.
    // Loaded from phrase entries with $u2+ flag (e.g. "do not", "did not", "does not").
    std::unordered_set<std::string> keep_sec_phrase_keys_;

    // Load the word list (en_list)
    bool loadDictionary(const std::string& path);

    // Load the rules file (en_rules)
    bool loadRules(const std::string& path);

    // Normalize text for processing
    std::string normalizeText(const std::string& text) const;

    // Split text into words/tokens preserving punctuation
    struct Token {
        std::string text;
        bool is_word;
        bool needs_space_before;
    };
    std::vector<Token> tokenizeText(const std::string& text) const;

    // Get phoneme codes for a single word
    std::string wordToPhonemes(const std::string& word) const;

    // wordToPhonemes is decomposed into a chain of mutually-exclusive
    // dispatch helpers; each returns "" if it cannot claim the word,
    // letting wordToPhonemes try the next. Order matches the original
    // monolithic function: capital_dict_ -> dict_/onlys_bare_dict_ ->
    // hyphenated -> possessive -> single-letter -> morphological-suffixes
    // (-ing/-ed/-ies/-s/-[Ce]s/-[ch/sh]es/-xes/-arily) ->
    // compound-prefixes -> applyRules fallback.
    std::string checkCapitalDict(const std::string& word,
                                 const std::string& norm) const;
    std::string checkMainDict(const std::string& norm) const;
    std::string checkHyphenated(const std::string& norm) const;
    std::string checkPossessive(const std::string& norm) const;
    std::string checkSingleLetter(const std::string& norm) const;
    std::string getStemPhonemes(const std::string& stem) const;
    std::string checkSuffixIng(const std::string& norm) const;
    std::string checkSuffixEd(const std::string& norm) const;
    std::string checkMorphologicalSuffixes(
        const std::string& norm) const;
    std::string checkCompoundPrefixes(const std::string& norm) const;
    std::string applyRulesFallback(const std::string& norm) const;

    // Apply rules to a word (for unknown words)
    // word_alt_flags: bitmask of $altN flags active for this word (-1 = look up from word_alt_flags_)
    // suffix_phoneme_only: when true, RULE_ENDING rules contribute their phoneme normally
    // (no stem re-phonemization). Mimics TranslateRules(word, NULL) behavior where
    // RULE_ENDING fires but doesn't trigger early return — the suffix phoneme is appended
    // to the accumulated first-pass phonemes rather than re-phonemizing the extracted stem.
    // Used when re-translating stems that contain further suffix rules (e.g. "ribosome"
    // contains "-some" RULE_ENDING; with suffix_phoneme_only=true, "ribosome" gives first-pass
    // phonemes rIb0soUm rather than re-phonemized ri:boUsoUm).
    std::string applyRules(const std::string& word, bool allow_suffix_strip = true,
                           int word_alt_flags = -1,
                           bool suffix_phoneme_only = false,
                           bool suffix_removed = false,
                           std::vector<bool>* out_replaced_e = nullptr,
                           std::vector<bool>* out_pos_visited = nullptr) const;

    // Best rule match at one scan position — returned by findBestRule.
    // (Decomposed out of applyRules to make the main scan loop short.)
    struct RuleMatchResult {
        int  score;
        std::string phonemes;
        int  advance;
        int  del_start;
        int  del_count;
        bool is_prefix;
        bool is_suffix;
        int  suffix_strip_len;
        int  suffix_flags;
    };

    // Resolve word_alt_flags: explicit param if >= 0, else look up the
    // word's own $altN bitmask from word_alt_flags_.
    int determineAltFlags(const std::string& word,
                          int word_alt_flags_param) const;

    // Try the 2-char group (with +35 bonus) and the 1-char group at
    // `pos`; return the best matching rule's score + side info.
    RuleMatchResult findBestRule(const std::string& word, int pos, int len,
                                 char pos_char,
                                 int word_alt_flags,
                                 const std::vector<bool>& replaced_e,
                                 bool allow_suffix_strip,
                                 bool suffix_phoneme_only,
                                 bool suffix_removed,
                                 const std::string& accumulated_phonemes) const;

    // Handle the phonSTRESS_PREV marker '=' at the start of a rule's
    // emit string: retroactively promote the last preceding stressable
    // vowel in `phonemes` to PRIMARY ('). Mutates both `emit` (consumes
    // the leading '=') and `phonemes` (inserts '\'' / demotes earlier
    // primaries).
    void applyStressPrev(std::string& emit,
                         std::string& phonemes) const;

    // RULE_ENDING SUFFIX terminal path: strip N chars, re-phonemize stem
    // (dict / verb_dict / rules), apply SUFX_I/E flags and the 'd#'
    // devoicing rule, return stem_ph + suffix_ph.
    std::string processSuffixRule(const std::string& word, int len,
                                  int word_alt_flags,
                                  const RuleMatchResult& match,
                                  bool allow_suffix_strip) const;

    // PREFIX rule retranslation: phonemize the suffix portion as a new
    // word, demote prefix/suffix primary stress per the compound rules,
    // and return prefix_ph + suffix_ph. Returns "" if the prefix has a
    // full stressable vowel and no stress markers — caller falls back
    // to in-context scanning of the rest of the word.
    std::string processPrefixRule(const std::string& word, int pos,
                                  const RuleMatchResult& match,
                                  std::string& phonemes) const;

    // applyStressPrev sub-helpers (decomposed for clarity).
    int findLastStressableVowel(const std::string& phonemes) const;

    // processSuffixRule sub-helpers.
    bool stemPhonemeFromDict(const std::string& stem,
                             const RuleMatchResult& match,
                             int word_alt_flags,
                             std::string& stem_ph) const;
    void appendMagicEIfNeeded(std::string& stem,
                              const std::string& suffix_ph,
                              int suffix_flags) const;
    void devoiceEdSuffix(const std::string& stem_ph,
                         std::string& suffix_ph) const;

    // processPrefixRule sub-helpers.
    bool prefixHasFullVowel(const std::string& phonemes) const;
    int  countPrefixVowels(const std::string& phonemes) const;
    bool prefixEndsInSchwa(const std::string& phonemes) const;
    int  countSuffixSyllables(const std::string& suffix) const;

    // Apply replace rules (preprocessing)
    std::string applyReplacements(const std::string& word) const;

    // Match a single rule at position pos in word
    // group_length: 1 for single-char groups, 2 for two-char groups
    // word_alt_flags: bitmask of $altN flags for $w_altN right-context matching
    int matchRule(const PhonemeRule& rule, const std::string& word, int pos,
                   std::string& out_phonemes, int& advance, int& del_fwd_start, int& del_fwd_count,
                   int group_length = 1,
                   const std::string& phonemes_so_far = "",
                   int word_alt_flags = 0,
                   const std::vector<bool>* replaced_e_arr = nullptr,
                   bool suffix_removed = false) const;

    // Match left context (scan backward from pos-1)
    bool matchLeftContext(const std::string& ctx_str, const std::string& word, int pos) const;

    // Match right context (scan forward from pos+match_len)
    bool matchRightContext(const std::string& ctx_str, const std::string& word, int pos,
                           int& del_fwd_count) const;

    // Check a single context character against word position
    bool matchContextChar(char ctx_char, const std::string& word, int word_pos,
                          bool at_word_start, bool at_word_end) const;

    // Parse phoneme code string into IPA
    std::string phonemesToIPA(const std::string& phoneme_str) const;

    // Convert a single phoneme code to IPA
    std::string singleCodeToIPA(const std::string& code) const;

    // Check if a phoneme code is a vowel
    bool isVowelCode(const std::string& code) const;

    // Post-process phoneme string for dialect
    std::string postProcessPhonemes(const std::string& phonemes) const;

    // Apply stress and clean up the phoneme string.
    // force_final_stress: when true, place primary on LAST stressable vowel (for $strend2 words).
    std::string processPhonemeString(const std::string& raw, bool force_final_stress = false) const;

    // Apply $N stress position override: force primary on Nth vowel (1-based)
    std::string applyStressPosition(const std::string& raw, int n) const;

    // phonemizeText sub-helpers. Each returns `true` when it claims the
    // token and has already appended IPA to `result` (caller skips the
    // remaining handlers for that token).
    bool expandNumberToken(const Token& token, std::string& result,
                           bool& first_word) const;
    bool spellAcronymToken(const Token& token, std::string& result,
                           bool& first_word) const;
    // POS context override: when the previous word(s) set expect_past
    // / expect_noun / expect_verb counters, look up token in the
    // matching side dict (past_dict_/noun_dict_/verb_dict_) and
    // replace ph_codes if found. No-op when the token is isolated or
    // a phrase already matched.
    void applyPosContextOverride(const Token& token,
                                 bool is_isolated_word,
                                 bool phrase_matched,
                                 int expect_past, int expect_noun,
                                 int expect_verb,
                                 std::string& ph_codes) const;
    // Hand-coded function-word allophone overrides: the/a/an/to/use.
    // Each branch rewrites ph_codes based on next-token onset or
    // previous-token POS. Bundled because they share the next-vowel-
    // initial and /j/-onset lookahead logic.
    void applyLemmaOverride(const Token& token, size_t ti,
                            const std::vector<Token>& tokens,
                            bool is_isolated_word,
                            size_t last_word_ti,
                            std::string& ph_codes) const;
    // Inter-word t-flap: "t#" tail (flappable-t marker on function
    // words like "at", "it") becomes "*" (flap) when the next word
    // starts with a vowel phoneme.
    void applyInterWordTFlap(size_t ti,
                             const std::vector<Token>& tokens,
                             std::string& ph_codes) const;
    // Cross-word @->3 rhotacization: trailing standalone "@" (schwa)
    // promotes to "3" (ɚ) before an r-initial next word. US dialect
    // only; gated off for isolated words. Diphthong-final "@" (e.g.
    // "i@" -> ɪə) is skipped.
    void applyCrossWordSchwaRhotic(size_t ti, bool is_en_us,
                                   bool is_isolated_word,
                                   const std::vector<Token>& tokens,
                                   std::string& ph_codes) const;
    // Cross-word /t/ flapping for the pronoun "it": trailing 't' in
    // the IPA flaps to ɾ before a vowel-initial next word. US dialect
    // only, sentence context only. Other 't'-final function words
    // (that, but, not) do NOT flap cross-word per the reference.
    void applyCrossWordTFlap(const Token& token, size_t ti,
                             bool is_en_us, bool is_isolated_word,
                             const std::vector<Token>& tokens,
                             std::string& ipa) const;
    // Decide whether to add a default primary-stress marker to the
    // IPA. Skipped for inherently-unstressed words: %-prefixed,
    // weak-schwa-only (@2/@5), $u-flagged in sentence context, and
    // the article a/an (a#/a#n).
    void maybeAddDefaultStress(const Token& token, bool phrase_matched,
                               bool is_isolated_word,
                               bool is_unstressed_word,
                               const std::string& ph_codes,
                               std::string& ipa) const;
    // Pre-vowel "the" allophone after phrase_dict_ produced a phrase
    // ending in "@2" (ðə): swap final ə (0xC9 0x99) -> ɪ (0xC9 0xAA).
    void applyPreVowelTheFixup(bool phrase_pre_vowel_the,
                               std::string& ipa) const;
    // Bump the expect_past/noun/verb counters when current token
    // carries a $pastf/$nounf/$verbf flag, then decrement all
    // counters by one (the reference set-then-decrement in same
    // step). Affects the NEXT word's POS-context lookups.
    void updatePosContextCounters(const Token& token,
                                  int& expect_past,
                                  int& expect_noun,
                                  int& expect_verb) const;
    // R-linking sandhi: append a linking ɹ when current IPA ends in
    // ɚ or ɜː and the next word starts with a vowel phoneme.
    // Suppressed before non-final conjunctions ("or"/"and") — those
    // are prosodic phrase boundaries. Period-abbrevs ("U.S.") and
    // unknown all-caps acronyms ("DNA") use their first letter's
    // phoneme to determine vowel-onset (R->A@ counts as vowel,
    // U->j as consonant).
    void applyRLinking(bool first_word, size_t ti,
                       const std::vector<Token>& tokens,
                       std::string& ipa) const;
    // applyRLinking sub-helpers.
    // conjunctionSuppressesLinking returns true iff the word at tj
    // is "or"/"and" (non-first-word) and at least one more word
    // follows before the next punctuation — the prosodic-boundary
    // suppression case.
    bool conjunctionSuppressesLinking(
            const std::vector<Token>& tokens, size_t tj,
            bool first_word) const;
    // nextWordIsVowelInitial returns true iff the phonemized first
    // letter (for period-abbrev / unknown all-caps) or full word
    // begins with a vowel code (after stripping stress markers).
    bool nextWordIsVowelInitial(
            const std::vector<Token>& tokens, size_t tj) const;
    // Skip non-word tokens at indices >= start until the first
    // non-empty word token is found. Stops (returns npos) on the
    // first non-word (punctuation) token.
    static size_t findNextWordStopOnPunct(
            const std::vector<Token>& tokens, size_t start);
    // Find the first non-empty word token at index >= start, NOT
    // stopping on punctuation (skipping past it). Returns npos if
    // none found.
    static size_t findNextNonEmptyWord(
            const std::vector<Token>& tokens, size_t start);
    // True iff the word token at `idx` begins with a vowel letter
    // AND is not a /j/-onset word (e.g. "university" / "Europe" /
    // "unique" -> jˌuː/jˈʊɹ/jˈuː count as consonant-initial for
    // sandhi). Reaches into wordToPhonemes for the /j/ check when
    // the first letter is u or e.
    bool wordIsVowelInitialNonYod(
            const std::vector<Token>& tokens, size_t idx) const;
    // Step D: shift the primary-stress marker '\'' inserted between
    // the two letters of a diphthong digraph (eI/aI/aU/OI/oU) so the
    // marker precedes the full digraph. e.g. "ke'It" -> "k'eIt".
    void fixDiphthongStressPosition(std::string& ph_codes) const;
    // Step B: $u-flagged or %-prefixed function-word stress
    // suppression. Strips primary stress marker '\'' from the
    // phoneme code when:
    //   (a) the word is in unstressed_words_ (or its base before
    //       apostrophe is) AND no phrase matched; OR
    //   (b) the dict phoneme starts with '%' and the only primary
    //       stress was inserted before 'a#' (the step-5
    //       "last-resort" placement).
    // Skipped for STEP_B_KEEP_SECONDARY_WORDS — those need step C
    // to demote primary to secondary, not strip outright.
    // Returns is_unstressed_word for use by the post-IPA stress
    // decision.
    bool applyStepB(const Token& token, bool phrase_matched,
                    bool is_isolated_word,
                    std::string& ph_codes) const;
    // Step C: stress assignment for function words in sentence
    // context. Handles KEEP_SECONDARY (stays ˌ before stressed
    // content, promoted to ˈ when phrase-final), NEEDS_SECONDARY
    // (\$u words that need ˌ added), $strend2 secondary-prefixed
    // words, -ing forms of $strend2 stems, $u2+ phrase entries,
    // and the leading-comma promotion gate. Final phase: $unstressend
    // at-end demotion. Pure mutation of ph_codes.
    void applyStepC(const Token& token, size_t ti,
                    const std::vector<Token>& tokens,
                    bool is_isolated_word, bool phrase_matched,
                    size_t last_word_ti,
                    const std::string& matched_phrase_key,
                    std::string& ph_codes) const;
    // Result of the clitic / phrase / split-phrase lookup. Caller
    // uses these to drive downstream processing (and to decide
    // whether to `continue` past the rest of the per-token pipeline
    // when clitic_matched is true).
    struct CliticOrPhraseResult {
        std::string ph_codes;
        bool phrase_matched = false;
        bool clitic_matched = false;
        bool phrase_pre_vowel_the = false;
        std::string matched_phrase_key;
        size_t advance_to = 0;  // updated ti, equals input ti if no
                                // advance happened
    };
    // tryCliticOrPhrase sub-paths. Each tries one specific lookup
    // and returns true iff a match was found. clitic_matched paths
    // (CLITIC_IPA, phrase_split_dict_) emit IPA directly and the
    // caller `continue`s the for-loop; phrase_matched paths
    // (PHRASE_DICT, phrase_dict_) populate r.ph_codes for the
    // downstream pipeline.
    bool tryEmitDirectClitic(
            const std::string& bigram, const Token& token,
            const std::vector<Token>& tokens, size_t tj,
            std::string& result, bool& first_word) const;
    bool tryMatchStaticPhrase(const std::string& bigram,
                              CliticOrPhraseResult& r) const;
    bool tryMatchLoadedPhrase(const std::string& bigram, size_t tj,
                              const std::vector<Token>& tokens,
                              CliticOrPhraseResult& r) const;
    bool tryEmitSplitPhrase(
            const std::string& bigram, const Token& token,
            const std::vector<Token>& tokens, size_t tj,
            std::string& result, bool& first_word) const;
    // Bigram cliticization + phrase lookup. Tries (in order) the
    // direct-IPA CLITIC_IPA table, the static PHRASE_DICT, the
    // loaded phrase_dict_, and phrase_split_dict_. The first three
    // either emit IPA directly (and advance ti) or set ph_codes for
    // downstream processing. Split-phrases emit both IPA segments
    // immediately. Caller is responsible for the period-abbreviation
    // fallback when neither path matches.
    CliticOrPhraseResult tryCliticOrPhrase(
            const Token& token, size_t ti,
            const std::vector<Token>& tokens,
            std::string& result, bool& first_word) const;
    // Period-abbrev expansion ("U.S.", "U.K.", "N.Y." -> spelled
    // letter-names with secondary+primary stress). Triggered when
    // the token contains '.'. Three sub-cases:
    //   - 2+ alpha letters -> emit IPA directly + return true
    //     (caller `continue`s).
    //   - 1 alpha letter -> set ph_codes from that letter, return
    //     false (caller falls through).
    //   - 0 alpha letters -> set ph_codes via wordToPhonemes,
    //     return false.
    bool expandPeriodAbbreviation(const Token& token,
                                  std::string& result,
                                  bool& first_word,
                                  std::string& ph_codes) const;
    // $atstart override: replace ph_codes with atstart_dict_ entry
    // for the first word token (e.g. "what" -> ",w02t" / ˌwʌt).
    // No-op when phrase_matched.
    void applyAtStartOverride(const Token& token, bool first_word,
                              bool phrase_matched,
                              std::string& ph_codes) const;
    // $atend override: replace ph_codes with atend_dict_ entry for
    // the last word token in a multi-word utterance (e.g. "to" at
    // end -> "tu:" / tuː). Uses raw phoneme string (no
    // processPhonemeString) — relies on \$u flag to suppress stress.
    void applyAtEndOverride(const Token& token, size_t ti,
                            size_t last_word_ti,
                            bool is_isolated_word,
                            bool phrase_matched,
                            std::string& ph_codes) const;
    // checkSuffixIng sub-helper: when the stem phonemes end in
    // "@L" (syllabic L) and the base orthography ends in "Xl"
    // where X is a vowel or non-t consonant, the syllabic context
    // dissolves once -ing is appended. Vowel+l -> "@l" (schwa
    // remains); other consonant+l -> "l" (schwa dropped). Skipped
    // when base ends in "tl" (e.g. "bottling" keeps @LI2N) or
    // "ngl" (e.g. "tingling" -> @-lI2N handled elsewhere).
    void simplifySyllabicLForIng(const std::string& base,
                                 std::string& sph) const;
    // Detect CVC / CVRC / -nc magic-e candidate patterns on the
    // orthographic base of a stripped -ing / -ed word. Outputs
    // `cvc` (base is magic-e candidate) and `nc` (base ends -nc).
    // `include_cvrc` toggles the vowel+r+g/c extension — true for
    // -ing (handles "charging"), false for -ed (legacy parity).
    // Exceptions encoded: -en/-an/-in/-on/-un endings that
    // phonemize to syllabic/schwa-n, -el, -w, -er suffixes (cvc
    // -> false).
    void detectStemPatternForSuffix(const std::string& base,
                                    bool include_cvrc,
                                    bool& cvc, bool& nc) const;
    // Magic-e applicability for CVC stems before -ing / -ed. Given
    // the base phonemes, returns true iff magic-e should be tried.
    // `weak_vowels` is the single-char code set treated as "weak":
    // "I@" for -ing, "I@3" for -ed (the rhotic schwa is already a
    // complete vowel, no magic-e needed).
    bool shouldUseMagicEForCvcStem(
            const std::string& base_ph,
            const std::string& weak_vowels) const;
    // Detect whether `w` starts with a PREFIX rule from ruleset_.
    // Used by checkSuffixEd to recognize compound words ("infrare"
    // = "infra"+"re") that should NOT be treated as -ed verb stems.
    // Reference: this guard is load-bearing — see the "Sister bug
    // pattern: hasPrefixAtStart fall-through" memo.
    bool hasPrefixAtStart(const std::string& w) const;
    // Determine the -ed suffix's phonemic form and concatenate with
    // the stem. Basic rule: t/d-final stem -> "I#d" (ɪd), unvoiced
    // final -> "t", else -> "d". Override: when stem-final voicing
    // gives "t" but the full-word rules give "d", trust the
    // full-word output (handles silent-e + SUFFIX rule interactions).
    // `base` is passed in only for debug logging.
    std::string computeEdSuffixVoicing(
            const std::string& sph,
            const std::string& norm,
            const std::string& base) const;
    // Pre-flight check for -ed suffix processing. Returns true when
    // `norm` is a viable -ed candidate. Filters out 7 skip cases:
    //   1. base ends in 'e' (handled by PREFIX rules).
    //   2. base ends in 'u' (handled by SUFFIX rule).
    //   3. -ced/-ged not in dict (SUFX_E rules handle).
    //   4. -nged not in dict.
    //   5. -eted not in dict.
    //   6. -mented after consonant without stress override.
    //   7. base has 3+ trailing consonants (unlikely verb stem).
    bool isEdSuffixCandidate(const std::string& norm,
                             const std::string& base) const;
    // -ies suffix handler: strip "ies", restore "y", phonemize the
    // stem, append voiced 'z'. Falls back to direct rules when the
    // stem's first vowel disagrees with the full-word rules' first
    // vowel (indicates a specific en_rules rule fired on the full
    // word). Returns "" when norm doesn't match -ies pattern or no
    // stem can be derived.
    std::string checkSuffixIes(const std::string& norm) const;
    // -s / -es dictionary-stem suffix handler. Strips one 's' (or
    // "es") and looks up the stem in onlys_bare_dict_ / verb_dict_
    // / dict_ / stress_pos_ in priority order. Voicing of the
    // suffix: sibilant-final -> "I#z", unvoiced-final -> "s",
    // otherwise -> "z". Skipped when stem ends in 'u' (-ious / -ous
    // adjectives). Returns "" when no match.
    std::string checkSuffixDictS(const std::string& norm) const;
    // -[Ce]s magic-e suffix handler: words ending in consonant+e+s
    // where the magic-e form (consonant+e) is a stem. Examples:
    // "gives" -> "give"+z, "makes" -> "make"+s. Skipped when the
    // consonant is a sibilant (s/z/x/c) or digraph sibilant (ch/sh)
    // — those are handled by the dict-s and ches/shes paths.
    std::string checkSuffixMagicEs(const std::string& norm) const;
    // -ches / -shes suffix: digraph sibilant + es -> /ɪz/.
    // "teaches" -> "teach"+I#z. Stem is norm minus "es".
    std::string checkSuffixChShEs(const std::string& norm) const;
    // -xes suffix: 'x' = /ks/ (cluster); -xes always -> /ksɪz/.
    // "boxes" -> "box"+I#z.
    std::string checkSuffixXes(const std::string& norm) const;
    // -arily suffix: "primarily", "ordinarily", "necessarily".
    // Strategy: phonemize "stem+ari" so open-syllable rules fire
    // on the stem, strip the trailing -ari phoneme, append 'e@rI#l%i.
    std::string checkSuffixArily(const std::string& norm) const;
    // -ed stem fallback chain when the CVC/nc magic-e branch is
    // unavailable. Tries (in order): doubled-consonant strip (when
    // prefix has 2+ vowel groups), -rs/-ns magic-e, plain stem,
    // doubled-consonant strip (other direction), magic-e add when
    // base ends consonant. Returns the derived stem phonemes; if
    // nothing fires, returns `current_sph` unchanged.
    std::string tryEdNonMagicEStemFallbacks(
            const std::string& base,
            bool base_has_stress_override,
            const std::string& current_sph) const;
    // -ing stem fallback chain when CVC/nc magic-e is unavailable.
    // Tries: -rs/-ns magic-e, plain stem, magic-e add when base
    // ends consonant (and base has a vowel). Returns sph (or
    // current_sph unchanged when nothing fires).
    std::string tryIngNonMagicEStemFallbacks(
            const std::string& base,
            bool base_has_stress_override,
            const std::string& current_sph) const;
    // phonemesToIPA sub-helper: dispatch a single phoneme code at
    // `pstr[i]`. Greedy-matches against MULTI_CODES (skipping false
    // diphthongs after %/= via last_was_unstress), absorbs trailing
    // variant-marker digits, emits the IPA glyph (with any pending
    // stress marker if the code is a syllable nucleus), and advances
    // `i` past the consumed input.
    void emitPhonemeCode(const std::string& pstr,
                         int& i, int len,
                         std::string& pending_stress,
                         bool& last_was_unstress,
                         bool& last_code_was_vowel,
                         std::string& result) const;
    // applyStepC sub-helpers. The first identifies "-ing forms of
    // $strend2 stems whose dict phoneme carries secondary stress"
    // — used both for the current token and for each lookahead
    // candidate in the promotion loop. The second runs the
    // promotion lookahead (keep_sec -> primary when no following
    // stressed content). The third runs the leading-comma gated
    // comma->primary promotion at end of step C.
    bool isIngOfStrendSecondary(const std::string& word) const;
    void tryPromoteKeepSecToPrimary(
            const std::string& token_lower, size_t ti,
            const std::vector<Token>& tokens,
            bool& keep_sec, std::string& ph_codes) const;
    void applyCommaToPrimaryPromotion(
            bool keep_sec, bool needs_sec, bool phrase_matched,
            bool is_isolated_word, std::string& ph_codes) const;
    // Per-entry flag bundle for loadDictionary. Populated by
    // parseEntryFlags from parts[2..] plus phonemes_str (parts[1]).
    struct EntryFlags {
        bool noun = false, verb = false, past = false;
        bool pastf = false, nounf = false, verbf = false;
        bool atend = false, capital = false, atstart = false;
        bool onlys = false, only = false;
        bool grammar = false;
        bool strend2 = false, u2 = false;
        int  stress_n = 0;
    };
    // Phrase entry "(word1 word2) phonemes [flags]" parser. Stores
    // 2-word bigram pronunciations in phrase_dict_ /
    // phrase_split_dict_ / keep_sec_phrase_keys_. Returns true iff
    // `line` was a phrase entry (consumed); false means caller should
    // continue normal word-entry processing.
    bool parsePhraseEntry(const std::string& line);
    // Flag scanner over parts[2..] + phonemes_str (parts[1]).
    // Side-effect inserts into unstressed_words_ /
    // u_plus_secondary_words_ / unstressend_words_ / abbrev_words_
    // / word_alt_flags_. Populates `flags` with the per-entry bits.
    void parseEntryFlags(const std::vector<std::string>& parts,
                         const std::string& norm_word,
                         const std::string& phonemes_str,
                         EntryFlags& flags);
    // Storage dispatch for a single en_list entry (post-parse).
    // Routes the entry to the correct member container based on
    // its flags: noun_dict_ / verb_dict_ / past_dict_ / atend_dict_
    // / atstart_dict_ / capital_dict_ / onlys_bare_dict_ / dict_,
    // plus side-effect sets (pastf_words_ / strend_words_ /
    // compound_prefixes_ / only_words_ / etc.). dialect_cond is the
    // result of parseLeadingDialect (0 = unconditional).
    void storeDictionaryEntry(const std::string& norm_word,
                              const std::string& phonemes_str,
                              int dialect_cond,
                              const EntryFlags& flags);
    // processPhonemeString sub-phases. Each operates in-place on
    // `ph` and may consult `ph_in` / `rule_boundary_after` /
    // `findCode` / `isVowelCodeFn` (the multi-char code table and
    // vowel-test passed as callbacks, mirroring the existing
    // step-helper pattern).
    //
    // applyStressPhases: step 5 (primary insertion), 5.0 (=-suffix
    // stress shift), 5a (secondary placement) and its three
    // post-steps (cleanup / trochaic-prefix / final-syllable).
    void applyStressPhases(
            std::string& ph, const std::string& ph_in,
            bool force_final_stress,
            const std::vector<bool>& rule_boundary_after,
            const std::function<std::string(size_t)>& findCode,
            const std::function<bool(const std::string&)>&
                isVowelCodeFn) const;
    // applyPhonemeStringLatePhases: dim0 + demote-adjacent +
    // en_us reductions (5.5b/5.5b2/5.5c/5.5c2/5.5d/5.5d2/5.5e/
    // 5.5f/5.5g/5b/5c/6/6b/6c/6e) + final compound-syllabic-n
    // step (always runs).
    void applyPhonemeStringLatePhases(
            std::string& ph, const std::string& ph_in,
            bool is_en_us,
            const std::vector<bool>& rule_boundary_after,
            const std::function<std::string(size_t)>& findCode,
            const std::function<bool(const std::string&)>&
                isVowelCodeFn) const;
    // phonemizeText per-token body for word tokens. Runs the full
    // dispatch pipeline (number / acronym / clitic / phrase / period
    // / POS / atstart / atend / lemma / stress steps / sandhi /
    // IPA / R-linking / t-flap / emit) and updates the POS-context
    // counters. Modifies `ti` (may advance via clitic match) and
    // `first_word`, appends to `result`, and reads tokens by index.
    void processWordToken(
            const Token& token, size_t& ti,
            const std::vector<Token>& tokens,
            bool is_isolated_word, size_t last_word_ti,
            bool is_en_us,
            int& expect_past, int& expect_noun, int& expect_verb,
            std::string& result, bool& first_word) const;
    // Result of stemPhonemeFromDict's dict-lookup chain (steps 1-3
    // of the bulletproof rewrite — see lookupStemInDicts).
    struct StemLookupResult {
        std::string stem_ph;       // looked-up phonemes ("" if miss)
        std::string matched_stem;  // key whose phonemes we took
        bool used_onlys_bare = false;
        bool found_dict_entry = false;
    };
    // Steps 1-3 of stemPhonemeFromDict's bulletproof lookup chain.
    // Tries verb_dict_ (when SUFX_V applies and suffix isn't -s),
    // then onlys_bare_dict_ (for -s suffix) or dict_, then a
    // magic-e-stripped fallback. NEVER aliases iterators across
    // containers — `matched_stem` tracks which key was hit so the
    // stress override in step 4 can look up the right $N entry.
    StemLookupResult lookupStemInDicts(
            const std::string& stem_norm, bool stem_is_onlys,
            bool suffix_is_s,
            const RuleMatchResult& match) const;
    // Step 4 of stemPhonemeFromDict: apply $N stress override if
    // the lookup hit a dict, otherwise re-phonemize the stem via
    // rules (with combined $altN flags) and apply any stress
    // override that survives the noun-form-only / verb-flag gates.
    void applyStemStressOrRulesFallback(
            const std::string& stem,
            const std::string& stem_norm,
            const StemLookupResult& lookup,
            int word_alt_flags,
            const RuleMatchResult& match,
            std::string& stem_ph) const;
};

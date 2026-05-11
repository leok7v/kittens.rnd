# Phonemizer Origin

The C++ phonemizer in this directory implements the same English
letter-to-phoneme rule format and stress algorithm as the **espeak-ng**
project — https://github.com/espeak-ng/espeak-ng — licensed under
**GPL-3.0-or-later**.

KittenML's KittenTTS (the upstream project that this repo's MLX /
Core ML / ggml backends serve) bundles a phonemizer called
"CEPhonemizer" that is a C++ port of the espeak-ng English rule
engine. Their distribution attributes neither the engine nor the
rule data to espeak-ng. This file fills that gap so anyone reading
the phonemizer source here can follow the lineage.


## What came from espeak-ng

### Data files (verbatim)

`app/Resources/nano/en_list` and `app/Resources/nano/en_rules` are the
**source-format dictionary files** from espeak-ng's
[`dictsource/en_list`](https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/en_list)
and
[`dictsource/en_rules`](https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/en_rules).
The syntax — `(left_ctx)X(right_ctx)Y` rules, dialect markers `?3` /
`?6`, entry flags `$noun` / `$verb` / `$pastf` / `$onlys` /
`$strend2` / `$alt1`-`$alt6` etc. — is espeak-ng's. These files
are GPL-3.0-or-later.

### Algorithm and naming conventions (ported)

The rule-action opcodes used throughout `phonemizer.cpp` —
`RULE_DOUBLE`, `RULE_STRESSED`, `RULE_SYLLABLE`, `RULE_NOVOWELS`,
`RULE_LETTERGP`, etc. — appear with the same names in espeak-ng's
[`src/libespeak-ng/translate.h`](https://github.com/espeak-ng/espeak-ng/blob/master/src/libespeak-ng/translate.h).

Internal function names referenced in source comments — `SetLetterBits()`,
`WritePhMnemonic()`, `SetWordStress()` — are espeak-ng functions
defined in
[`src/libespeak-ng/tr_languages.c`](https://github.com/espeak-ng/espeak-ng/blob/master/src/libespeak-ng/tr_languages.c)
and
[`src/libespeak-ng/dictionary.c`](https://github.com/espeak-ng/espeak-ng/blob/master/src/libespeak-ng/dictionary.c).

The letter-group taxonomy (`LETTERGP_A` = vowels "aeiou",
`LETTERGP_B` = hard consonants, `LETTERGP_F` = voiceless consonants,
etc.) and the ASCII→IPA phoneme mnemonic encoding (the `ipa1[]`
table espeak-ng uses in `dictionary.c`) are reproduced one-for-one in
`rule_parser.h` and `ipa_table.h`.


## What was added on top in this repo

`phonemizer/phonemizer.cpp` and `phonemizer/phonemizer.h` have been
substantially refactored from the KittenML/CEPhonemizer port that
landed here — function decomposition (the original ~1300-line
`phonemizeText` is now a 35-line orchestrator over ~80 helpers),
single-entry/single-exit cleanup, removal of synthetic state flags,
and a battery of correctness fixes for stale-iterator and stale-key
bugs in the dictionary lookup path. Those changes are first-party.

None of the refactor changes the underlying algorithm: the goal was
strictly to make the existing logic match this repo's house style
without altering the phoneme strings the engine produces. Every
commit during that refactor preserved bench-md5 8/8 parity against
the pre-refactor reference output.


## License posture

The repository as a whole is Apache-2.0 (see `LICENSE` / `NOTICE.md`).
The two espeak-ng-derived rule data files (`en_list` and `en_rules`)
retain their upstream license (GPL-3.0-or-later) and are bundled here
for the engine to load at runtime.

If you reuse this code: treat the `en_list` / `en_rules` files as
GPL-3 obligations, and treat `phonemizer.cpp` / `phonemizer.h` as
inheriting the same algorithmic lineage — the C++ implementation was
written with espeak-ng's source as a reference, not in true clean-
room isolation.

# kittens.rnd

A research playground for running KittenTTS on Apple platforms via three
different inference backends — MLX, Core ML, and ggml/llama.cpp — so we
can compare quality, latency, memory, and engineering complexity head to
head on the same hardware.

The original [KittenTTS] is a tiny (~25 M parameter) text-to-speech
model from [KittenML]; the published reference implementation runs on
ONNX Runtime, and [KittenTTS-swift] is the official Swift port that
also goes through ONNX. This repo is orthogonal to those: it treats
KittenTTS as a fixed reference and asks what the lightest, fastest,
most native way to actually ship that model on a Mac and an iPhone
looks like once you skip ONNX entirely.

The specific weights we work with are
[`KittenML/kitten-tts-nano-0.1`][hf-model] — the 25 M-parameter "nano"
checkpoint. Other KittenML models are listed [on Hugging Face][hf-org].

[KittenML]: https://github.com/KittenML
[KittenTTS]: https://github.com/KittenML/KittenTTS
[KittenTTS-swift]: https://github.com/KittenML/KittenTTS-swift
[hf-model]: https://huggingface.co/KittenML/kitten-tts-nano-0.1
[hf-org]: https://huggingface.co/KittenML


## What's in here

```
app/         SwiftUI macOS app (KittensRnD) — three-backend A/B player
core/        Shared text/IO pipeline (preprocess, chunk, phonemize, load)
mlx/         MLX-Swift adapter — Apple's array library, dynamic shapes
coreml/      Core ML adapter — Apple's framework, fixed shape buckets
ggml/        ggml/llama.cpp adapter — single C file, CPU only
phonemizer/  CEPhonemizer C++ engine (used by all three backends)
scripts/     Python — HuggingFace -> our model formats, parity tests
vendors/     mlx-swift (vendored) and llama.cpp (submodule)
```

The Xcode project (`KittensRnD.xcodeproj`) sits at the repo root and
treats each top-level folder as a filesystem-synchronized group, so the
on-disk layout and the Xcode navigator stay in sync without any manual
project bookkeeping.


## Three backends, one model

All three backends run the exact same KittenTTS weights and produce
bit-identical audio for a given input. They differ in how the inference
graph is built and executed:

* **MLX** — the model is reimplemented in Swift on top of MLX-Swift.
  Fully dynamic shapes, easy to read, lazy-evaluated. Good for research
  iteration; uses Apple's MLX runtime which targets both CPU and GPU.

* **Core ML** — the PyTorch model is converted to a Core ML package via
  `coremltools`, with text-encoder and generator stages converted
  separately. Fixed-shape "buckets" for L (phoneme count) and N (audio
  frames) are pre-compiled; the runtime picks the smallest bucket that
  fits each chunk. Quantization variants the upstream Kittens app
  ships (fp32 / fp16 / int8w / int8wa); only **int8w** and **fp32**
  are bundled here because those are the two that empirically work
  end-to-end. fp16 hangs in MLModel load on some machines (the ANE
  ML Program loader churns fallback buffers for the largest bucket)
  and int8wa quantizes activations as well as weights, which strips
  fine vowel formants and produces breathy/whispery output. Both
  unbundled variants are reproducible via
  `scripts/convert_to_coreml.py`. Default is int8w — runs cleanly on
  both ANE and CPU, ~26 MB.

* **ggml/llama.cpp** — the model is rebuilt op-by-op in a single
  ~2600-line C file (`ggml/kittens-tts.c`) against the ggml graph API,
  with weights loaded from a custom GGUF file. CPU only — the model is
  dominated by tiny LSTM ops where ggml-metal's per-op kernel-launch
  cost outweighs its compute.


## Status / findings so far

* All three backends produce intelligible speech matching the reference
  PyTorch implementation.
* On Apple Silicon (M-series) the ggml CPU path runs at roughly 9× real
  time for the bundled "nano" model, so a Metal/ANE port is not worth
  the complexity for this workload.
* Mid-sentence audio pauses in long paragraphs were tracked down to a
  combination of the chunker splitting on sub-clause punctuation and
  the inter-chunk gap being inserted unconditionally. The chunker now
  splits only on `.!?`, and MLX/ggml pass whole sentences as one chunk.


## Build

Prerequisite: Xcode 16+ on Apple Silicon, Python 3.11+ for the
conversion scripts, CMake 3.20+ for ggml.

```sh
git clone --recurse-submodules https://github.com/leok7v/kittens.rnd
cd kittens.rnd
./scripts/build_ggml.sh         # builds vendors/llama.cpp/build-cpu/...
open KittensRnD.xcodeproj       # build & run the app
```

`build_ggml.sh` is a one-shot — Xcode does not invoke CMake. Re-run it
only if you bump the llama.cpp submodule or change ggml link flags.


## Headless bench

The app supports a CLI bench mode that runs each backend on the same
text and writes WAVs to `./tmp/`:

```sh
./Build/Products/Debug/KittensRnD.app/Contents/MacOS/KittensRnD \
    --bench --text "Hello world." --runs 3
```

By default it runs the full 8-row matrix (mlx, ggml, and 6 Core ML
variant×compute combos — 2 bundled variants × 3 compute units).
Use `--configs "mlx,ggml"` to run a subset.


## Credits

* **KittenTTS** by [KittenML] — the model, the reference Python
  implementation ([KittenTTS]), the official Swift/ONNX port
  ([KittenTTS-swift]), and the weights ([hf-model]). Without any of
  those there is nothing for this repo to play with.
* **CEPhonemizer (rule data)** — the `en_list` / `en_rules` phoneme
  rule files bundled under `app/Resources/nano/` are taken from the
  upstream KittenTTS distribution. The C++ G2P engine that consumes
  them (`phonemizer/phonemizer.cpp`) is first-party.
* **[MLX-Swift]** by Apple — vendored under `vendors/mlx-swift` for
  build reproducibility.
* **[llama.cpp / ggml]** by Georgi Gerganov and contributors — pulled
  as a Git submodule under `vendors/llama.cpp`. Only the `ggml/`
  subtree is linked; nothing from llama.cpp's LLM inference layer is
  used.

[MLX-Swift]: https://github.com/ml-explore/mlx-swift
[llama.cpp / ggml]: https://github.com/ggml-org/llama.cpp


## License

Apache 2.0 — see `LICENSE`. The KittenTTS model weights and the
phonemizer engine retain their respective upstream licenses; see the
`NOTICE.md` file for attribution details.

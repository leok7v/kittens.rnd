kittens.rnd
Copyright 2026 Leo Kuznetsov

This product is a research-and-development playground that runs the
KittenTTS model on Apple platforms via three different inference
backends. The code in this repository is licensed under the Apache
License, Version 2.0; see LICENSE for the full text.

The repository bundles or depends on the following third-party
components, each of which retains its own license:

KittenTTS
  https://github.com/KittenML/KittenTTS
  Hugging Face: https://huggingface.co/KittenML/kitten-tts-nano-0.1
  License: Apache License 2.0
  Used here: model weights (kitten_tts_nano_v0_8.safetensors,
  kitten_full.gguf, voices.safetensors).

  The phonemizer rule data files (en_list, en_rules) bundled under
  app/Resources/nano/ originate from the espeak-ng project
  (https://github.com/espeak-ng/espeak-ng, GPL-3.0-or-later) and
  retain that license. The C++ phonemizer engine in phonemizer/
  implements the same rule format and stress algorithm. See
  phonemizer/ORIGIN.md for the full lineage notes.

MLX-Swift
  https://github.com/ml-explore/mlx-swift
  License: MIT
  Used here: Vendored under vendors/mlx-swift for build
  reproducibility. See vendors/mlx-swift/LICENSE.

llama.cpp / ggml
  https://github.com/ggml-org/llama.cpp
  License: MIT
  Used here: Pulled as a Git submodule under vendors/llama.cpp.
  Only the ggml/ subtree (graph engine, CPU backend, GGUF) is linked;
  nothing from llama.cpp's LLM inference layer is used.

swift-numerics
  https://github.com/apple/swift-numerics
  License: Apache 2.0
  Used here: transitive dependency of MLX-Swift.

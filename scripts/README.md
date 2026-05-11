# scripts

Python tooling for converting the upstream KittenTTS model into the
formats consumed by each of the three Swift backends, plus
end-to-end parity tests.

The model artifacts that ship under `app/Resources/nano/` and
`app/Resources/coreml/` were produced by these scripts. The scripts are
checked in so the conversions are reproducible and reviewable, not
because you need to run them to use the app.


## What each script does

* **`convert_to_mlx_safetensors.py`** — pulls the original PyTorch
  state-dict from the upstream HuggingFace repo and writes a flat
  `kitten_tts_nano_v0_8.safetensors` plus `voices.safetensors` (the
  407-row voice prompt table) and `config.json`. These are consumed
  directly by the MLX backend.

* **`convert_to_coreml.py`** — converts the PyTorch model to a Core ML
  package, separately for the text encoder and the generator stage.
  Produces a fixed-shape "bucket" set (L=N×phoneme cap, N=audio frame
  cap) per quantization variant (fp32 / fp16 / int8w / int8wa). The
  app picks the smallest bucket that fits each chunk at runtime.

* **`convert_to_gguf.py`** — converts the PyTorch model to a custom
  GGUF file with all weights concatenated in the layout that
  `ggml/kittens-tts.c` expects. The file format is GGUF v3 but the
  tensor naming is project-specific; this is the only producer.

* **`tts_e2e.py`** — reference PyTorch end-to-end runner. Generates a
  WAV from text using the original implementation, useful as ground
  truth when checking the Swift backends.

* **`compare_wavs.py`** — pairwise spectral diff (cos similarity, peak
  RMSE, peak-frequency drift) between two WAVs. Used to validate that
  a backend matches the reference within tolerance after a numeric
  fix.

* **`torch_kitten.py`** — a lightweight PyTorch reimplementation of
  the inference graph. The MLX and ggml ports were translated from
  this rather than from the original ONNX export, because the ONNX
  graph aggressively fuses ops in ways that obscure the math.


## Running

These scripts assume Python 3.11+ with PyTorch, transformers,
huggingface-hub, soundfile, numpy. A `requirements.txt` is included
for reference but the conversions are one-shots — install whichever
deps the specific script you want to run actually imports.

# Local-LLM-Launcher-GUI

A friendly, browser-based control panel for downloading and running large language
models on your own computer — with **vLLM** or **llama.cpp** — built for people who
don't want to memorize esoteric command-line flags.

Every setting is rated **green / yellow / red against your actual hardware and the
model you picked**, with a one-or-two-sentence plain-English explanation of what it
does. A live memory gauge shows whether the model will fit *before* you launch, and
failed launches are translated from raw engine logs into human explanations.

## Credits

This project is built on the shoulders of
**[vllm-cli](https://github.com/Chen-zexi/vllm-cli) by Chen-zexi** (MIT license).
The flag catalog concept, configuration profiles, server lifecycle management, and
model discovery are adapted from vllm-cli. Thank you!

Engines: [vLLM](https://github.com/vllm-project/vllm) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp)

## Features

- **Hardware panel** — detects NVIDIA GPUs (with live VRAM use), Apple Silicon,
  CPU/RAM, and which engines are installed (vLLM native, vLLM Docker, llama.cpp).
- **Model library** — finds models already on your machine (HuggingFace cache and
  any GGUF folders you add), shows size/format/quantization, and a fit badge
  (Fits / Tight / Won't fit) for your hardware.
- **Search & download** — search HuggingFace from the GUI; for GGUF repos you pick a
  single quant file (with sizes shown) instead of downloading the whole repo.
- **Launch screen** — pick a model, get hardware-computed presets
  (Safe / Max context / Max speed / Tight fit), tune flags with traffic-light
  feedback, and watch the memory gauge update live. The launch button is GO-green,
  CAUTION-amber, or locked when settings would fail.
- **Server dashboard** — running servers with status lights, live logs, one-click
  endpoint copy (OpenAI-compatible `/v1`), a built-in test chat, and plain-English
  explanations when a launch dies (out-of-memory, quantization mismatch, gated
  model, port in use, …).

## Install

Requirements: Python 3.10+, and at least one engine —
[vLLM](https://docs.vllm.ai) (pip or Docker) and/or
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`).

```bash
pip install .          # from this repo's root
local-llm-launcher     # opens http://127.0.0.1:8765 in your browser
```

The GUI binds to 127.0.0.1 only (not reachable from other machines).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # backend tests

cd frontend
npm install
npm run dev                     # dev server with API proxy (run the backend too)
npm run build                   # builds into src/local_llm_launcher/static/
```

## License

MIT — see [LICENSE](LICENSE). Portions adapted from
[vllm-cli](https://github.com/Chen-zexi/vllm-cli), © Chen-zexi, MIT license.

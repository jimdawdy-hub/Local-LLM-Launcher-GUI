# Background: how vllm-cli works, and what this project changed

[vllm-cli](https://github.com/Chen-zexi/vllm-cli) by **Chen-zexi** is a
terminal (text menu) application for configuring and running vLLM servers. It
solves a real problem well: vLLM has dozens of command-line flags, and most
users want the same handful of configurations over and over. This document
explains how vllm-cli is built, which ideas this project carried over, and
what's different in **Local-LLM-Launcher-GUI**.

## How vllm-cli works

vllm-cli (~28,000 lines of Python) is organized into a few key pieces:

| Area | What it does |
|---|---|
| `config/` | A **flag schema** (`schemas/argument_schema.json`) — every vLLM flag with its type, default, category, and a short description, plus **profiles** (`schemas/default_profiles.json`) — named bundles of flag values like `standard`, `high_throughput`, `low_memory`, and hardware-specific GPT-OSS profiles. |
| `server/` | `VLLMServer` — launches `vllm serve ...` as a subprocess, manages a process group (so Ctrl+C doesn't kill it accidentally), tails its log output into an in-memory queue plus a log file, and polls `/health` and `/v1/models` to detect when the server is actually ready. |
| `models/` | Scans the HuggingFace cache (`~/.cache/huggingface/hub`) and Ollama's model directories to find models already on disk, and reads `config.json` for metadata. |
| `system/` | Runs `nvidia-smi` to report GPU name, total/free VRAM, and compute capability; also reports CPU, RAM, and which vLLM features are available. |
| `ui/` | A [Rich](https://github.com/Textualize/rich)-based terminal interface — menus, tables, a live log viewer, and a "GPU stats bar." |
| `proxy/` | An experimental multi-model proxy that can run several models behind one API endpoint with sleep/wake GPU management. |

The result is a polished `vllm-cli` command: pick a model, pick (or
customize) a profile, and it builds and runs the right `vllm serve ...`
command for you, with a live dashboard.

## What we kept (and adapted)

These ideas from vllm-cli are the foundation of this project — credited in
source file headers throughout:

- **A flag catalog as data, not code.** Just like vllm-cli's
  `argument_schema.json`, this app's `data/flags_vllm.json` and
  `data/flags_llamacpp.json` describe every supported flag — type, default,
  category, and (new) a **plain-English explanation** for non-experts. The
  catalog drives both the command builder and the UI form.
- **Named presets over raw flags.** vllm-cli's `default_profiles.json`
  inspired this app's presets (*Safe*, *Max context*, *Max speed*, *Tight
  fit*) — except here they're **computed from your live hardware and the
  selected model**, not static JSON.
- **Server lifecycle management.** `engines/base.py`'s `LocalServer` is a
  direct descendant of vllm-cli's `VLLMServer`: subprocess in its own process
  group (clean Ctrl+C isolation), log file + tailing, health polling. It's
  simplified to read logs from disk (so it survives a GUI restart) and
  generalized to work for llama.cpp's `llama-server` too.
- **Model discovery from the HuggingFace cache.** `discovery.py` scans
  `~/.cache/huggingface/hub` the same way vllm-cli's `models/discovery.py`
  does — reading `config.json`, detecting quantization method, and (new)
  detecting GGUF files and their quantization level from the filename.
- **GPU detection via `nvidia-smi`.** `hardware.py` follows the same approach
  as vllm-cli's `system/gpu.py` — parse `nvidia-smi --query-gpu=...
  --format=csv`.
- **Dynamic tensor-parallel defaults.** vllm-cli's `ProfileManager` sets
  `tensor_parallel_size` to the GPU count on multi-GPU systems automatically;
  this app's presets do the same.

## What's new

| Area | vllm-cli | This project |
|---|---|---|
| **Interface** | Terminal (Rich TUI) | Browser GUI (FastAPI + React), so it's approachable without a terminal |
| **Engines** | vLLM only | vLLM (native or Docker) **and llama.cpp** (`llama-server`, GGUF models) |
| **Flag guidance** | Description + hint strings | **Red/yellow/green ratings per flag**, computed live against your actual hardware and the selected model — see [ARCHITECTURE.md](ARCHITECTURE.md#the-advisor) |
| **Memory budgeting** | Shown as raw GB in system info | A **segmented memory gauge** showing model weights vs. conversation memory (KV cache) vs. working space, against what's actually free right now |
| **Apple Silicon** | Not supported (vLLM has no Metal backend) | Detected; the GUI steers Apple Silicon users to llama.cpp automatically |
| **Failure handling** | Raw log output | A **failure-translation table** (`data/failures.json`) maps known error patterns (OOM, quantization mismatch, reasoning-parser/model-family mismatch, port conflicts, gated repos, removed flags) to plain-English explanations |
| **Downloads** | Not built in | Search HuggingFace, download a full repo or a single GGUF quant file, with live progress |
| **Testing model output** | None | Built-in chat box that proxies to the running server's OpenAI-compatible API |
| **Command construction** | Always emits the full profile | Emits **only flags you explicitly set** — see the changelog entry on `--swap-space` for why this matters across engine versions |

## A real example: the advisor in action

During development, a 31B AWQ-quantized model (`QuantTrio/gemma-4-31B-it-AWQ-6Bit`)
failed to load with:

```
Failed to load model - not enough GPU memory ... CUDA out of memory.
Tried to allocate 1.31 GiB. GPU 0 has ... 944.44 MiB free
```

Total VRAM across both GPUs was sufficient — but GPU 0 (the one driving the
desktop's monitors) had less *free* memory than the other card, by about
400 MB, because the desktop environment and browser were using it. The
advisor now checks **each GPU's free memory individually** against its share
of the model weights, and will warn (or block) *before* you click Launch if
one card is short — naming the card and how much to free. This is exactly the
kind of "cryptic flag, hours of trial and error" problem this project exists
to shorten.

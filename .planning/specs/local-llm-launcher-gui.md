# Spec: Local-LLM-Launcher-GUI

## What it is

A friendly, browser-based GUI for downloading and launching local LLMs with **vLLM** or
**llama.cpp**, aimed at inexperienced users on varied hardware (NVIDIA GPUs, Apple
Silicon, CPU-only). It removes the "esoteric flag trial and error" by rating every
option red/yellow/green against the user's actual hardware and the selected model,
with short plain-English explanations for every flag.

Credit: core concepts (flag catalog, profiles, server lifecycle management, model
discovery) adapted from [vllm-cli](https://github.com/Chen-zexi/vllm-cli) by Chen-zexi
(MIT license). Credit appears in README, the GUI's About section, and source headers
where code is adapted.

## Users & success criteria

- A user who has never touched vLLM can: see their hardware, pick/download a model,
  see instantly whether it fits, click Launch with safe defaults, and get a working
  OpenAI-compatible endpoint plus a built-in chat box to verify it.
- A power user can open "Advanced" sections and tune any supported flag, still with
  traffic-light feedback.
- Wrong-by-default footguns (e.g. `gpu_memory_utilization 0.90` on 16 GB consumer
  cards, FP8 35B on 2×16 GB, GGUF on vLLM) are flagged BEFORE launch, not after a
  10-minute failed load.

## Architecture

- **Backend:** Python 3.10+ FastAPI app (package `local_llm_launcher`), launched via
  `local-llm-launcher` console script; serves REST API + built frontend; opens browser.
- **Frontend:** React + Vite + Tailwind, built to static assets committed under the
  package (end user needs no Node).
- **Engines:**
  - vLLM — two launch modes, auto-detected: *native* (`vllm` on PATH / importable) and
    *Docker* (`vllm/vllm-openai` image; required flags `--ipc=host`, `--gpus all`,
    HF cache volume mount handled automatically).
  - llama.cpp — `llama-server` binary (PATH, common install locations, or
    user-configured path). If missing, GUI shows per-platform install guidance.
- **Processes:** subprocess lifecycle adapted from vllm-cli's `VLLMServer`
  (process group, log file + ring buffer, health polling on `/health` & `/v1/models`).
  State survives GUI restarts via a small JSON registry (`~/.local-llm-launcher/`).

## Core features (v1)

### 1. Hardware panel
- Detect: NVIDIA GPUs (name, VRAM total/free, compute capability via nvidia-smi),
  Apple Silicon (chip + unified memory), CPU cores, system RAM, free disk space,
  vLLM availability (native/docker), llama.cpp availability.
- Plain-English summary, e.g. "2× RTX 5060 Ti (16 GB each, 32 GB total). Good for
  models up to ~35B at 4-bit."

### 2. Model library
- Discover installed models: HuggingFace cache (safetensors repos) and GGUF files
  (HF cache + configurable folders).
- Each model card shows: size on disk, format (safetensors/GGUF), quantization (from
  config.json / GGUF filename), recommended engine, and a fit badge
  (green/yellow/red) for current hardware.
- Search HuggingFace Hub and download (via huggingface_hub) with progress shown in
  the GUI. For GGUF repos, list the individual .gguf files with sizes so the user
  picks one quant file, not the whole repo.

### 3. Launch screen (the heart of the app)
- Pick model → engine auto-suggested (GGUF → llama.cpp; safetensors → vLLM; on Apple
  Silicon vLLM is marked red/unsupported).
- VRAM budget gauge: estimated weights + overhead + KV cache vs available memory,
  updates live as flags change.
- Flag panel: curated catalog (~25 vLLM flags, ~15 llama.cpp flags), each with:
  - short plain-English explanation (one or two sentences, no jargon without
    a parenthetical),
  - a traffic light evaluated against (hardware, model, current value),
  - a hover/expand "why" message when yellow/red.
- Presets: "Safe (recommended)", "Max context", "Max speed", "Tight fit" — computed
  from hardware + model, not static.
- Launch → live log stream with a startup progress hint ("model loading can take
  10-15 min; silence during compile is normal"), success/failure detection with
  a translated error message (failure-pattern table from real-world vLLM failures).

### 4. Server dashboard
- List running servers (engine, model, port, uptime, endpoint URL with copy button).
- Live logs, stop/restart.
- Built-in minimal chat box (backend proxies to the server's OpenAI endpoint) to
  prove the endpoint works.

### 5. Advisor engine (tested core logic)
- Weight size: actual bytes on disk of model files (most reliable), falls back to
  param-count heuristics.
- Budget: sum of selected GPU VRAM − ~1.7 GB/GPU CUDA+NCCL overhead, × gpu_memory_utilization
  (vLLM) or whole-file + context cost (llama.cpp). Apple Silicon: unified memory ×
  a recommended fraction (llama.cpp Metal).
- KV-cache estimate scaled by max_model_len / context size.
- Per-flag rules engine (data-driven), e.g.:
  - `gpu_memory_utilization` > free/total per GPU → red; > 0.90 on consumer card → yellow
  - `tensor_parallel_size` > GPU count → red; not a divisor of GPU count → yellow
  - GGUF model selected with vLLM engine → red ("use llama.cpp")
  - `--quantization` explicitly set → yellow ("vLLM auto-detects; mismatch aborts launch")
  - cpu-offload enabled → yellow ("makes it fit, ~10× slower")
  - `--n-gpu-layers` (llama.cpp) more layers than VRAM allows → yellow/red
- Overall fit: green (≤75% of budget), yellow (75–95%), red (>95% or hard blocker).

## Out of scope (v1)
- Multi-model proxy, LoRA adapters, Windows native (WSL works), AMD/Intel GPUs
  (detected and reported as "CPU/llama.cpp recommended"), auto-download of llama.cpp
  binaries, authentication (binds 127.0.0.1 only).

## Quality bar
- TDD for advisor, command builders, hardware parsing, model discovery (pytest).
- Frontend builds clean; manual smoke + Playwright-less API smoke test.
- No secrets stored; HF token kept in OS keyring-less local config file chmod 600.

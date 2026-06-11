# Architecture

This document covers the technology choices, the full module map, and how
data flows through the app — from "user moves a slider" to "command launched
on the GPU."

## Stack choices, and why

| Layer | Choice | Why |
|---|---|---|
| Backend framework | **FastAPI** (Python) | The hardware/model logic (`nvidia-smi` parsing, HuggingFace cache scanning, subprocess management) is naturally Python — it's the same language vLLM and the model ecosystem already use, so there's no language bridge for anything filesystem- or process-related. FastAPI gives typed request/response models (via Pydantic) and automatic validation with very little boilerplate. |
| Frontend framework | **React + Vite** | A component-per-view structure maps cleanly onto the app's five screens (Dashboard, Models, Launch, Servers, Settings), each of which is mostly "fetch some JSON, render a form or a list." Vite gives a fast dev server with an API proxy and a simple, fast production build. |
| Styling | **Plain CSS with design tokens** (`theme.css`), no UI framework | The traffic-light system (red/yellow/green) and the segmented memory gauge are bespoke visual elements that don't map onto a generic component library. A small token-based stylesheet (colors, spacing, radii as CSS variables) keeps ~30 components visually consistent without pulling in Tailwind/MUI/etc. for what is fundamentally a few dozen custom components. |
| Packaging | Frontend builds **into** `src/local_llm_launcher/static/` | `pip install .` ships a complete app — Python backend *and* prebuilt frontend assets — as one package. End users never need Node.js or `npm`. |
| State/persistence | Flat JSON files under `~/.local-llm-launcher/` | No database needed: the app tracks a handful of running-server records, download jobs, and settings. JSON files are human-inspectable and trivially backed up. `settings.json` (which may hold a HuggingFace token) is written with `chmod 600`. |
| Process management | `subprocess.Popen` with `start_new_session=True` | Each launched model server runs in its own process group, isolated from the GUI's own signal handling — stopping the GUI doesn't kill running models, and stopping a model server (SIGTERM → SIGKILL fallback) doesn't affect the GUI or other servers. |
| Tests | `pytest` (backend only) | The advisor's rules, the command builders, and the server lifecycle are the parts most likely to have subtle bugs (and the parts a wrong answer is most costly for — e.g. silently emitting a flag the installed engine doesn't support). These are pure-Python and fast to test in isolation; the frontend was verified with real browser screenshots and a live end-to-end model launch instead of a JS test suite. |

## Directory layout

```
Local-LLM-Launcher-GUI/
├── docs/                         # you are here
├── src/local_llm_launcher/
│   ├── app.py                    # FastAPI app factory; serves API + built frontend
│   ├── api.py                    # All REST routes
│   ├── __main__.py               # `local-llm-launcher` CLI entry point
│   ├── hardware.py                # GPU / Apple Silicon / CPU / RAM / engine detection
│   ├── discovery.py               # Find installed models (HF cache + GGUF folders)
│   ├── catalog.py                 # Loads flag catalogs from data/*.json
│   ├── advisor.py                 # The traffic-light rules engine (see below)
│   ├── failures.py                # Translates engine log errors to plain English
│   ├── downloads.py               # HuggingFace search + download manager
│   ├── registry.py                # Persists/manages running server processes
│   ├── config.py                  # User settings (HF token, GGUF folders, etc.)
│   ├── data/
│   │   ├── flags_vllm.json         # vLLM flag catalog (label, help, type, category)
│   │   ├── flags_llamacpp.json     # llama.cpp flag catalog
│   │   └── failures.json           # Log-pattern → plain-English message
│   ├── engines/
│   │   ├── _args.py                # Shared: config dict → CLI args using the catalog
│   │   ├── base.py                 # LocalServer: subprocess lifecycle, logs, health
│   │   ├── vllm_native.py          # `vllm serve ...` command builder
│   │   ├── vllm_docker.py          # `docker run vllm/vllm-openai ...` command builder
│   │   └── llamacpp.py             # `llama-server ...` command builder
│   └── static/                     # Built frontend (generated, committed)
├── frontend/
│   └── src/
│       ├── App.jsx                 # Shell: nav rail + tab routing
│       ├── api.js                  # fetch wrapper for the backend API
│       ├── components.jsx          # Led, Badge, Toast, VramGauge, FitVerdict
│       ├── theme.css               # Design tokens + all styling
│       └── views/
│           ├── Dashboard.jsx        # Hardware summary, engines, running servers
│           ├── Models.jsx           # Installed models, search, downloads
│           ├── Launch.jsx           # The flag panel + advisor + launch button
│           ├── Servers.jsx          # Running servers, logs, test chat
│           └── Settings.jsx         # HF token, GGUF folders, llama.cpp path, about
└── tests/                           # pytest — 74 tests across all backend modules
```

## Data flow: from slider to subprocess

1. **Hardware detection** (`hardware.py`) runs `nvidia-smi`, checks for Apple
   Silicon (`sysctl`/`platform`), reads CPU/RAM/disk via `psutil`, and checks
   whether `vllm` (pip or Docker image) and `llama-server` are available. This
   is cached for 5 seconds (`api.py`'s `get_hardware()`) so rapid UI
   interactions don't hammer `nvidia-smi`, but the **Launch** screen polls
   every 8 seconds so free-VRAM numbers stay current while you're tuning
   settings.

2. **Model discovery** (`discovery.py`) scans
   `~/.cache/huggingface/hub/models--*/snapshots/<latest>/` for weight files
   (`.safetensors`, `.gguf`, `.bin`), reads `config.json` for architecture
   details (layer count, head dimensions — used for KV-cache math), and
   detects quantization from `config.json` or, for GGUF, from the filename
   (e.g. `Q4_K_M`, `IQ2_XS`).

3. **The Launch screen** loads the flag catalog for the chosen engine
   (`catalog.py` → `data/flags_*.json`) and computed presets
   (`advisor.presets()`). Each preset is a `{flag: value}` dict — there are no
   static "profiles" shipped as files; presets like *Max context* are computed
   by binary-searching context length against the live memory budget.

4. **As you change a setting**, the frontend POSTs `{engine, repo_id, config}`
   to `/api/advise`. `advisor.advise()`:
   - merges your config onto the catalog defaults (for *display/estimation*
     only — see below),
   - runs engine-specific rules (`_advise_vllm` / `_advise_llamacpp`),
   - computes a memory budget (weights + KV cache + working buffer vs. what's
     free, per GPU),
   - returns an overall verdict (green/yellow/red with a headline and detail
     lines) plus a per-flag rating.

   The frontend renders this as the LED next to each flag, the "why" note
   underneath it, the segmented memory gauge, and the big launch button's
   color.

5. **On Launch**, the frontend POSTs `{engine_mode, repo_id, config}` to
   `/api/servers`. The chosen engine module (`engines/vllm_native.py`,
   `vllm_docker.py`, or `llamacpp.py`) calls `_args.build_args_and_env()`,
   which converts **only the keys present in `config`** into CLI flags using
   the catalog's `cli_flag` mapping — catalog defaults are *never* silently
   added to the command line (see [CHANGELOG.md](CHANGELOG.md), the
   `--swap-space` entry, for why this is load-bearing).

6. **`registry.py`** wraps the resulting `argv`/`env` in a `LocalServer`
   (`engines/base.py`), starts it via `subprocess.Popen`, and persists a
   record to `~/.local-llm-launcher/servers.json` so the **Servers** tab shows
   it (and can reattach to it) even after a GUI restart.

7. **The Servers tab** polls `/api/servers` for status (running, healthy via
   `/health`, exit code) and `/api/servers/{id}/logs` for the last ~1000 log
   lines. If a server has stopped with a non-zero exit code,
   `failures.translate()` scans those logs against `data/failures.json` and
   shows a plain-English explanation instead of (or alongside) the raw log.

8. **The built-in chat box** POSTs to `/api/servers/{id}/chat`, which the
   backend proxies to the running server's own
   `http://127.0.0.1:<port>/v1/chat/completions` — letting you confirm the
   model actually answers without leaving the GUI or knowing what `curl` is.

## The advisor

`advisor.py` is the core of the project's value proposition, so it's worth
detailing on its own.

### Inputs

- **Hardware** (`Hardware.to_dict()`): GPU list (name, total/free VRAM,
  compute capability), Apple Silicon info, CPU cores, RAM, which engines are
  available.
- **Model** (`LocalModel.to_dict()`): repo ID, format (`safetensors`/`gguf`),
  total size on disk, quantization, `config.json` contents (for KV-cache
  math), and — for GGUF — the list of available quant files.
- **Config**: the user's current `{flag: value}` choices.

### Memory budget math

For vLLM: `needed = weights_size + kv_cache_estimate + working_buffer`, where
`kv_cache_estimate` is computed from the model's actual `num_hidden_layers`,
`num_key_value_heads`, and head dimension when `config.json` has them
(falling back to a per-parameter heuristic for GGUF models without that
data), scaled by context length and `kv_cache_dtype` (fp8 halves it).
`available = sum(VRAM of GPUs in use) × gpu_memory_utilization`.

The **per-GPU load-headroom check** (added after a real failure — see
[CHANGELOG.md](CHANGELOG.md)) separately checks each individual GPU's *current
free memory* against its share of the weights plus a fixed overhead, because
total capacity can be sufficient while one card (typically the one driving
the desktop) is short.

For llama.cpp: `available` is unified memory × 75% (Apple Silicon), GPU VRAM
minus a small driver overhead (NVIDIA), or system RAM × 80% (CPU-only).

### Rules

Each engine has a function (`_advise_vllm`, `_advise_llamacpp`) that:
- declares **hard blockers** (red, launch disabled) — e.g. a GGUF file
  selected with the vLLM engine, or vLLM selected on Apple Silicon;
- rates **individual flags** — e.g. `gpu_memory_utilization` above what's
  actually free (red), above 0.90 on a consumer card (yellow),
  `reasoning_parser` mismatched against the detected model family (red, since
  this aborts the launch instantly), `cpu_offload_gb > 0` (yellow, "~10x
  slower");
- returns an **overall verdict** that's the worst of: the memory-fit
  percentage, any red flag, and the per-GPU headroom check.

### Presets

Presets are **functions of (model, hardware)**, not static templates:
- *Safe* uses all available GPUs at a conservative utilization and context.
- *Max context* binary-searches the largest context length that keeps the
  memory estimate under ~90% of budget.
- *Tight fit* (vLLM) computes `gpu_memory_utilization` from each GPU's
  *currently free* memory, minus a small safety margin.

## llama.cpp integration details

llama.cpp support required a few things vLLM didn't:

- **GGUF file selection.** A single HuggingFace repo can contain many `.gguf`
  files (different quantizations of the same model). `discovery.py` lists
  them all with sizes; the Launch screen lets you pick one
  (`config.gguf_file`), and `engines/llamacpp.py` resolves it to a path.
- **Binary discovery.** `hardware.py` searches common install locations for
  `llama-server`, **preferring source builds** (e.g.
  `~/Projects/llama.cpp/build*/bin/llama-server`, which are typically built
  with CUDA/Metal support) over generic prebuilt binaries (which are often
  CPU-only). The path can be overridden in Settings.
- **Shared library path.** Source/prebuilt llama.cpp binaries ship `.so`
  files alongside the binary; `engines/llamacpp.py` sets `LD_LIBRARY_PATH` to
  the binary's directory automatically.
- **Auto-fit.** Recent llama.cpp versions measure GPU memory and choose
  `--n-gpu-layers` automatically ("fitting params to device memory"); this
  app leaves that flag unset by default rather than forcing `999`, and the
  catalog explains both the automatic and manual options.

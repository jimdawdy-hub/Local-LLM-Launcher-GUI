# Changelog

All notable changes to this project, in the order they happened. Dates are
when the work was done.

## 2026-06-11

### Fixed: Open WebUI ignored the launcher's model connections after first boot

**The problem:** the Open WebUI launch feature (below) passed running-model
endpoints via `OPENAI_API_BASE_URLS` env vars — but Open WebUI treats those as
"PersistentConfig": it reads them **only on its very first ever boot**, saves
them into its own internal database (`webui.db`), and silently ignores the env
vars on every boot after that. So once the user had opened Open WebUI's
settings even once, newly launched models never appeared, and each one had to
be added by hand under Admin Panel → Settings → Connections.

**Fix:** since the launcher owns the Open WebUI process and only launches it
when it's stopped, `openwebui.py` now merges the running models' endpoints
directly into that saved config before starting it (`merge_connections()`):

- endpoints for currently running models are appended (or re-enabled if
  already saved — `localhost` and `127.0.0.1` are recognized as the same
  server, so no duplicates);
- stale entries the launcher itself added earlier (recognizable by their
  `sk-local` placeholder key) for models no longer running are pruned;
- connections the user added by hand — including real OpenAI API keys — are
  never touched.

The database is located the same way Open WebUI itself finds it (`$DATA_DIR`,
else the `data/` folder inside the installed `open_webui` package — found by
asking the interpreter named in the `open-webui` script's shebang, since it's
often installed under a different Python than the launcher). The env vars are
still passed too, covering a truly fresh install. Verified against a copy of
the real `webui.db` on this machine; 5 new tests (10 total for Open WebUI).

### Added: Open WebUI launcher on the Dashboard

A new panel under "Running now" on the Dashboard launches
[Open WebUI](https://github.com/open-webui/open-webui) — a polished chat
interface — and wires it to your running models in one click.

- The app detects whether `open-webui` is installed (on PATH). If it isn't,
  the button is greyed out and a copyable textbox shows the install command
  (`pip install open-webui`).
- When it is installed, clicking **Launch Open WebUI** (1) starts
  `open-webui serve --port 3000`, (2) pre-connects it to every model server
  currently running here by passing their OpenAI-compatible endpoints via Open
  WebUI's `OPENAI_API_BASE_URLS` / `OPENAI_API_KEYS` env vars (so the models
  are ready to chat with on open), and (3) opens your browser to it
  automatically once its `/health` endpoint responds.
- Open WebUI is tracked with the same process lifecycle as model servers
  (persisted to `~/.local-llm-launcher/openwebui.json`, survives GUI restarts)
  and can be stopped from the same panel.
- Port defaults to 3000 to avoid llama.cpp's 8080. New module
  `openwebui.py`, three API routes (`/api/openwebui`, `/launch`, `/stop`), and
  5 tests.

### Added: text-only mode for multimodal (vision/audio) models

**The problem:** `QuantTrio/gemma-4-31B-it-AWQ-6Bit` is a *multimodal* model
(`Gemma4ForConditionalGeneration` — it has a vision tower and audio encoder
alongside the language model). Launched normally, vLLM loads the entire
multimodal stack, and the vision/audio weights plus multimodal profiling
memory contributed to a weight-load OOM on 2×16 GB cards. For pure text chat,
all of that is wasted VRAM.

**Fix:**
- Added **`--language-model-only`** (vLLM) and **`--no-mmproj`** (llama.cpp)
  to the flag catalogs as a "Text-only mode (skip vision/audio)" toggle, with
  plain-English help.
- `discovery.py` now detects multimodal models from `config.json` (a
  `*ForConditionalGeneration` architecture, or a `vision_config` /
  `audio_config` / `image_token_id` field) and exposes a `multimodal` flag on
  each model.
- `advisor.py` now, for a detected multimodal model: yellow-flags the
  text-only toggle as a suggestion when it's off (green confirmation when on),
  and — when the fit is tight or over budget — puts "this is a vision/audio
  model; turn on Text-only mode" as the *first* remedy in the overall verdict.
  When text-only is on, the advisor notes the real memory use will be below
  the estimate (which conservatively assumes the full model).
- The model-fit badges on the Models tab now reflect this too.

Verified on real hardware: the 31B model previously OOM'd **during weight
load** (building the LM head) on 2×16 GB cards. Relaunched with text-only mode
on (vLLM args confirmed `language_model_only: True`), its full weights loaded
in 11.63 GiB per worker and it sailed past that exact failure point — proving
the feature works. It then hit a *second, different* memory gate (KV cache),
which led directly to the next fix below.

### Added: KV-cache gate prediction (the "loads then refuses to start" failure)

**The problem (found while verifying text-only mode above):** vLLM checks
memory in two sequential gates — first it loads the weights, then it reserves
the KV cache (the GPU memory that holds the running conversation). The 31B
model cleared the weight gate but failed the KV gate:

```
ValueError: ... 0.75 GiB KV cache is needed, which is larger than the
available KV cache memory (0.09 GiB). ... estimated maximum model length is 384.
```

After the weights (~11.6 GB/card) and vLLM's activation/compile working set
(~1.8 GB/card) filled the 0.85-utilization pool on a 16 GB card, only ~0.09 GB
was left for KV cache — far short of what a 4096-token context needs. The model
loaded and *then* refused to start. The advisor hadn't predicted this: its
memory math was aggregate (total VRAM across both cards) and used a flat 1 GB
working buffer, so it rated the config "tight but should load."

**Fix:** `advisor.py` now models this gate explicitly and per-GPU. After
weights and a calibrated per-GPU working set (`VLLM_WORKING_SET_PER_GPU_GB`,
set to 1.8 GB from the observed load), it checks whether the requested
context's KV cache still fits in each card's pool. When it doesn't, the verdict
explains it in plain English and gives ordered remedies — lower the context to
a computed value that *would* fit, raise `gpu_memory_utilization` (only if the
card has free room), and/or enable fp8 KV compression. The post-launch failure
translator was also corrected (its pattern missed the word "the" in vLLM's
actual message) and now fires for this error.

**Net result:** for this genuinely tight 31B model on 2×16 GB with a desktop
running, the app now tells you *before* a ~15-minute load attempt that it will
clear the weight gate but stall at the KV gate, and what to change (free the
display GPU per the original vllm-cli tip, drop to a smaller model, or shorten
the context) — instead of letting you discover it the hard way.

### Fixed: per-GPU memory load-headroom check ("display tax")

**The problem:** A 31B AWQ model (`QuantTrio/gemma-4-31B-it-AWQ-6Bit`, 23.2 GB)
was launched on 2× RTX 5060 Ti (16 GB each) with `--tensor-parallel-size 2`.
Total VRAM was sufficient, and the advisor's overall verdict said "Tight fit —
should load." It crashed anyway:

```
Failed to load model - not enough GPU memory ... CUDA out of memory.
Tried to allocate 1.31 GiB. GPU 0 has a total capacity of 15.49 GiB of
which 944.44 MiB is free.
```

**Root cause:** GPU 0 (the one driving the desktop's monitors) had ~2 GB less
*free* memory than GPU 1, because the desktop environment, browser, etc. were
using it. The model needed ~14.6 GB free per card; GPU 0 only had ~13.5 GB.
Total-capacity math hid this per-card shortfall.

**Fix:** `advisor.py` now separately checks each GPU's *currently free*
memory against its share of the model weights plus a fixed CUDA/loading
overhead (`CUDA_OVERHEAD_GB` + `LOAD_BUFFER_GB`). If the busiest-but-still-used
GPU is short, the overall verdict is downgraded (yellow if short by ≤1.5 GB,
red if more), and the message names the specific GPU and how much memory to
free. `data/failures.json` also gained a pattern for "not enough GPU memory"
during *load* (distinct from a generic CUDA OOM), explaining the same
display-tax phenomenon after the fact.

### Fixed: stale free-memory readings on the Launch screen

**The problem:** the advisor's GPU-memory numbers only refreshed when a
setting changed. If the panel sat open while the user closed a browser tab to
free VRAM, the displayed numbers (and the warning above) didn't update.

**Fix:** the Launch screen now re-requests `/api/advise` every 8 seconds in
addition to on every config change, so free-memory figures track
`nvidia-smi` in near-real-time.

### Fixed: misleading "other programs" attribution

**The problem:** the load-headroom warning blamed 100% of a GPU's
used-but-not-by-any-process memory on "other programs," when part of it
(roughly 400 MB observed) is the NVIDIA driver's own reserved overhead, which
doesn't appear in `nvidia-smi`'s process list at all.

**Fix:** wording changed to "other programs *and the graphics driver* are
holding X GB," and the Dashboard's per-GPU bars are labeled "held (apps +
driver)" instead of "used," to set accurate expectations.

### Added: reasoning-parser / model-family mismatch detection

**The problem:** launching `QuantTrio/gemma-4-31B-it-AWQ-6Bit` (a Gemma-4
model) with `--reasoning-parser qwen3` aborted instantly:

```
RuntimeError: Qwen3ReasoningParser reasoning parser could not locate
think start/end tokens in the tokenizer!
```

**Fix:** `advisor.py` now detects the model's family from its repo ID
(`qwen3`, `deepseek_r1`, `gemma4` patterns) and:
- **red-flags** a reasoning-parser choice that doesn't match the detected
  family, naming the correct value, *before* launch — this gates the overall
  verdict so the launch button is disabled;
- yellow-flags a parser choice when the family can't be confirmed;
- yellow-flags when a detected reasoning family has no parser set at all
  (the previous, original behavior).

`data/failures.json` gained a matching pattern to translate this error if it
occurs anyway.

### Fixed: catalog defaults leaking into launch commands (`--swap-space` crash)

**The problem:** launching `google/gemma-4-12B` with vLLM failed immediately:

```
vllm: error: unrecognized arguments: --swap-space 4
```

**Root cause:** `engines/_args.py`'s `build_args_and_env()` started from
*all* catalog defaults and overlaid the user's config on top — so even
flags the user never touched (like `swap_space`, defaulting to `4`) were
emitted. A recent vLLM release removed `--swap-space` entirely, so the
hardcoded default broke every native vLLM launch on this machine.

**Fix:**
- `build_args_and_env()` now iterates **only over keys present in the user's
  config** — catalog defaults are used for *display and advisor estimates*
  only, never silently added to the command line.
- Removed `swap_space` from `data/flags_vllm.json` entirely (no longer a
  supported flag in current vLLM).
- `n_gpu_layers` (llama.cpp) default changed from `999` to "automatic" (`null`),
  matching recent llama.cpp's built-in GPU-memory auto-fit ("fitting params to
  device memory" in its logs) — so the app no longer overrides a feature the
  engine now does better itself.
- Added an `"unrecognized arguments"` pattern to `data/failures.json` so any
  future flag removed by an engine update produces a plain-English hint
  instead of a bare argparse error.
- Added a regression test (`test_only_explicit_config_is_emitted`) asserting
  that an empty/near-empty config produces a command containing none of
  `--swap-space`, `--dtype`, `--gpu-memory-utilization`,
  `--enable-prefix-caching`, `--batch-size`, `--n-gpu-layers`, or
  `--split-mode`.

### Fixed: llama.cpp engine selection (CPU-only fallback vs. CUDA build)

**The problem:** the app initially found and used a prebuilt **CPU-only**
`llama-server` binary (downloaded as a fallback during initial setup), so a
9B GGUF model loaded entirely on the CPU (Xeon E5-2690 v3) instead of the two
RTX 5060 Tis.

**Fix:** `hardware.py`'s `_LLAMA_LOCATIONS` search order now checks
**source-build directories first** (`~/Projects/llama.cpp/build*/bin/`),
since a from-source build is far more likely to have CUDA (`GGML_CUDA=ON`) or
Metal enabled than a generic prebuilt release. The user's existing CUDA build
(confirmed via `--list-devices` to see both GPUs, `ARCHS=1200` for Blackwell)
was pinned in `~/.local-llm-launcher/settings.json`. Re-running the same 9B
model afterward showed both GPUs in the device list, weights split ~5.7 GB /
~5.1 GB across them, healthy in ~7 seconds.

`engines/llamacpp.py` also now sets `LD_LIBRARY_PATH` to the binary's
directory when an absolute path is configured, since these builds keep their
`.so` files alongside the executable.

## 2026-06-10 — 2026-06-11: Initial build

### Project setup

- Cloned and reviewed [vllm-cli](https://github.com/Chen-zexi/vllm-cli) by
  Chen-zexi (MIT) for architecture and conventions.
- Wrote spec and plan documents (`.planning/specs/`, `.planning/plans/`).
- Scaffolded the Python package (`pyproject.toml`, `src/local_llm_launcher/`)
  and a Vite + React frontend (`frontend/`).

### Core backend (TDD)

- `hardware.py`: GPU detection via `nvidia-smi` parsing, Apple Silicon
  detection, CPU/RAM/disk via `psutil`, engine availability checks
  (`vllm` on PATH/importable, `vllm/vllm-openai` Docker image,
  `llama-server` in common locations).
- `discovery.py`: scans the HuggingFace cache for installed models, detects
  format (safetensors/GGUF), quantization (from `config.json` or GGUF
  filename), and parameter count (from repo name heuristics).
- `data/flags_vllm.json`, `data/flags_llamacpp.json`: curated flag catalogs —
  ~20 vLLM flags and ~15 llama.cpp flags, each with a plain-English
  explanation, type, category, and (where applicable) choices/range.
- `catalog.py`: loads and validates the flag catalogs.
- `advisor.py`: the traffic-light rules engine — overall fit verdict, memory
  budget (weights + KV cache + buffer vs. available), and per-flag ratings
  for both engines, plus computed presets.
- 24 advisor tests covering: small/huge/tight model fits, GGUF-on-vLLM and
  safetensors-on-llama.cpp blockers, Apple Silicon and no-GPU blockers,
  `gpu_memory_utilization` and `tensor_parallel_size` rules, quantization
  override warnings, CPU-offload warnings, context-length-vs-model-limit
  warnings, reasoning-parser hints, and llama.cpp-specific rules
  (cache-type/flash-attention dependency, thread count vs. CPU cores).

### Engines, server lifecycle, registry, downloads, API

- `engines/_args.py`, `vllm_native.py`, `vllm_docker.py`, `llamacpp.py`:
  command builders — each takes a model + config dict and returns
  `{argv, env, port, health_url}`.
- `engines/base.py`: `LocalServer` — subprocess lifecycle (start/stop with
  process-group signals), log file + tailing, health checks, status
  serialization for persistence.
- `registry.py`: `ServerManager` — launches servers, persists running-server
  records to `~/.local-llm-launcher/servers.json`, port-conflict detection.
- `downloads.py`: HuggingFace search, repo file listing (with GGUF quant file
  sizes), threaded download manager with progress polling.
- `failures.py` + `data/failures.json`: translates known engine error
  patterns (CUDA OOM, quantization mismatch, gated repo, port in use, missing
  binary, etc.) into plain English.
- `api.py`: full REST surface — `/api/hardware`, `/api/models` (+ search,
  repo detail, downloads), `/api/catalog/{engine}`, `/api/advise`,
  `/api/presets`, `/api/servers` (CRUD + logs + stop + chat),
  `/api/settings`, `/api/about`.
- `app.py` / `__main__.py`: FastAPI app factory serving the API and the built
  frontend; `local-llm-launcher` console script opens the browser.
- 67 tests passing (backend total at this point).

### Frontend

- Design pass via the frontend-design skill: a "launch control" dark theme
  (`theme.css`) — status LEDs (shape *and* color, for accessibility), badges,
  a segmented VRAM gauge as the signature visual element.
- Five views: **Dashboard** (hardware summary, engines, running servers),
  **Models** (installed models with fit badges, HuggingFace search/download),
  **Launch** (model/engine pickers, presets, the full flag panel with live
  advisor feedback and memory gauge, launch button), **Servers** (status,
  logs, stop/remove, built-in test chat), **Settings** (HF token, GGUF
  folders, llama.cpp path, About/credits).
- Frontend builds into `src/local_llm_launcher/static/` so `pip install .`
  ships everything.

### End-to-end verification

- Installed a CUDA-enabled `llama-server` build (later superseded by
  preferring the user's existing source build — see 06-11 fixes above).
- Launched `unsloth/Qwen3-1.7B-GGUF` via the GUI's own API, confirmed
  `/health`, and got a real chat completion through `/api/servers/{id}/chat`.
- Took Playwright screenshots of all five views to verify the design landed
  as intended.
- 67 tests passing; first commits made, with credit to Chen-zexi/vllm-cli in
  README, LICENSE, and source headers.

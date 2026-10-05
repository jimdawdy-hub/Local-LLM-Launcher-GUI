# Changelog

All notable changes to this project, in the order they happened. Dates are
when the work was done.

## 2026-10-04 — v0.4.8

- Fixed the sidebar showing a stale version (it still said v0.4.5). The
  sidebar now reads the version from the server's `/api/about`, so it can no
  longer drift from the installed release. A regression test refuses a
  hand-typed version in the source or built bundle.

## 2026-10-04 — v0.4.7

- Added `--allow-host NAME` (repeatable) so the GUI can be reached through a
  `tailscale serve` address or similar private name. The server still listens
  on 127.0.0.1 only; the DNS-rebinding guard now accepts those exact extra names
  and refuses wildcards. Engine source updates still require a direct local
  connection.

## 2026-10-04 — v0.4.6

- Fixed the launcher rejecting `--split-mode tensor` with `q8_0` or `q4_0`
  K/V cache before the engine could run. Current llama.cpp supports this
  combination (upstream PR #23792); Flash Attention on/auto remains required.
- Kept the selected cache types and GPU proportions unchanged in generated
  commands. Guidance now distinguishes NVFP4 model weights from K/V cache
  precision and warns that older engine builds may require an update, layer
  splitting, or f16 cache.
- Added builder, advisor and API regressions for Q5/NVFP4 GGUF metadata,
  compressed K/V combinations, Flash Attention refusal and format boundaries.
  These are synthetic launcher checks, not a GPU/model performance benchmark.

## 2026-09-28 — v0.4.5

One **Use MTP** checkbox for both engines, and plainer explanations for every
setting. Merged via PR #21, including the fixes from its pre-merge code review
(seven reviewers plus an independent validator; all five findings confirmed and
fixed below). 404 tests passing.

> **Verification note:** MTP flag selection was checked against the real source
> of vLLM 0.8.5, 0.9.2, 0.10.0, 0.10.2, 0.11.0 and 0.30.0 and llama.cpp build
> 11235, and the GGUF reader against llama.cpp's own `gguf-py`. No MTP launch
> on real GPUs has been run for this release, and no speed-up is claimed.

### Added: Use MTP (multi-token prediction) for vLLM and llama.cpp

- One **Use MTP** checkbox per engine. The launcher works out the flags the
  installed engine accepts, so nothing needs to be typed by hand:
  - **vLLM:** the method names and supported model families are read from the
    selected runtime's own source files (without importing vLLM). vLLM 0.11+
    gets the generic `mtp`; older releases get the family name they expect
    (`deepseek_mtp` for DeepSeek-V3, MiMo and GLM-4.5, `qwen3_next_mtp`,
    `ernie_mtp`). `num_speculative_tokens` comes from the model's MTP layer count.
  - **llama.cpp:** `--spec-type draft-mtp --spec-draft-n-max 3`, detected from
    `llama-server --help` and `--version`. The GGUF header is read to confirm the
    file really has MTP layers (sharded files included), and a separate
    `mtp-*.gguf` head next to the model is passed with `--spec-draft-model`.
    Per-architecture support starts at the build where llama.cpp added it
    (checked through build 11235).
- When the model has MTP layers but the installed engine can't use them, the
  setting turns red, launch is blocked, and the message says to update the
  engine or turn off MTP to run the model without it. A model without MTP
  layers is told to turn MTP off instead.
- The GGUF picker no longer defaults to a separate MTP head file as the model.

### Fixed before release (PR #21 review)

- **A different model's MTP head is no longer used.** In a shared GGUF folder,
  any `mtp-*.gguf` file was accepted, so a model without MTP layers could get a
  green verdict and be launched with another model's head. A head must now be
  built for the model's architecture (or be its `<arch>-assistant` head, as
  Gemma 4 ships).
- **Head files named like `model-mtp-Q8_0.gguf` are recognized**, matching
  llama.cpp's own rule (the name contains `mtp-`), both for MTP and for keeping
  head files out of the default model choice.
- **Your own `--spec-type` in extra raw flags now replaces the launcher's.**
  llama.cpp adds up repeated `--spec-type` values, so the old "your flags take
  priority" note was wrong; the launcher now leaves its own `draft-mtp` out and
  says to include it in your list.
- **A corrupt or hostile GGUF header shows the red "could not read" message**
  instead of failing the request. Declared sizes beyond the end of the file and
  arrays nested too deeply are rejected before anything is allocated.
- Tests now cover the warning (yellow) paths for partial engine evidence and
  the unreadable-file path.

### Changed: plainer setting explanations

- Every setting's help text for both engines was rewritten to say what the
  setting does and what it trades off, with NUMA, CUDA graphs, tensor
  parallelism, KV cache and backend choices explained in plain terms. The NUMA
  summary on the Launch screen and the NUMA advice messages were reworded too.

## 2026-09-27 — v0.4.0

Engine updates from source, multi-GPU and memory placement for llama.cpp, and
vLLM backend controls for consumer Blackwell GPUs, plus the fixes found while
reviewing them. Merged via PRs #17 and #18.

### Added: build an engine from source (issue #11)

- **Settings → Build current engine source** builds official llama.cpp
  (CPU or CUDA) or native vLLM (NVIDIA CUDA) on Linux, in separate folders
  under the app data directory. It shows prerequisites, the exact source
  revision, progress, and logs, and nothing is downloaded or upgraded until you
  choose to build.
- Only a validated executable is selected for future launches. A failed build
  keeps the previous executable, running servers keep theirs, and older copies
  stay available so you can restore an earlier executable path.

### Added: multi-GPU and memory placement for llama.cpp (issues #13, #15)

- **GPU selection and proportions:** device selection, proportional
  `--tensor-split`, main-GPU selection, and the `layer`, `row`, and
  experimental `tensor` split modes, each with compatibility checks.
- **MoE expert placement in RAM** (`--cpu-moe`, `--n-cpu-moe`) and
  **microbatch size** (`--ubatch-size`).
- **NUMA:** optional llama.cpp thread placement, and an **Interleave memory**
  option that runs native llama.cpp or vLLM under `numactl --interleave=all`.
- The memory-mapping and RAM-lock controls are translated to `--load-mode` on
  newer llama.cpp binaries that dropped the old flags.

### Added: vLLM backend controls for consumer Blackwell GPUs

- **Linear, MoE (mixture-of-experts), and attention backend selectors** under
  Advanced, including FlashInfer options and the optional B12X kernels for
  SM120/SM121 GPUs such as the RTX 5060 Ti. **Automatic** stays the default
  and adds no flags.
- The launcher checks the **selected runtime itself**, not its own Python: the
  flags and choices that vLLM executable accepts, and whether the optional
  `b12x` package is installed next to it. Docker images cannot be inspected,
  so their support is shown as unverified.
- Known conflicts are rejected before launch with a plain explanation: B12X
  on a GPU that is not SM120/SM121, B12X attention without BF16 computation,
  with an unsupported conversation-memory (KV cache) format, with context
  parallelism, or on MLA (latent attention) models, and B12X MoE with expert
  parallelism or with a quantization format known to be incompatible (it needs
  NVFP4 or MXFP4 expert weights).
- A float32 model with **automatic** precision qualifies for B12X attention,
  because vLLM runs float32 checkpoints as BF16 on these GPUs; the advice notes
  that embedding models run as float16 instead.
- vLLM 0.30's attention backend is named **`B12X`**. Typing `B12X_ATTN`, the
  name used in release discussion, gets a direct correction.
- Anything typed in **Extra arguments** is read the way vLLM reads it:
  underscore spellings such as `--linear_backend`, backend values in any case
  or with dashes (`B12X`, `flashinfer-b12x`), and vLLM's own `--device-ids`.

### Changed: behavior worth knowing about

- **GPU numbers always mean what `nvidia-smi` shows.** Native vLLM launches now
  set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. This includes a `CUDA_VISIBLE_DEVICES`
  set before starting the launcher, which CUDA would otherwise number
  fastest-card-first.
- **"Which GPUs to use" must be a comma-separated list** of distinct GPU
  numbers (or CUDA GPU IDs). Entries such as `0 1` or `gpu1`, which the engine
  would silently misread, are now refused with an explanation.
- **Downloads skip duplicate weight files.** When a model ships its weights as
  safetensors and also as older `.bin`/`.pt` files, only the safetensors copy
  vLLM actually loads is downloaded.
- **Installed-model sizes count only the files vLLM loads,** so models that
  ship the same weights several times now show their real size (for example
  `sentence-transformers/all-MiniLM-L6-v2` went from 0.27 GB to 0.08 GB). The
  Fit / Tight / Won't fit verdicts use the corrected size.
- **vLLM memory advice budgets only the GPUs the server will see,** including
  a `CUDA_VISIBLE_DEVICES` inherited from the launcher's environment.

### Fixed: settings, launches, and the dashboard (issues #7, #8, #9, #10, #16)

- Default-on vLLM prefix caching and chunked prefill can now be explicitly
  turned off (#7).
- A saved Hugging Face token can be kept, replaced, or cleared, and its
  on-screen mask is never saved in its place (#8).
- Running the tests no longer touches your real settings folder (#9).
- Settings are written atomically with private file permissions. Unknown API
  routes return JSON errors, and a malformed reply from a model server returns
  a clear gateway error (#10).
- Failed launches keep their logs, concurrent launches no longer grab the same
  port, and the download limit holds when downloads start at the same time
  (#10).
- Download progress counts only the files requested, and a finished download
  triggers one model scan instead of repeated scans (#10).
- The breadcrumb, used-VRAM display, live hardware updates, and the Refresh
  button's busy/success/error feedback now match reality (#10, #16).
- Hardware and server-list refreshes no longer pile up overlapping requests.
- Stopping a server no longer freezes status, logs, or other servers while it
  shuts down.
- The llama.cpp memory estimate counts a full-precision (F32) conversation
  memory cache at twice the size of F16; it was counted as F16.
- A llama.cpp you compiled yourself in its usual folder (for example
  `~/llama.cpp/build`) is preferred over a generic `llama-server` on your PATH,
  which is often CPU-only, as the documentation describes. Duplicate changelog
  content and unused code were removed.

### Fixed: engine source builds

- A failed build removes its downloaded source and build environment to free
  disk space, keeps its log for diagnosis, and never removes the executable
  currently selected.
- An engine path you restored by hand in Settings is no longer overwritten by
  an older completed build when you reopen Settings.
- A freshly built vLLM gets up to 300 seconds (was 30) for its first
  `--version` check, which loads its libraries from a cold start; a healthy
  build is no longer marked failed.

### Fixed: vLLM backend checks stay fast and accurate

- Launch checks now run **before** the server list is locked, so a slow runtime
  check never freezes status, logs, or Stop.
- The Launch page's advice never waits on a runtime check. It shows the last
  result, or "still being checked", and refreshes in the background.
- Each vLLM installation is checked by one process at a time, and callers that
  arrive meanwhile share its answer. The help check gets 60 seconds (was 8) to
  allow vLLM's normal start-up time.
- Check results are reused for 10 minutes, and are dropped as soon as the vLLM
  executable, its Python, or its installed packages change, so an open Launch
  page no longer re-runs the heavy check every minute.
- If the background check cannot start, advice shows the result as unknown and
  retries on the next refresh instead of failing.
- `device_ids = 0` means GPU 0 (it was treated as "all GPUs" for Docker and
  ignored by the GPU check).

Tests: 360 passing.

## 2026-08-11 — v0.3.1

### Fixed: server lifecycle and loopback binding (issues 1–6)

- **Port auto-increment now actually works.** The reassigned free port was
  previously computed *after* the command line was built, so a second server
  on a busy default port launched with the old port and failed to bind. The
  free port is now resolved into the config before the command is built.
- **"Stop" can no longer kill an unrelated process group.** Server records now
  persist the process's *start time* alongside its PID, so a recycled PID (for
  example after a reboot) is detected and never killed.
- **Model servers and Open WebUI now default to listening on `127.0.0.1`** —
  the same loopback-only behavior as the GUI itself — instead of `0.0.0.0`.
  A new **LAN access** toggle in Settings re-enables `0.0.0.0` (and the
  `--host` flag) when you actually want other devices to reach them.
- **Host header validation added.** The API now rejects requests whose `Host`
  header isn't `127.0.0.1` or `localhost`, closing the DNS-rebinding attack
  that could otherwise reach state-changing endpoints from a malicious page.
- **Docker launches no longer leak the HF token on disk.** The temporary
  `--env-file` holding the Hugging Face token is deleted when the server stops
  (or fails to start, or is removed).
- **Stop failures are reported honestly.** Stopping a server that no longer
  exists returns "no such server" (404); a server that *refuses* to stop now
  gets its own clear error (409) instead of the misleading 404.

### Fixed: Hugging Face model search (issue 12)

- Search now finds **all** model families, including NVFP4 and other
  image-text-to-text models that the old text-generation-only filter hid.
- Multi-word queries now AND their terms, so `glimmer muse NVFP4` returns
  models matching all three words (NVFP4, GGUF, MLX, etc. are matched against
  both names and tags).
- Removed arguments that the current `huggingface-hub` (1.27) dropped, which
  made every search fail with "check your internet connection."

### Added: model names link to their Hugging Face pages

Model names on both the search results and the "Installed on this computer"
lists now open the model's page on huggingface.co in a new tab.

## 2026-06-14 — v0.3.0

### Rebrand: OpsPulse-inspired frontend redesign

Complete visual overhaul of the frontend from the original "AI-generated dark theme" to a professional SaaS dashboard aesthetic inspired by [OpsPulse by Orbix Studio](https://me.muz.li/orbix-studio/opspulse-ai-operations-compliance-saas-dashboard-design).

**Design language:**
- Light background (`#f8f9fc`) with dark sidebar (`#111827`) — replaces the all-dark layout
- Plus Jakarta Sans (display + body) and JetBrains Mono (code) — replaces Space Grotesk / IBM Plex Sans
- White surface cards with subtle borders and shadows
- Metric cards with colored top borders and icon accents
- Status badges with dot indicators (green/amber/red/neutral)
- Structured data tables with striped headers

**New components:**
- `TopBar` — sticky header with breadcrumb navigation and action buttons
- `MetricCard` — OpsPulse-style stat cards for the Dashboard
- `RingGauge` — SVG circular progress indicator (replaces segmented VRAM bar)
- `StatusBadge` — dot + label badge component
- `ThemeToggle` — dark/light mode switch with localStorage persistence

**Restyled views:** Dashboard, Models, Launch, Servers, Settings — all updated to match the new design language.

### Added: dark/light theme toggle

A 🌙/☀️ toggle button in the top bar switches between the light OpsPulse theme and a dark variant. The user's preference is saved to localStorage and persists across sessions.

- CSS variables overridden via `[data-theme="dark"]` on `:root`
- Dark theme uses muted dark surfaces (`#181b25`), adjusted border/shadow colors, and brighter accent colors for contrast
- Hardcoded color overrides for data tables, log boxes, and hover states

## 2026-06-13 — v0.2.0

### Added: KV cache offload to system RAM (`--no-kv-offload`)

A new advanced toggle for llama.cpp forces the conversation memory (KV cache)
into system RAM instead of GPU VRAM. This is a last-resort option for when the
model weights fit on the GPU but the KV cache doesn't, even after compression.

- Added `--no-kv-offload` to `flags_llamacpp.json` as an advanced boolean flag.
- The advisor gives a **red "HEAVY SPEED PENALTY"** warning explaining the
  20–50% speed hit at moderate context and 2×+ at long context, with an extra
  warning for the worst-case combination (context >131K tokens + CPU KV).
- When enabled, the memory budget zeroes out KV cache from VRAM so the gauge
  shows only weights + compute buffers on the GPU.
- 5 new advisor tests covering: red flag, long-context extra warning, budget
  zeroing, reduced needed_gb, and no-flag when disabled.

### Added: free port auto-increment

The launcher, Open WebUI, and model servers no longer crash or refuse to start
when their default port is already in use. They now automatically find the next
free port and use it, printing a message when the port changes.

- New `find_free_port()` utility in `registry.py`.
- `__main__.py`: auto-increments from the requested port (default 8765).
- `openwebui.py`: auto-increments from the requested port (default 3000).
- `registry.py`: model server launcher auto-increments on port conflict.

### Fixed: TOCTOU race on settings file permissions (security)

`config.py` previously wrote the settings file (containing the HuggingFace
token) with `write_text()` (creates world-readable 0o644) and then called
`os.chmod(0o600)` in a separate syscall — leaving a window where any local
process could read the token. Now uses `os.open()` with mode 0o600 atomically.

### Fixed: HF token exposed in Docker process list (security)

`vllm_docker.py` previously passed the HuggingFace token via `-e
HUGGING_FACE_HUB_TOKEN=<token>`, making it visible in `ps aux` to any local
user. Secrets are now passed via `--env-file` (a temp file only Docker reads),
while non-secret env vars still use `-e`.

### Fixed: division by zero in GPU memory bar

`Dashboard.jsx` could produce `NaN%` in the GPU progress bar when
`vram_total_mb` was 0 (e.g. integrated GPU edge case or reporting error). Added
a guard.

### Fixed: dead code in download filter

`Models.jsx` had a `|| Date.now() === 0` clause in its download filter — always
false, likely a leftover debug expression. Removed.

### Fixed: modal accessibility

The GGUF file picker modal (`Models.jsx`) was missing `role="dialog"`,
`aria-modal`, focus management, and Escape key handling. Now includes all
three, plus auto-focus on the first interactive element when opened.

### Fixed: toast accessibility

Toast notifications (`components.jsx`) now have `aria-live="polite"` for screen
readers and a dismiss button so error toasts can be closed before their 9-second
timeout.

### Fixed: thread safety on hardware cache

`api.py`'s `_hw_cache` dict was read and written from FastAPI's thread pool
without synchronization. Added a `threading.Lock` around all access.

### Fixed: error messages leaking internal paths

API responses for search and repo-detail errors previously forwarded raw
exception text (which could contain tokens in URLs, internal file paths, or
stack traces). Now returns only sanitized, user-friendly messages.

### Added: PATCH /settings endpoint

A new `PATCH /api/settings` endpoint allows explicitly clearing settings (e.g.
`{"hf_token": null}`) — the existing PUT endpoint silently ignored null values.

### Added: download concurrency cap

`DownloadManager` now limits concurrent downloads to 3. Excess requests get a
clear error message instead of spawning unlimited daemon threads.

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

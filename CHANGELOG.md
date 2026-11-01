# Changelog

## 2026-10-31 — v0.6.1

- Render a Dashboard metric card for every detected GPU (supporting 3+ GPUs)
  with a responsive auto-fit grid instead of only displaying `GPU 0` and `GPU 1`.
- Update the Dashboard VRAM Budget subtitle to reflect the actual number of
  detected GPUs (`Used across all N GPUs`, `Used across both GPUs`, `Used on GPU`,
  or `No GPUs detected`).
- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` when probing `llama-server --list-devices`
  and launching `llama.cpp` so `CUDA0`, `CUDA1`, `CUDA2`, etc. match `nvidia-smi`
  PCI bus device numbering (consistent with vLLM and SGLang).
- Show GPU model names alongside `CUDA0`/`CUDA1`/`CUDA2` in the Launch tab's
  `llama.cpp devices:` summary line.

## 2026-10-25 — v0.6.0

- Refresh SGLang integration and parser choices for SGLang v0.5.22. Correct
  host binding, GPU selection, executable choice, and multimodal guidance.
- Add Hugging Face TGI v3.3.7 as a Docker engine, with setup guidance and
  conservative fit estimates. TGI is archived upstream.
- Support Podman's Docker command wrapper for TGI GPU selection, and require
  the exact launch image tag when detecting its availability. Clean temporary
  token files after failed container exits and support multiple selected GPUs.
- Show installed llama.cpp, vLLM, SGLang and Docker engine versions beside
  cached latest GitHub release versions on the Dashboard.
- Keep engine versions readable when an executable path is long or the
  dashboard is narrow.
- Expose llama.cpp prompt chunk reuse and vLLM CPU KV cache offloading.
- Keep SGLang API keys out of launcher controls because v0.5.22 logs its
  server arguments. Correct its quantization choices and align model fit
  recommendations with the Launch default.

## 2026-10-20 — v0.5.7

- Keep downloaded models visible when a newer cache revision contains metadata
  without model weights.
- Add **Edit** beside favorite launches to open their saved settings before starting.
- Add **Archive** with confirmation, the exact recovery-file location, and one
  private file containing archived favorites (#30, #32).
- Add **Reload previous settings** with the observed outcome of the retained
  model launch; preserve settings after removing a server and observed
  successful/failed outcomes across restarts (#31).
- Use bright green, yellow and red status badges and larger checkmark indicators
  in light and dark themes.
- Check the selected llama-server's support for prompt and terminal-color
  options before launch, retaining the original saved values.
- Replace machine-specific examples and inference-run anecdotes with general
  hardware and engine guidance.

## 2026-10-18 — v0.5.6

- Add named launch favorites, server configuration display, and benchmark
  summaries from supported engine logs (#28).
- Keep host binding, credentials and internal fields out of favorites. Relaunches
  follow the current Settings LAN policy.

## 2026-10-16 — v0.5.5

- Add temperature, system prompt and explicit host controls.
- Expose CPU MoE placement, expert-cache, feed-forward offload and lazy-loading controls.
- Add context shifting, slots, terminal color and system-prompt-file settings.
- Add reasoning format/effort/budget and speculative draft-model controls.
- Validate incompatible loading and placement settings before launch.

## 2026-10-06 — v0.5.1

- Make the MTP draft count configurable, with a conservative default of one token.
- Account for micro-batch/context working buffers, quantized-cache layer copies,
  conversation slots and recurrent state in llama.cpp memory estimates.
- Explain CPU fallback when a model exceeds GPU capacity.
- Add a configurable freeze watchdog for llama.cpp.

## 2026-10-05 — v0.5.0

- Read model metadata for hybrid, sliding-window and shared-cache estimates.
- Include speculative decoding memory and text-only encoder exclusions.
- Allow an explicit warned launch when memory estimates alone are unfavorable;
  invalid configurations and missing engines remain blocked.

## 2026-10-04 — v0.4.8

- Read the sidebar version from the installed backend.

## 2026-10-04 — v0.4.7

- Add exact-name allowed-host configuration for private reverse proxies.
- Keep loopback binding and local-only engine-update authorization.

## 2026-10-04 — v0.4.6

- Allow supported tensor splitting with quantized K/V cache and Flash Attention.
- Distinguish model weight compression from conversation-cache precision.

## 2026-09-28 — v0.4.5

- Add MTP selection for supported vLLM and llama.cpp model families.
- Detect engine versions, supported methods and separate MTP head files.
- Explain unsupported engine/model combinations and preserve raw-argument overrides.
- Improve plain-English settings guidance.

## 2026-09-27 — v0.4.0

- Add opt-in, isolated engine source builds with requirement checks and rollback paths.
- Add llama.cpp device selection, GPU proportions, tensor splitting and CPU expert controls.
- Add optional NUMA placement based on visible CPU/memory topology.
- Add vLLM linear, attention and MoE backend controls, including selected-runtime
  capability checks and architecture-specific validation.
- Translate memory-mapping controls for newer llama.cpp releases.
- Improve settings isolation, launch concurrency, resource estimates and diagnostics.

## 2026-08-11 — v0.3.1

- Fix server lifecycle, shutdown, port reservation and loopback binding.
- Improve Hugging Face search and link model names to repository pages.

## 2026-06-14 — v0.4.0

### Added: SGLang engine support

[SGLang](https://github.com/sgl-project/sglang) is now a fourth engine option
alongside vLLM native, vLLM Docker, and llama.cpp. SGLang is a high-performance
serving framework with RadixAttention prefix caching and structured generation.

**What's included:**
- New flag catalog (`data/flags_sglang.json`) with 22 curated flags covering
  context length, memory management, quantization, reasoning parsers, tool call
  parsers, attention backends, structured output backends, and more.
- Engine builder (`engines/sglang.py`) — launches via
  `python -m sglang.launch_server --model-path <repo> --host 0.0.0.0`.
- Hardware detection — SGLang availability checked via `importlib.util.find_spec`.
- Full advisor support — traffic-light ratings, memory budget math, per-GPU
  load-headroom checks, KV-cache gate prediction, reasoning-parser family
  detection (with hyphen/underscore normalization for SGLang's `deepseek-r1`
  vs vLLM's `deepseek_r1`), and 4 computed presets (Safe, Max context, Max
  speed, Tight fit).
- Dashboard shows SGLang availability in the Engines panel.
- Settings page shows SGLang install instructions when not found.
- SGLang-specific failure patterns in the error translator.
- 12 new tests (4 engine builder + 8 advisor including DeepSeek-R1
  normalization). 121 tests total, all passing.

**SGLang-unique features exposed:**
- `--mem-fraction-static` (GPU memory fraction, like vLLM's gpu_memory_utilization)
- `--schedule-policy` with `lpm` (longest prefix match) for RadixAttention
- `--chunked-prefill-size` for long prompt handling
- `--kv-cache-dtype` with `fp8_e4m3`/`fp8_e5m2` options
- `--attention-backend` (flashinfer, triton, etc.)
- `--grammar-backend` (xgrammar, outlines, llguidance)
- `--reasoning-parser` with SGLang-specific parsers (qwen3-thinking, kimi, step3)
- `--tool-call-parser` with SGLang-specific parsers (qwen3_coder, deepseekv31, etc.)

### Known issue: SGLang TVM-FFI bug on NVIDIA Blackwell (RTX 5060 Ti / SM120)

Runtime testing of SGLang is blocked on Blackwell-generation GPUs by an
upstream bug in the `apache-tvm-ffi` dependency. During CUDA graph capture,
the TVM JIT compiler calls `_get_rocm_target()` in
`tvm_ffi/cpp/extension.py` — a function that should only run on AMD ROCm
hardware — and crashes because it cannot detect a ROCm architecture on an
NVIDIA card. This happens regardless of `--disable-cuda-graph` because
SGLang's piecewise CUDA graph path still triggers the same JIT kernel
(compiling `apply_rope_inplace`).

**Workaround:** Launch with `--attention-backend triton --sampling-backend
pytorch` to bypass the FlashInfer JIT compilation entirely. The app does
not apply this automatically — set these in the Extra raw flags field on
the Launch page.

**Background — the three known TVM-FFI bug categories:**

1. **JIT compilation / ninja failures.** During CUDA graph capture or the
   model's first forward pass, `tvm_ffi` triggers a runtime JIT step (via
   `ninja` and `nvcc`) to build C++/CUDA kernels. Stale cache artifacts in
   `~/.cache/tvm-ffi` can cause linker errors or header mismatches. Fix:
   clear the cache directory to force a clean rebuild, or use
   `--sampling-backend pytorch` to skip the JIT step.

2. **Conflicting `apache-tvm-ffi` versions.** Sub-dependencies like
   `flashinfer-python` or `xgrammar` pin specific pre-release versions
   (e.g. `apache-tvm-ffi==0.1.0b15`). A regular `pip install sglang` may
   fail to resolve because it prefers the stable `0.1.0` release over the
   pre-release. Fix: pass `--prerelease=allow` to `uv pip` or `pip`.

3. **C-ABI type code mismatches.** Breaking updates to `tvm_ffi` have
   changed internal type codes (e.g. 72 → 80). Loading an older compiled
   `.so` against a newer FFI version causes segfaults or dtype errors. Fix:
   ensure all ecosystem packages (`tilelang`, `flashinfer`, `sglang`) are
   updated and recompiled together in the same environment.

Upstream tracking: https://github.com/sgl-project/sglang/issues

## 2026-06-14 — v0.3.0

- Redesign the interface and add light/dark themes.

## 2026-06-13 — v0.2.0

- Add cache offload and automatic selection of a free server port.
- Protect settings and Docker credentials, improve modal/toast accessibility,
  and bound concurrent downloads.
- Add PATCH support for settings and make hardware caching thread-safe.

## 2026-06-11

- Add Open WebUI integration and connection configuration.
- Add text-only model advice and per-GPU weight/cache/headroom estimates.
- Refresh memory readings and translate common parser, memory and argument errors.
- Prefer configured accelerator builds and preserve explicit launch arguments.

## 2026-06-10 — Initial build

- Add the Python API, hardware detection, flag catalogs, model discovery/downloads,
  launch advice, engine adapters, server management and browser interface.
- Adapt model discovery and configuration concepts from the MIT-licensed vllm-cli.

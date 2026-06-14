# Plan: Add SGLang Engine Support

## Overview
Add SGLang as a fourth engine option alongside vLLM native, vLLM Docker, and llama.cpp.

## Files to Create
1. `src/local_llm_launcher/data/flags_sglang.json` — curated SGLang flag catalog
2. `src/local_llm_launcher/engines/sglang.py` — command builder

## Files to Modify
3. `src/local_llm_launcher/catalog.py` — add "sglang" to ENGINES tuple
4. `src/local_llm_launcher/hardware.py` — detect SGLang availability
5. `src/local_llm_launcher/advisor.py` — add `_advise_sglang()` + presets
6. `src/local_llm_launcher/registry.py` — import and wire sglang builder
7. `src/local_llm_launcher/api.py` — add sglang to engine routing
8. `src/local_llm_launcher/data/failures.json` — add SGLang-specific error patterns
9. `frontend/src/views/Launch.jsx` — add SGLang to ENGINE_LABELS and routing
10. `frontend/src/views/Dashboard.jsx` — show SGLang availability
11. `frontend/src/views/Settings.jsx` — SGLang install instructions
12. `CHANGELOG.md` — document the addition

## Tests to Write
13. `tests/test_engines.py` — SGLang command builder tests
14. `tests/test_advisor.py` — SGLang advisor tests

## SGLang Key Flags (curated selection)
- `--model-path` — model path (HF repo or local)
- `--host` / `--port` — HTTP server binding
- `--tensor-parallel-size` — multi-GPU splitting
- `--mem-fraction-static` — GPU memory fraction (like vLLM's gpu_memory_utilization)
- `--chunked-prefill-size` — long prompt handling
- `--max-prefill-tokens` — prefill batch limit
- `--dtype` — math precision
- `--quantization` — quantization method
- `--kv-cache-dtype` — KV cache compression
- `--trust-remote-code` — custom model code
- `--context-length` — max context override
- `--schedule-policy` — request scheduling
- `--reasoning-parser` — thinking model support
- `--tool-call-parser` — function calling
- `--served-model-name` — model name alias
- `--api-key` — authentication
- `--attention-backend` — attention kernel choice
- `--grammar-backend` — structured output backend
- `--enable-multimodal` — vision/audio support
- `--load-format` — weight loading format
- `--data-parallel-size` — DP parallelism
- `--device` — compute device
- `--random-seed` — reproducibility
- `--log-level` — logging verbosity

## SGLang Launch Command
```
python -m sglang.launch_server --model-path <repo_id> --host 0.0.0.0 --port 30000 [flags]
```

## Implementation Order
1. Flag catalog → Engine builder → Catalog update
2. Hardware detection → Advisor → Registry
3. API routing → Frontend → Failure patterns
4. Tests → Documentation

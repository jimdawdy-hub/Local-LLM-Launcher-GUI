# Plan: Local-LLM-Launcher-GUI

## Stack
- Backend: Python 3.10+, FastAPI + uvicorn, huggingface_hub, psutil. Tests: pytest.
- Frontend: Vite + React 18 + Tailwind. Built assets committed into the Python package.

## Layout
```
Local-LLM-Launcher-GUI/
├── README.md                    # with credit to Chen-zexi/vllm-cli
├── LICENSE                      # MIT
├── pyproject.toml               # console script: local-llm-launcher
├── src/local_llm_launcher/
│   ├── __init__.py / __main__.py
│   ├── app.py                   # FastAPI factory, static mount, browser open
│   ├── api.py                   # routes
│   ├── hardware.py              # GPU / Apple Silicon / RAM / disk / engine detection
│   ├── advisor.py               # traffic-light engine (pure functions)
│   ├── catalog.py               # flag catalog loader + presets
│   ├── registry.py              # persistent server registry (~/.local-llm-launcher)
│   ├── discovery.py             # installed models (HF cache + GGUF dirs)
│   ├── downloads.py             # HF search + download manager (threads + progress)
│   ├── chatproxy.py             # proxy chat to OpenAI endpoint
│   ├── engines/
│   │   ├── base.py              # LocalServer lifecycle (adapted from vllm-cli)
│   │   ├── vllm_native.py / vllm_docker.py / llamacpp.py   # command builders
│   └── data/
│       ├── flags_vllm.json      # curated flags + plain-English text + rules hints
│       ├── flags_llamacpp.json
│       └── failures.json        # log-pattern → friendly error translation
│   └── static/                  # built frontend (committed)
├── frontend/                    # React source
└── tests/                       # pytest
```

## Build order (TDD on core logic)
1. Scaffold: pyproject, package, git, venv, pytest runs.
2. `hardware.py` — tests with faked `nvidia-smi` output / platform; then impl.
3. `discovery.py` — tests with temp fake HF cache; then impl.
4. Flag catalogs (data) + `catalog.py` loader + tests for schema completeness.
5. `advisor.py` — the rules engine; thorough tests (fit math, each rule, edge cases:
   no GPU, Apple Silicon, multi-GPU, GGUF-on-vLLM...).
6. Command builders (3 engines) — tests assert exact argv for given configs.
7. `engines/base.py` lifecycle + `registry.py` — tests with a dummy long-running
   process (sleep) instead of real vLLM.
8. `downloads.py` (HF search/download; download tested against tiny file or mocked).
9. `api.py` + `app.py` — FastAPI TestClient smoke tests for all routes.
10. Frontend (invoke frontend-design skill first): views = Dashboard, Models,
    Launch, Servers. Polling for logs/progress (no websockets in v1 — simpler).
11. Build frontend → copy dist to package static → end-to-end smoke (start app,
    curl API, check page serves).
12. README + About credit + screenshots placeholder; final commit.

## Key risks & mitigations
- Python 3.14 is default `python3` but pip is 3.12 → create venv with a pinned
  working interpreter; document.
- vLLM startup is slow/silent → UI expectations + failure-pattern translation.
- Frontend bug surface → keep to 4 views, no router lib (tab state), TanStack-free,
  plain fetch + polling.

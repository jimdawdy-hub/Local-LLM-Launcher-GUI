# Contributing

Thanks for considering a contribution! This project exists to take some of
the pain out of running local LLMs, and there are a *lot* of GPUs, model
families, and engine versions out there that the original author couldn't
possibly test alone. If something didn't work right on your hardware, that's
useful information even if you don't write a line of code.

## Vibe coders welcome

If you got here by asking an AI assistant to "look at this app and fix X" —
**you're in exactly the right place.** This entire project was built through
AI-assisted development (see [ABOUT.md](ABOUT.md)), and that's not an
exception to how it's maintained, it's the expected workflow. You don't need
to be a Python or React expert to contribute meaningfully:

- You understand your *own hardware and the error you hit* better than anyone.
- An AI assistant (Claude Code, or whatever you use) can read this codebase
  and the docs in this folder just as well as it read vllm-cli's.
- A good bug report — the exact model, the exact settings, and the **full log
  file** from `~/.local-llm-launcher/logs/` — is often more valuable than a
  hasty PR.

If you do open a PR with AI-assisted code, that's great — just make sure:
1. **The tests pass** (`pytest` from the repo root) and the frontend builds
   (`cd frontend && npm run build`).
2. **You actually ran it** against a real model on real hardware, if the
   change touches the advisor, command builders, or server lifecycle. This
   project's whole premise is "tested against reality, not just plausible,"
   and several of its bug fixes only surfaced by launching real models on
   real GPUs (see [CHANGELOG.md](CHANGELOG.md)).
3. **You explain *why***, not just what — especially for advisor rules. A
   rule like "warn if `gpu_memory_utilization > 0.90` on consumer cards"
   needs the *reason* (the pre-load headroom check fails) so the next person
   — human or AI — knows when the rule still applies and when it doesn't.

## Ways to contribute (no code required)

- **New model family quirks.** If a model family needs a specific
  `--reasoning-parser`, `--tool-call-parser`, chat template flag, or has a
  known quantization gotcha, that's a one- or two-line addition to
  `advisor.py`'s family-detection rules or `data/failures.json`.
- **New failure patterns.** Hit a cryptic error that this app didn't
  translate? Add the log snippet and a plain-English explanation to
  `data/failures.json` — see existing entries for the format.
- **Hardware reports.** Different GPU (AMD, older NVIDIA, more VRAM, Apple
  Silicon variants), different OS — even just "I ran this on X and it worked
  / the numbers were off by Y" helps tune the memory-budget constants in
  `advisor.py`.
- **Flag catalog gaps.** Missing a flag you need? `data/flags_vllm.json` and
  `data/flags_llamacpp.json` are plain JSON — add an entry with a `label`,
  `help` (plain English!), `type`, `category`, and `cli_flag`.

## Ways to contribute (code)

### Setup

```bash
git clone <this-repo>
cd Local-LLM-Launcher-GUI
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cd frontend
npm install
```

### Running it during development

```bash
# Terminal 1: backend
local-llm-launcher --no-browser

# Terminal 2: frontend dev server (proxies /api to the backend)
cd frontend && npm run dev
```

### Before opening a PR

```bash
pytest                          # backend tests
cd frontend && npm run build    # frontend must build cleanly
```

### Where things live

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map. The short
version:

- **New vLLM/llama.cpp flag?** → `src/local_llm_launcher/data/flags_*.json`
- **New advisor rule?** → `src/local_llm_launcher/advisor.py` (and a test in
  `tests/test_advisor.py` — this file has the most test coverage for a
  reason)
- **New failure translation?** → `src/local_llm_launcher/data/failures.json`
- **UI changes?** → `frontend/src/views/*.jsx`, styled via
  `frontend/src/theme.css`

### Code style

- Backend: plain Python, type hints where they help, no framework beyond
  FastAPI/Pydantic. Keep rules in `advisor.py` as small, named, testable
  pieces — each one should map to a real failure mode (ideally one you've
  seen).
- Frontend: functional React components, no global state library — props and
  `useState`/`useEffect` are sufficient at this size. Keep the LED/badge/gauge
  components in `components.jsx` shared rather than reimplemented per view.
- Don't add a flag, preset, or rule "because the engine supports it" — add it
  because it solves a real problem someone will hit. The flag catalogs are
  deliberately curated subsets, not full mirrors of `vllm serve --help` /
  `llama-server --help`.

## Reporting issues

Please include:
1. Your hardware (GPU(s) + VRAM, or Apple Silicon model + memory, or CPU/RAM).
2. The model (HuggingFace repo ID) and which engine you selected.
3. The exact settings shown on the Launch screen (a screenshot is great).
4. The full log file from `~/.local-llm-launcher/logs/<engine>_<model>_<timestamp>.log`
   — not just the last few lines, the whole thing. The launcher header line
   at the top shows the exact command that was run.

That's usually enough for someone (human or AI) to reproduce the issue or add
a new advisor rule / failure translation for it.

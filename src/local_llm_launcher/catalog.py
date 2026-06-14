"""Flag catalog loading. Curated flag lists live in data/flags_*.json.

Catalog concept adapted from vllm-cli's argument schema
(https://github.com/Chen-zexi/vllm-cli by Chen-zexi, MIT license).
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict

ENGINES = ("vllm", "llamacpp", "sglang")


@lru_cache(maxsize=None)
def load_catalog(engine: str) -> Dict[str, Any]:
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Expected one of {ENGINES}.")
    path = resources.files("local_llm_launcher.data") / f"flags_{engine}.json"
    return json.loads(path.read_text())


def defaults(engine: str) -> Dict[str, Any]:
    """Flat {key: default} for all flags that have a non-None default."""
    return {
        f["key"]: f["default"]
        for f in load_catalog(engine)["flags"]
        if f.get("default") is not None
    }

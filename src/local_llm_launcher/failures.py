"""Translate raw engine log output into plain-English failure explanations."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import List, Optional


@lru_cache(maxsize=1)
def _patterns():
    path = resources.files("local_llm_launcher.data") / "failures.json"
    return json.loads(path.read_text())["patterns"]


def translate(log_lines: List[str]) -> Optional[str]:
    """Return a friendly explanation for the first known failure pattern found,
    scanning the most recent lines first."""
    for line in reversed(log_lines):
        for p in _patterns():
            if p["match"] in line:
                return p["message"]
    return None

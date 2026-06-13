"""App settings persisted in ~/.local-llm-launcher/settings.json (chmod 600 —
it may hold a Hugging Face token)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "hf_token": None,
    "gguf_folders": [],
    "llamacpp_path": None,
}


class Settings:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        from .registry import APP_DIR
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.path = self.app_dir / "settings.json"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = dict(DEFAULTS)
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                stored = json.loads(self.path.read_text())
                if isinstance(stored, dict):
                    self.data.update(stored)
            except (OSError, json.JSONDecodeError):
                pass

    def save(self) -> None:
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(self.data, indent=2))

    def update(self, changes: Dict[str, Any]) -> None:
        for key in DEFAULTS:
            if key in changes:
                self.data[key] = changes[key]
        self.save()

    def public(self) -> Dict[str, Any]:
        """Settings safe to send to the browser — token masked."""
        out = dict(self.data)
        if out.get("hf_token"):
            out["hf_token"] = "********"
            out["hf_token_set"] = True
        else:
            out["hf_token_set"] = False
        return out

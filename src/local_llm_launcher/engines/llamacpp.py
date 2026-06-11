"""Command builder for llama.cpp's llama-server."""
from __future__ import annotations

import os
from typing import Any, Dict

from ._args import build_args_and_env


def _pick_gguf_path(model: Dict[str, Any], config: Dict[str, Any]) -> str:
    files = model.get("gguf_files") or []
    wanted = config.get("gguf_file")
    if wanted:
        for f in files:
            if f["filename"] == wanted:
                return f["path"]
    if files:
        return files[0]["path"]
    return model["path"]


def build(model: Dict[str, Any], config: Dict[str, Any], binary: str = "llama-server") -> Dict[str, Any]:
    cfg = {k: v for k, v in config.items() if k != "gguf_file"}
    flags, env, extra = build_args_and_env("llamacpp", cfg)
    port = int(config.get("port", 8080))
    # Prebuilt llama.cpp releases ship their shared libraries next to the binary;
    # without LD_LIBRARY_PATH pointing there, the server can't start.
    if os.sep in binary:
        bindir = os.path.dirname(os.path.abspath(binary))
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bindir}:{existing}" if existing else bindir
    argv = [binary, "-m", _pick_gguf_path(model, config)] + flags + extra
    return {
        "argv": argv,
        "env": env,
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
    }

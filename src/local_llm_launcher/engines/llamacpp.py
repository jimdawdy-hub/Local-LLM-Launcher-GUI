"""Command builder for llama.cpp's llama-server."""
from __future__ import annotations

import os
from typing import Any, Dict

from ._args import build_args_and_env
from .mtp import is_head
from .placement import validate, wrap
from .. import hardware


def is_mmproj(name: str) -> bool:
    """llama.cpp's image-projector files ("mmproj-*.gguf"); loaded only through --mmproj."""
    return "mmproj" in name.lower()


def pick_gguf_path(model: Dict[str, Any], config: Dict[str, Any]) -> str:
    files = model.get("gguf_files") or []
    wanted = config.get("gguf_file")
    if wanted:
        for f in files:
            if f["filename"] == wanted:
                return f["path"]
    # A separate MTP head or image-projector (mmproj) file is never the model itself.
    models = [f for f in files if not is_head(f["filename"]) and not is_mmproj(f["filename"])]
    if files:
        return (models or files)[0]["path"]
    return model["path"]


def build(model: Dict[str, Any], config: Dict[str, Any], binary: str = "llama-server") -> Dict[str, Any]:
    validate('llamacpp', config)
    cfg = {k: v for k, v in config.items() if k != "gguf_file"}
    loading = []
    if (cfg.get("no_mmap") or cfg.get("mlock")) and hardware.llama_capabilities(binary)["load_mode"]:
        no_mmap, mlock = cfg.pop("no_mmap", False), cfg.pop("mlock", False)
        loading = ["--load-mode", "mlock" if no_mmap and mlock else "none" if no_mmap else "mmap+mlock"]
    flags, env, extra = build_args_and_env("llamacpp", cfg)
    port = int(config.get("port", 8080))
    host = config.get("host", "127.0.0.1")
    # Prebuilt llama.cpp releases ship their shared libraries next to the binary;
    # without LD_LIBRARY_PATH pointing there, the server can't start.
    if os.sep in binary:
        bindir = os.path.dirname(os.path.abspath(binary))
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bindir}:{existing}" if existing else bindir
    argv = [binary, "-m", pick_gguf_path(model, config)]
    if host:
        argv.extend(["--host", host])
    argv += flags + loading + list(config.get("_mtp_args") or ()) + extra
    return {
        "argv": wrap(argv, config),
        "env": env,
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
    }

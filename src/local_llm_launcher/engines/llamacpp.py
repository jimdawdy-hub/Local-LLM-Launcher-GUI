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


def unsupported_server_options(config: Dict[str, Any], binary: str) -> Dict[str, str]:
    selected = [key for key in ('system_prompt', 'system_prompt_file', 'color')
                if config.get(key) not in (None, '')]
    if not selected:
        return {}
    support = hardware.llama_capabilities(binary).get('server_options', {})
    messages = {}
    for key in selected:
        if support.get(key) is False:
            message = f"The selected llama-server does not accept --{key.replace('_', '-')}. Leave this launch setting empty."
            if key.startswith('system_prompt'):
                message += ' Set system instructions in your chat client.'
            messages[key] = message
    return messages


def build(model: Dict[str, Any], config: Dict[str, Any], binary: str = "llama-server") -> Dict[str, Any]:
    if "context_shift" in config and "ctx_shift" not in config:
        config = {**config, "ctx_shift": config["context_shift"]}
    validate('llamacpp', config)
    unsupported = unsupported_server_options(config, binary)
    if unsupported:
        raise ValueError(' '.join(unsupported.values()))
    if config.get("cuda_allreduce") not in (None, "nccl", "internal"):
        raise ValueError("CUDA AllReduce must be automatic, nccl, or internal")
    cfg = {k: v for k, v in config.items() if k not in ("gguf_file", "host", "context_shift")}
    loading = []
    if (cfg.get("no_mmap") or cfg.get("mlock")) and hardware.llama_capabilities(binary)["load_mode"]:
        no_mmap, mlock = cfg.pop("no_mmap", False), cfg.pop("mlock", False)
        loading = ["--load-mode", "mlock" if no_mmap and mlock else "none" if no_mmap else "mmap+mlock"]
    flags, env, extra = build_args_and_env("llamacpp", cfg)
    # Match nvidia-smi's PCI bus ordering so CUDA0, CUDA1, ... correspond to GPU 0, GPU 1, ...
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    port = int(config.get("port", 8080))
    raw_host = config.get("host")
    host = "127.0.0.1" if raw_host is None else str(raw_host).strip()
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

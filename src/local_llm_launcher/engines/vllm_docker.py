"""Command builder for vLLM via the official Docker image (vllm/vllm-openai).

The container always serves on port 8000 internally; we map the requested host
port onto it. `--ipc=host` is required or multiprocess workers can hang.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from ._args import build_args_and_env

IMAGE = "vllm/vllm-openai:latest"
HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def build(model: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Build vLLM flags with the *container* port, not the host port.
    container_cfg = dict(config)
    host_port = int(config.get("port", 8000))
    container_cfg["port"] = 8000
    flags, env, extra = build_args_and_env("vllm", container_cfg)

    container_name = f"llml-{uuid.uuid4().hex[:8]}"
    gpus = '"device=' + str(config["device_ids"]) + '"' if config.get("device_ids") else "all"

    argv = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", gpus,
        "-p", f"{host_port}:8000",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        "--ipc=host",
    ]
    for var, value in env.items():
        if var == "CUDA_VISIBLE_DEVICES":
            continue  # GPU selection handled by --gpus above
        argv.extend(["-e", f"{var}={value}"])
    argv.append(IMAGE)
    argv.extend(["--model", model["repo_id"], "--host", "0.0.0.0"] + flags + extra)

    return {
        "argv": argv,
        "env": {},
        "port": host_port,
        "health_url": f"http://127.0.0.1:{host_port}/health",
        "container_name": container_name,
    }

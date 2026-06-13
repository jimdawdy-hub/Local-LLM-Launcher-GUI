"""Command builder for vLLM via the official Docker image (vllm/vllm-openai).

The container always serves on port 8000 internally; we map the requested host
port onto it. `--ipc=host` is required or multiprocess workers can hang.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict

from ._args import build_args_and_env

IMAGE = "vllm/vllm-openai:latest"
HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def _write_env_file(env: Dict[str, str]) -> str:
    """Write env vars to a temp file for Docker --env-file (avoids ps exposure)."""
    fd, path = tempfile.mkstemp(prefix="llml-env-", suffix=".env")
    with os.fdopen(fd, "w") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")
    return path


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

    # Pass secrets via --env-file instead of -e to avoid exposing them in ps output.
    env_file = None
    non_secret_env = {}
    for var, value in env.items():
        if var == "CUDA_VISIBLE_DEVICES":
            continue  # GPU selection handled by --gpus above
        if "TOKEN" in var.upper() or "KEY" in var.upper() or "SECRET" in var.upper():
            if env_file is None:
                env_file = _write_env_file({})
            with open(env_file, "a") as f:
                f.write(f"{var}={value}\n")
        else:
            non_secret_env[var] = value
    if env_file:
        argv.extend(["--env-file", env_file])
    for var, value in non_secret_env.items():
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

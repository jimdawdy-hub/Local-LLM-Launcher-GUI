"""Command builder for SGLang (pip-installed).

SGLang is a high-performance serving framework with RadixAttention prefix
caching and structured generation.  Launch command:
    python -m sglang.launch_server --model-path <repo_id> --host 0.0.0.0 --port 30000
"""
from __future__ import annotations

from typing import Any, Dict

from ._args import build_args_and_env


def build(model: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    flags, env, extra = build_args_and_env("sglang", config)
    port = int(config.get("port", 30000))
    argv = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model["repo_id"],
        "--host", "0.0.0.0",
    ] + flags + extra
    return {
        "argv": argv,
        "env": env,
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
    }

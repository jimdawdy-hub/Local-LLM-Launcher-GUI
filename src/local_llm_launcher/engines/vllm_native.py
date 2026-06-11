"""Command builder for native (pip-installed) vLLM.

Adapted from vllm-cli's server command construction
(https://github.com/Chen-zexi/vllm-cli by Chen-zexi, MIT license).
"""
from __future__ import annotations

from typing import Any, Dict

from ._args import build_args_and_env


def build(model: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    flags, env, extra = build_args_and_env("vllm", config)
    port = int(config.get("port", 8000))
    argv = ["vllm", "serve", model["repo_id"]] + flags + extra
    return {
        "argv": argv,
        "env": env,
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
    }

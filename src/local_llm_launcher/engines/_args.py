"""Shared helper: turn a config dict into CLI args using the flag catalog."""
from __future__ import annotations

import shlex
from typing import Any, Dict, List, Tuple

from .. import catalog


def build_args_and_env(engine: str, config: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Returns (argv_flags, env, extra_args) for the given engine config.

    Flags with `"flag": null` in the catalog are routed to environment variables
    (when `env_var` is set) or returned as parsed extra args (`extra_args`).
    """
    spec_by_key = {f["key"]: f for f in catalog.load_catalog(engine)["flags"]}
    merged = catalog.defaults(engine)
    merged.update({k: v for k, v in config.items() if v is not None})

    argv: List[str] = []
    env: Dict[str, str] = {}
    extra: List[str] = []

    for key, value in merged.items():
        spec = spec_by_key.get(key)
        if spec is None or value is None:
            continue
        cli_flag = spec.get("flag")
        if cli_flag is None:
            if key == "extra_args":
                try:
                    extra = shlex.split(str(value))
                except ValueError:
                    extra = []
            elif spec.get("env_var"):
                env[spec["env_var"]] = str(value)
            continue
        if spec["type"] == "bool":
            if value:
                argv.append(cli_flag)
        else:
            argv.extend([cli_flag, str(value)])

    return argv, env, extra

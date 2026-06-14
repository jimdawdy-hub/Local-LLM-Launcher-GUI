"""Server manager: owns running servers, persists them across GUI restarts."""
from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engines import llamacpp, sglang, vllm_docker, vllm_native
from .engines.base import LocalServer

APP_DIR = Path.home() / ".local-llm-launcher"


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(preferred: int, max_tries: int = 100) -> int:
    """Return `preferred` if free, otherwise the next free port above it."""
    port = preferred
    for _ in range(max_tries):
        if not port_in_use(port):
            return port
        port += 1
    raise RuntimeError(
        f"No free port found starting from {preferred} (checked {max_tries} ports)."
    )


class ServerManager:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.log_dir = self.app_dir / "logs"
        self.state_file = self.app_dir / "servers.json"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.servers: Dict[str, LocalServer] = {}
        self._reload()

    # -------------------------------------------------------------- persistence

    def _reload(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            records = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for rec in records:
            try:
                srv = LocalServer.from_record(rec, self.log_dir)
            except (KeyError, TypeError):
                continue
            # Keep recently-dead servers too so their logs/errors stay visible.
            self.servers[srv.server_id] = srv

    def _save(self) -> None:
        records = [s.to_record() for s in self.servers.values()]
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(self.state_file)

    # ------------------------------------------------------------------- launch

    def build_spec(self, engine_mode: str, model: Dict[str, Any], config: Dict[str, Any],
                   llamacpp_binary: Optional[str] = None) -> Dict[str, Any]:
        if engine_mode == "vllm-native":
            return vllm_native.build(model, config)
        if engine_mode == "vllm-docker":
            return vllm_docker.build(model, config)
        if engine_mode == "llamacpp":
            return llamacpp.build(model, config, binary=llamacpp_binary or "llama-server")
        if engine_mode == "sglang":
            return sglang.build(model, config)
        raise ValueError(f"Unknown engine mode '{engine_mode}'")

    def launch(self, engine_mode: str, model: Dict[str, Any], config: Dict[str, Any],
               llamacpp_binary: Optional[str] = None) -> LocalServer:
        spec = self.build_spec(engine_mode, model, config, llamacpp_binary)
        if port_in_use(spec["port"]):
            spec["port"] = find_free_port(spec["port"])
        srv = LocalServer(
            server_id=uuid.uuid4().hex[:12],
            engine=engine_mode,
            model_label=model["repo_id"],
            port=spec["port"],
            argv=spec["argv"],
            env=spec.get("env") or {},
            log_dir=self.log_dir,
            container_name=spec.get("container_name"),
        )
        if not srv.start():
            raise RuntimeError("The server process failed to start. Check the logs for details.")
        self.servers[srv.server_id] = srv
        self._save()
        return srv

    # ------------------------------------------------------------------ queries

    def list(self) -> List[Dict[str, Any]]:
        return [s.status() for s in self.servers.values()]

    def get(self, server_id: str) -> Optional[LocalServer]:
        return self.servers.get(server_id)

    def stop(self, server_id: str) -> bool:
        srv = self.servers.get(server_id)
        if not srv:
            return False
        ok = srv.stop()
        self._save()
        return ok

    def remove(self, server_id: str) -> bool:
        srv = self.servers.pop(server_id, None)
        if not srv:
            return False
        if srv.is_running():
            srv.stop()
        self._save()
        return True

    def stop_all(self) -> None:
        for srv in self.servers.values():
            if srv.is_running():
                srv.stop()
        self._save()

"""Server manager: owns running servers, persists them across GUI restarts."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engines import llamacpp, mtp, vllm_backends, vllm_docker, vllm_native
from .engines.base import LocalServer
from . import catalog

APP_DIR = Path(os.environ.get("LOCAL_LLM_LAUNCHER_HOME") or Path.home() / ".local-llm-launcher").expanduser()


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(preferred: int, max_tries: int = 100,
                   reserved: Optional[set[int]] = None) -> int:
    """Return `preferred` if free, otherwise the next free port above it."""
    port = preferred
    for _ in range(max_tries):
        if port not in (reserved or ()) and not port_in_use(port):
            return port
        port += 1
    raise RuntimeError(
        f"No free port found starting from {preferred} (checked {max_tries} ports)."
    )


def _watchdog_minutes(engine_mode: str, config: Dict[str, Any]) -> int:
    """The freeze watchdog covers llama.cpp only (vLLM has no /slots)."""
    if engine_mode != "llamacpp":
        return 0
    value = config.get("watchdog_minutes")
    if value is None:
        value = catalog.defaults("llamacpp").get("watchdog_minutes", 0)
    return max(int(value), 0)


class ServerManager:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.log_dir = self.app_dir / "logs"
        self.state_file = self.app_dir / "servers.json"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lifecycle_locks: Dict[str, threading.Lock] = {}
        self._stopping: set[str] = set()
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
        with self._lock:
            records = [s.to_record() for s in self.servers.values()]
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(records, indent=2))
            tmp.replace(self.state_file)

    # ------------------------------------------------------------------- launch

    def build_spec(self, engine_mode: str, model: Dict[str, Any], config: Dict[str, Any],
                   llamacpp_binary: Optional[str] = None, vllm_binary: Optional[str] = None) -> Dict[str, Any]:
        if engine_mode == "vllm-native":
            return vllm_native.build(model, config, binary=vllm_binary or "vllm")
        if engine_mode == "vllm-docker":
            return vllm_docker.build(model, config)
        if engine_mode == "llamacpp":
            return llamacpp.build(model, config, binary=llamacpp_binary or "llama-server")
        raise ValueError(f"Unknown engine mode '{engine_mode}'")

    def launch(self, engine_mode: str, model: Dict[str, Any], config: Dict[str, Any],
               llamacpp_binary: Optional[str] = None, vllm_binary: Optional[str] = None,
               hardware: Optional[Dict[str, Any]] = None) -> LocalServer:
        # Validate before taking the lock: probing the vLLM runtime can take many
        # seconds, and status polling, logs and stop all need this lock.
        vllm_backends.validate(engine_mode, model, config, hardware, vllm_binary or "vllm")
        config = dict(config)
        if config.get("use_mtp"):
            decision = mtp.resolve(engine_mode, model, config, vllm_binary=vllm_binary or "vllm",
                                   llama_binary=llamacpp_binary or "llama-server")
            if decision["level"] == "red":
                raise ValueError(decision["message"])
            config["_mtp_args"] = decision["args"]
        with self._lock:
            # Reserve the port before building/spawning: a new process may not
            # listen yet when another launch arrives.
            default_port = catalog.defaults("llamacpp")["port"] if engine_mode == "llamacpp" \
                else catalog.defaults("vllm")["port"]
            reserved = {s.port for s in self.servers.values()
                        if s.server_id in self._stopping or s.is_running()}
            config["port"] = find_free_port(int(config.get("port", default_port)),
                                            reserved=reserved)
            spec = self.build_spec(engine_mode, model, config, llamacpp_binary, vllm_binary)
            srv = LocalServer(
                server_id=uuid.uuid4().hex[:12],
                engine=engine_mode,
                model_label=model["repo_id"],
                port=spec["port"],
                argv=spec["argv"],
                env=spec.get("env") or {},
                log_dir=self.log_dir,
                container_name=spec.get("container_name"),
                env_file=spec.get("env_file"),
                watchdog_minutes=_watchdog_minutes(engine_mode, config),
            )
            started = srv.start()
            if not started:
                srv._cleanup_env_file()
            self.servers[srv.server_id] = srv
            self._save()
            if not started:
                raise RuntimeError("The server process failed to start. Check the logs for details.")
        return srv

    # ------------------------------------------------------------------ queries

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.status() for s in self.servers.values()]

    def get(self, server_id: str) -> Optional[LocalServer]:
        with self._lock:
            return self.servers.get(server_id)

    def _shutdown(self, server_id: str, *, remove: bool = False) -> bool:
        with self._lock:
            srv = self.servers.get(server_id)
            if not srv:
                return False
            lifecycle_lock = self._lifecycle_locks.setdefault(server_id, threading.Lock())

        # Waiting for a process (or another shutdown of this server) must not
        # block queries, launches, or lifecycle operations on other servers.
        with lifecycle_lock:
            with self._lock:
                if self.servers.get(server_id) is not srv:
                    return False  # a preceding remove already finished
                self._stopping.add(server_id)
            try:
                if remove and not srv.is_running():
                    srv._cleanup_env_file()
                    ok = True
                else:
                    ok = srv.stop()
                if remove:
                    ok = ok and not srv.is_running()
                    if ok:
                        with self._lock:
                            del self.servers[server_id]
                            del self._lifecycle_locks[server_id]
                return ok
            finally:
                with self._lock:
                    self._stopping.discard(server_id)
                    self._save()

    def stop(self, server_id: str) -> bool:
        return self._shutdown(server_id)

    def remove(self, server_id: str) -> bool:
        return self._shutdown(server_id, remove=True)

    # ----------------------------------------------------------------- watchdog

    def watchdog_tick(self, now: Optional[float] = None) -> List[str]:
        """Stop servers whose work loop has been frozen for their set minutes.

        Returns the ids stopped. A frozen llama-server keeps accepting requests and
        never answers them, and nothing else ever stops it.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            watched = [s for s in self.servers.values() if s.watchdog_minutes > 0]
        stopped = []
        for srv in watched:
            if not srv.is_running() or srv.probe() != "stuck":
                srv.stuck_since = None
                continue
            if srv.stuck_since is None:
                srv.stuck_since = now
            elif now - srv.stuck_since >= srv.watchdog_minutes * 60:
                minutes = f"{srv.watchdog_minutes} minute{'s' if srv.watchdog_minutes != 1 else ''}"
                srv.note(f"watchdog: the server accepted requests but did no work for "
                         f"{minutes}, so the launcher stopped it.")
                srv.stuck_since = None
                if self.stop(srv.server_id):
                    stopped.append(srv.server_id)
        return stopped

    def start_watchdog(self, interval: float = 30.0) -> threading.Thread:
        def loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    self.watchdog_tick()
                except Exception:  # noqa: BLE001 - the watchdog must outlive one bad probe
                    pass

        thread = threading.Thread(target=loop, name="server-watchdog", daemon=True)
        thread.start()
        return thread

    def stop_all(self) -> None:
        with self._lock:
            server_ids = list(self.servers)
        for server_id in server_ids:
            self.stop(server_id)

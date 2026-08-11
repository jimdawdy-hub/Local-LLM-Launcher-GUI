"""Server process lifecycle: launch, monitor, tail logs, stop.

Adapted from vllm-cli's VLLMServer (https://github.com/Chen-zexi/vllm-cli by
Chen-zexi, MIT license), simplified: logs go straight to a file and are tailed
from disk, which survives GUI restarts.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil


class LocalServer:
    def __init__(
        self,
        server_id: str,
        engine: str,
        model_label: str,
        port: int,
        argv: List[str],
        env: Dict[str, str],
        log_dir: Path,
        container_name: Optional[str] = None,
        pid: Optional[int] = None,
        started_at: Optional[str] = None,
        log_path: Optional[str] = None,
        pid_create_time: Optional[float] = None,
        env_file: Optional[str] = None,
    ) -> None:
        self.server_id = server_id
        self.engine = engine
        self.model_label = model_label
        self.port = port
        self.argv = argv
        self.env = env
        self.container_name = container_name
        self.process: Optional[subprocess.Popen] = None
        self.pid = pid
        self.pid_create_time = pid_create_time
        self.started_at = started_at
        self.env_file = Path(env_file) if env_file else None
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_path:
            self.log_path = Path(log_path)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = model_label.replace("/", "_")
            self.log_path = log_dir / f"{engine}_{safe}_{stamp}.log"

    # ------------------------------------------------------------------ control

    def start(self) -> bool:
        if self.is_running():
            return False
        env = os.environ.copy()
        env.update(self.env)
        try:
            with open(self.log_path, "ab") as log_file:
                log_file.write(
                    f"[launcher] {datetime.now().isoformat()} starting: {' '.join(self.argv)}\n".encode()
                )
            log_handle = open(self.log_path, "ab")
            self.process = subprocess.Popen(
                self.argv,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,  # isolate from our Ctrl+C / signals
            )
            log_handle.close()  # child keeps its own descriptor
            self.pid = self.process.pid
            try:
                self.pid_create_time = psutil.Process(self.pid).create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.pid_create_time = None
            self.started_at = datetime.now(timezone.utc).isoformat()
            return True
        except (OSError, subprocess.SubprocessError) as e:
            with open(self.log_path, "ab") as log_file:
                log_file.write(f"[launcher] failed to start: {e}\n".encode())
            return False

    def stop(self, timeout: float = 15.0) -> bool:
        # Docker containers need `docker stop`; killing the client can orphan them.
        if self.container_name:
            try:
                subprocess.run(["docker", "stop", "-t", "10", self.container_name],
                               capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
        pid = self.process.pid if self.process else self.pid
        if self.process is None and self.pid_create_time is not None and not self._pid_matches():
            self._cleanup_env_file()
            return True  # pid was recycled; never kill an innocent process
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            deadline = time.time() + timeout
            while time.time() < deadline:
                if not self.is_running():
                    self._cleanup_env_file()
                    return True
                time.sleep(0.2)
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            time.sleep(0.3)
        if not self.is_running():
            self._cleanup_env_file()
        return not self.is_running()

    def _cleanup_env_file(self) -> None:
        """Remove the temp env file (HF token for docker) once the server is gone."""
        if self.env_file:
            try:
                self.env_file.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------- status

    def is_running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        if self.pid:
            if self.pid_create_time is not None and not self._pid_matches():
                return False  # pid recycled by an unrelated process
            try:
                os.kill(self.pid, 0)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                return False
        return False

    def _pid_matches(self) -> bool:
        """True only when the live process at self.pid has our stored start time."""
        try:
            return abs(psutil.Process(self.pid).create_time() - self.pid_create_time) < 1.0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def exit_code(self) -> Optional[int]:
        if self.process is not None:
            return self.process.poll()
        return None

    def status(self) -> Dict[str, Any]:
        running = self.is_running()
        return {
            "id": self.server_id,
            "engine": self.engine,
            "model": self.model_label,
            "port": self.port,
            "running": running,
            "pid": self.pid,
            "started_at": self.started_at,
            "exit_code": self.exit_code(),
            "log_path": str(self.log_path),
            "endpoint": f"http://127.0.0.1:{self.port}/v1",
        }

    def health(self) -> bool:
        try:
            r = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def tail_logs(self, n: int = 100) -> List[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 256 * 1024))
                data = f.read().decode("utf-8", errors="replace")
            lines = data.splitlines()
            return lines[-n:]
        except OSError:
            return []

    # -------------------------------------------------------------- persistence

    def to_record(self) -> Dict[str, Any]:
        return {
            "server_id": self.server_id,
            "engine": self.engine,
            "model_label": self.model_label,
            "port": self.port,
            "argv": self.argv,
            "env_keys": sorted(self.env.keys()),  # never persist env values (tokens)
            "container_name": self.container_name,
            "pid": self.pid,
            "pid_create_time": self.pid_create_time,
            "started_at": self.started_at,
            "log_path": str(self.log_path),
            "env_file": str(self.env_file) if self.env_file else None,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any], log_dir: Path) -> "LocalServer":
        return cls(
            server_id=record["server_id"],
            engine=record["engine"],
            model_label=record["model_label"],
            port=record["port"],
            argv=record.get("argv") or [],
            env={},
            log_dir=log_dir,
            container_name=record.get("container_name"),
            pid=record.get("pid"),
            started_at=record.get("started_at"),
            log_path=record.get("log_path"),
            pid_create_time=record.get("pid_create_time"),
            env_file=record.get("env_file"),
        )

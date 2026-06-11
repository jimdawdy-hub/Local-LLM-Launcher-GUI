"""Open WebUI integration: detect, launch, and track a local Open WebUI server.

Open WebUI (https://github.com/open-webui/open-webui) is a full-featured chat
interface that talks to OpenAI-compatible endpoints — exactly what the model
servers this app launches expose. We manage it with the same LocalServer
lifecycle used for model engines, persisted separately so it survives GUI
restarts.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .engines.base import LocalServer
from .registry import APP_DIR, port_in_use

# Open WebUI's own default is 8080, which collides with llama.cpp here, so we
# default to 3000 (the port its Docker examples use).
DEFAULT_PORT = 3000
INSTALL_COMMAND = "pip install open-webui"
# Model servers launched here don't require an API key, but Open WebUI still
# sends an Authorization header, so we give it a harmless placeholder.
PLACEHOLDER_KEY = "sk-local"


def _open_browser_when_ready(url: str, health_url: str, timeout: float = 240.0) -> None:
    """Poll Open WebUI's /health, then open the browser. Opens anyway on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(health_url, timeout=3.0).status_code == 200:
                webbrowser.open(url)
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    webbrowser.open(url)  # last resort — let the user see whatever state it's in


class OpenWebUIManager:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.log_dir = self.app_dir / "logs"
        self.state_file = self.app_dir / "openwebui.json"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.server: Optional[LocalServer] = None
        self._reload()

    # ------------------------------------------------------------- detection

    def binary(self) -> Optional[str]:
        return shutil.which("open-webui")

    def installed(self) -> bool:
        return self.binary() is not None

    # ----------------------------------------------------------- persistence

    def _reload(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            rec = json.loads(self.state_file.read_text())
            self.server = LocalServer.from_record(rec, self.log_dir)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            self.server = None

    def _save(self) -> None:
        if self.server is not None:
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.server.to_record(), indent=2))
            tmp.replace(self.state_file)
        elif self.state_file.exists():
            self.state_file.unlink()

    # ---------------------------------------------------------------- status

    def status(self) -> Dict[str, Any]:
        running = bool(self.server is not None and self.server.is_running())
        port = self.server.port if self.server is not None else DEFAULT_PORT
        return {
            "installed": self.installed(),
            "running": running,
            "port": port,
            "url": f"http://localhost:{port}",
            "install_command": INSTALL_COMMAND,
            "log_path": str(self.server.log_path) if self.server is not None else None,
        }

    # ---------------------------------------------------------------- control

    def launch(
        self,
        port: int = DEFAULT_PORT,
        connect_urls: Optional[List[str]] = None,
        open_browser: bool = True,
    ) -> Dict[str, Any]:
        """Start Open WebUI, optionally pre-connected to running model endpoints,
        and open the browser once it's ready.

        `connect_urls` are OpenAI-compatible base URLs (e.g.
        "http://localhost:8081/v1"). They're passed via the env vars Open WebUI
        reads on startup so the running models show up as ready-to-use
        connections without manual setup.
        """
        binary = self.binary()
        if binary is None:
            raise RuntimeError(
                "Open WebUI is not installed. Install it first with: " + INSTALL_COMMAND
            )
        if self.server is not None and self.server.is_running():
            status = self.status()
            if open_browser:
                webbrowser.open(status["url"])
            return status
        if port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use by another program. "
                f"Close it or pick a different port."
            )

        env: Dict[str, str] = {}
        urls = [u for u in (connect_urls or []) if u]
        if urls:
            env["ENABLE_OPENAI_API"] = "true"
            env["OPENAI_API_BASE_URLS"] = ";".join(urls)
            env["OPENAI_API_KEYS"] = ";".join([PLACEHOLDER_KEY] * len(urls))
            if len(urls) == 1:  # some versions read the singular form
                env["OPENAI_API_BASE_URL"] = urls[0]
                env["OPENAI_API_KEY"] = PLACEHOLDER_KEY

        srv = LocalServer(
            server_id="openwebui",
            engine="openwebui",
            model_label="Open WebUI",
            port=port,
            argv=[binary, "serve", "--port", str(port)],
            env=env,
            log_dir=self.log_dir,
        )
        if not srv.start():
            raise RuntimeError("Open WebUI failed to start. Check the logs for details.")
        self.server = srv
        self._save()

        if open_browser:
            threading.Thread(
                target=_open_browser_when_ready,
                args=(f"http://localhost:{port}", f"http://localhost:{port}/health"),
                daemon=True,
            ).start()

        status = self.status()
        status["connected_models"] = len(urls)
        return status

    def stop(self) -> bool:
        if self.server is None:
            return False
        ok = self.server.stop()
        self._save()
        return ok

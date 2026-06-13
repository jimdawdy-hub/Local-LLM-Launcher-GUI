"""Open WebUI integration: detect, launch, and track a local Open WebUI server.

Open WebUI (https://github.com/open-webui/open-webui) is a full-featured chat
interface that talks to OpenAI-compatible endpoints — exactly what the model
servers this app launches expose. We manage it with the same LocalServer
lifecycle used for model engines, persisted separately so it survives GUI
restarts.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from .engines.base import LocalServer
from .registry import APP_DIR, find_free_port, port_in_use

# Open WebUI's own default is 8080, which collides with llama.cpp here, so we
# default to 3000 (the port its Docker examples use).
DEFAULT_PORT = 3000
INSTALL_COMMAND = "pip install open-webui"
# Model servers launched here don't require an API key, but Open WebUI still
# sends an Authorization header, so we give it a harmless placeholder.
PLACEHOLDER_KEY = "sk-local"


def _url_fingerprint(url: str) -> Tuple[str, Optional[str], Optional[int], str]:
    """Identity of an endpoint for dedup: localhost and 127.0.0.1 are the same
    server, and a trailing slash doesn't make a different one."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host in ("127.0.0.1", "0.0.0.0", "::1"):
        host = "localhost"
    return (parts.scheme, host, parts.port, parts.path.rstrip("/"))


# Per-connection settings Open WebUI stores for entries added through its UI;
# we mirror them so our entries behave identically.
_NEW_CONNECTION_CONFIG = {
    "enable": True,
    "tags": [],
    "prefix_id": "",
    "model_ids": [],
    "connection_type": "local",
    "auth_type": "bearer",
}


def merge_connections(db_path: Path, urls: List[str]) -> bool:
    """Merge model endpoints into Open WebUI's *saved* connection list.

    Open WebUI only honors the OPENAI_API_BASE_URLS env vars on its very first
    boot; after that the list lives in its own database (webui.db) and the env
    vars are silently ignored ("PersistentConfig"). Since we own the process
    and only call this while it's stopped, we edit that saved list directly:

    - endpoints in `urls` are appended (or re-enabled if already saved, even
      under a 127.0.0.1-vs-localhost spelling difference);
    - entries *we* added earlier (recognizable by PLACEHOLDER_KEY) for models
      no longer running are removed;
    - everything the user configured by hand is left untouched.

    Returns True if the saved config was updated, False if there was nothing
    to update (no database yet — the env vars cover that first boot).
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        return False
    try:
        con = sqlite3.connect(db_path, timeout=5.0)
        try:
            row = con.execute(
                "SELECT id, data FROM config ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return False
            row_id, cfg = row[0], json.loads(row[1])

            openai = cfg.setdefault("openai", {})
            saved_urls = list(openai.get("api_base_urls") or [])
            saved_keys = list(openai.get("api_keys") or [])
            saved_keys += [""] * (len(saved_urls) - len(saved_keys))
            saved_confs = openai.get("api_configs") or {}

            wanted = {_url_fingerprint(u): u for u in urls if u}
            merged: List[Tuple[str, str, Dict[str, Any]]] = []
            for i, saved_url in enumerate(saved_urls):
                key = saved_keys[i]
                conf = dict(saved_confs.get(str(i)) or {})
                fp = _url_fingerprint(saved_url)
                if fp in wanted:  # already saved — make sure it's usable
                    conf["enable"] = True
                    wanted.pop(fp)
                elif key == PLACEHOLDER_KEY:
                    continue  # ours, for a model that's gone — drop it
                merged.append((saved_url, key, conf))
            for url in wanted.values():
                merged.append((url, PLACEHOLDER_KEY, dict(_NEW_CONNECTION_CONFIG)))

            openai["api_base_urls"] = [m[0] for m in merged]
            openai["api_keys"] = [m[1] for m in merged]
            openai["api_configs"] = {str(i): m[2] for i, m in enumerate(merged)}
            if urls:
                openai["enable"] = True

            con.execute(
                "UPDATE config SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(cfg), row_id),
            )
            con.commit()
            return True
        finally:
            con.close()
    except (sqlite3.Error, ValueError, TypeError, OSError):
        return False  # best effort — the env vars still cover a first boot


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

    def _webui_db_path(self) -> Optional[Path]:
        """Locate the webui.db Open WebUI will use, resolving its data
        directory the same way Open WebUI itself does: $DATA_DIR if set,
        otherwise a `data/` folder inside the installed open_webui package."""
        env_dir = os.environ.get("DATA_DIR")
        if env_dir:
            return Path(env_dir).expanduser() / "webui.db"

        # open-webui is often installed under a different Python than ours
        # (e.g. pip --user vs. our venv), so ask the interpreter named in the
        # launcher script's shebang line where the package lives.
        for interpreter in self._candidate_interpreters():
            try:
                out = subprocess.run(
                    [interpreter, "-c",
                     "import importlib.util; s = importlib.util.find_spec('open_webui'); "
                     "print(s.submodule_search_locations[0] if s and s.submodule_search_locations else '')"],
                    capture_output=True, text=True, timeout=15.0,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
            if out:
                return Path(out) / "data" / "webui.db"
        return None

    def _candidate_interpreters(self) -> List[str]:
        interpreters: List[str] = []
        binary = self.binary()
        if binary:
            try:
                shebang = Path(binary).read_text(errors="ignore").split("\n", 1)[0]
            except OSError:
                shebang = ""
            if shebang.startswith("#!"):
                tokens = shebang[2:].split()
                if tokens and tokens[0].endswith("/env") and len(tokens) > 1:
                    resolved = shutil.which(tokens[1])
                    if resolved:
                        interpreters.append(resolved)
                elif tokens:
                    interpreters.append(tokens[0])
        interpreters.append(sys.executable)  # last resort: our own environment
        return interpreters

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
            port = find_free_port(port)
        if port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use by another program. "
                f"Close it or pick a different port."
            )

        env: Dict[str, str] = {}
        urls = [u for u in (connect_urls or []) if u]

        # The env vars below only count on Open WebUI's very first boot; once
        # it has saved its config (webui.db), they're ignored. It's stopped
        # right now, so merge the endpoints into that saved config directly.
        db_path = self._webui_db_path()
        if db_path is not None:
            merge_connections(db_path, urls)

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

"""Tests for the Open WebUI manager and its API routes."""
import sys

from fastapi.testclient import TestClient

from local_llm_launcher.app import create_app
from local_llm_launcher.openwebui import INSTALL_COMMAND, OpenWebUIManager


def test_status_when_not_installed(tmp_path, monkeypatch):
    mgr = OpenWebUIManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "binary", lambda: None)
    st = mgr.status()
    assert st["installed"] is False
    assert st["running"] is False
    assert st["install_command"] == INSTALL_COMMAND
    assert st["url"].startswith("http://localhost:")


def test_launch_refuses_when_not_installed(tmp_path, monkeypatch):
    import pytest
    mgr = OpenWebUIManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "binary", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        mgr.launch()


def test_launch_and_persist(tmp_path, monkeypatch):
    mgr = OpenWebUIManager(app_dir=tmp_path)
    # Pretend a harmless long-running process is the open-webui binary.
    monkeypatch.setattr(mgr, "binary", lambda: sys.executable)
    monkeypatch.setattr(
        "local_llm_launcher.openwebui.port_in_use", lambda port: False
    )

    # Build a LocalServer that runs a sleep instead of `open-webui serve`.
    # Subclass so from_record (a classmethod) still works on GUI restart.
    import local_llm_launcher.openwebui as owui

    class FakeLocalServer(owui.LocalServer):
        def __init__(self, **kwargs):
            kwargs["argv"] = [sys.executable, "-c", "import time; time.sleep(60)"]
            super().__init__(**kwargs)

    monkeypatch.setattr(owui, "LocalServer", FakeLocalServer)

    st = mgr.launch(port=45999, open_browser=False)
    try:
        assert st["running"] is True
        assert st["port"] == 45999

        # A fresh manager (GUI restart) sees the same running instance.
        mgr2 = OpenWebUIManager(app_dir=tmp_path)
        assert mgr2.status()["running"] is True
    finally:
        assert mgr.stop()
    assert mgr.status()["running"] is False


def test_launch_wires_model_connection_env(tmp_path, monkeypatch):
    mgr = OpenWebUIManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "binary", lambda: sys.executable)
    monkeypatch.setattr("local_llm_launcher.openwebui.port_in_use", lambda port: False)

    import local_llm_launcher.openwebui as owui
    captured = {}

    class CapturingServer(owui.LocalServer):
        def __init__(self, **kwargs):
            captured.update(kwargs.get("env") or {})
            kwargs["argv"] = [sys.executable, "-c", "import time; time.sleep(60)"]
            super().__init__(**kwargs)

    monkeypatch.setattr(owui, "LocalServer", CapturingServer)

    mgr.launch(port=45998, open_browser=False,
               connect_urls=["http://localhost:8081/v1", "http://localhost:8080/v1"])
    try:
        assert captured["ENABLE_OPENAI_API"] == "true"
        assert captured["OPENAI_API_BASE_URLS"] == "http://localhost:8081/v1;http://localhost:8080/v1"
        assert captured["OPENAI_API_KEYS"].count(";") == 1  # one key per URL
    finally:
        mgr.stop()


def test_api_openwebui_status_route():
    client = TestClient(create_app())
    r = client.get("/api/openwebui")
    assert r.status_code == 200
    body = r.json()
    assert "installed" in body and "running" in body and "install_command" in body

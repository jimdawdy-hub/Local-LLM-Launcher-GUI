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
    monkeypatch.setattr(mgr, "_webui_db_path", lambda: None)  # never the real db
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
    monkeypatch.setattr(mgr, "_webui_db_path", lambda: None)  # never the real db
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


# --------------------------------------------------- saved-config merging
#
# Open WebUI only reads OPENAI_API_BASE_URLS env vars on its very first boot;
# after that the connection list lives in its webui.db ("PersistentConfig").
# merge_connections() edits that saved list directly, while Open WebUI is
# stopped, so launched models show up on every boot — not just the first.

import json
import sqlite3

from local_llm_launcher.openwebui import PLACEHOLDER_KEY, merge_connections


def _make_webui_db(path, openai_section):
    """Create a minimal webui.db with the same config table Open WebUI uses."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE config (id INTEGER NOT NULL, data JSON NOT NULL, "
        "version INTEGER NOT NULL, created_at DATETIME DEFAULT (CURRENT_TIMESTAMP) NOT NULL, "
        "updated_at DATETIME DEFAULT (CURRENT_TIMESTAMP), PRIMARY KEY (id))"
    )
    con.execute(
        "INSERT INTO config (id, data, version) VALUES (1, ?, 0)",
        (json.dumps({"openai": openai_section}),),
    )
    con.commit()
    con.close()


def _read_openai(path):
    con = sqlite3.connect(path)
    data = con.execute("SELECT data FROM config WHERE id = 1").fetchone()[0]
    con.close()
    return json.loads(data)["openai"]


def test_merge_appends_new_url_and_enables(tmp_path):
    db = tmp_path / "webui.db"
    _make_webui_db(db, {
        "enable": False,
        "api_base_urls": ["https://api.openai.com/v1"],
        "api_keys": ["sk-real"],
        "api_configs": {"0": {"enable": True}},
    })

    assert merge_connections(db, ["http://localhost:8081/v1"]) is True

    openai = _read_openai(db)
    assert openai["enable"] is True
    assert openai["api_base_urls"] == [
        "https://api.openai.com/v1", "http://localhost:8081/v1",
    ]
    assert openai["api_keys"] == ["sk-real", PLACEHOLDER_KEY]
    # the new entry's per-connection config exists and is enabled
    assert openai["api_configs"]["1"]["enable"] is True
    # the user's entry is untouched
    assert openai["api_configs"]["0"] == {"enable": True}


def test_merge_treats_localhost_and_127_as_same_server(tmp_path):
    db = tmp_path / "webui.db"
    _make_webui_db(db, {
        "enable": True,
        "api_base_urls": ["http://127.0.0.1:8081/v1"],
        "api_keys": ["test"],
        "api_configs": {"0": {"enable": False, "auth_type": "bearer"}},
    })

    merge_connections(db, ["http://localhost:8081/v1"])

    openai = _read_openai(db)
    # no duplicate added; the existing equivalent entry was re-enabled
    assert openai["api_base_urls"] == ["http://127.0.0.1:8081/v1"]
    assert openai["api_configs"]["0"]["enable"] is True
    assert openai["api_configs"]["0"]["auth_type"] == "bearer"  # preserved


def test_merge_prunes_stale_launcher_entries_only(tmp_path):
    db = tmp_path / "webui.db"
    _make_webui_db(db, {
        "enable": True,
        # 8087 was added by us for a model that's gone; 8080 is the user's own
        "api_base_urls": [
            "http://localhost:8087/v1", "http://127.0.0.1:8080/v1",
        ],
        "api_keys": [PLACEHOLDER_KEY, "user-key"],
        "api_configs": {"0": {"enable": True}, "1": {"enable": True}},
    })

    merge_connections(db, ["http://localhost:8081/v1"])

    openai = _read_openai(db)
    assert openai["api_base_urls"] == [
        "http://127.0.0.1:8080/v1", "http://localhost:8081/v1",
    ]
    assert openai["api_keys"] == ["user-key", PLACEHOLDER_KEY]
    # api_configs re-indexed to match the new list positions
    assert set(openai["api_configs"].keys()) == {"0", "1"}


def test_merge_missing_db_is_a_noop(tmp_path):
    assert merge_connections(tmp_path / "nope.db", ["http://localhost:8081/v1"]) is False


def test_launch_merges_into_saved_config(tmp_path, monkeypatch):
    db = tmp_path / "webui.db"
    _make_webui_db(db, {
        "enable": True,
        "api_base_urls": ["https://api.openai.com/v1"],
        "api_keys": ["sk-real"],
        "api_configs": {"0": {"enable": True}},
    })

    mgr = OpenWebUIManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "binary", lambda: sys.executable)
    monkeypatch.setattr(mgr, "_webui_db_path", lambda: db)
    monkeypatch.setattr("local_llm_launcher.openwebui.port_in_use", lambda port: False)

    import local_llm_launcher.openwebui as owui

    class FakeLocalServer(owui.LocalServer):
        def __init__(self, **kwargs):
            kwargs["argv"] = [sys.executable, "-c", "import time; time.sleep(60)"]
            super().__init__(**kwargs)

    monkeypatch.setattr(owui, "LocalServer", FakeLocalServer)

    mgr.launch(port=45997, open_browser=False,
               connect_urls=["http://localhost:8081/v1"])
    try:
        openai = _read_openai(db)
        assert "http://localhost:8081/v1" in openai["api_base_urls"]
    finally:
        mgr.stop()

"""Tests for the server manager and its persistence."""
import sys

import pytest

from local_llm_launcher.registry import ServerManager, port_in_use

GGUF = {
    "repo_id": "org/m-GGUF", "path": "/x/m-Q4.gguf", "format": "gguf",
    "size_bytes": 1, "source": "folder", "quant": "Q4_K_M", "config": {},
    "gguf_files": [{"filename": "m-Q4.gguf", "path": "/x/m-Q4.gguf", "size_bytes": 1, "quant": "Q4_K_M"}],
    "param_count_b": 8.0,
}


def test_launch_records_and_persists(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    # Swap the built command for a harmless long-running process.
    monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
        "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
        "env": {}, "port": 45123,
    })
    srv = mgr.launch("llamacpp", GGUF, {})
    try:
        assert srv.is_running()
        assert len(mgr.list()) == 1

        # A new manager instance (fresh GUI start) sees the same server.
        mgr2 = ServerManager(app_dir=tmp_path)
        assert len(mgr2.list()) == 1
        srv2 = mgr2.get(srv.server_id)
        assert srv2.is_running()
        assert srv2.pid == srv.pid
    finally:
        assert mgr.stop(srv.server_id)
    assert not srv.is_running()


def test_launch_port_conflict_raises(tmp_path, monkeypatch):
    import socket
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        assert port_in_use(port)
        mgr = ServerManager(app_dir=tmp_path)
        monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
            "argv": ["true"], "env": {}, "port": port,
        })
        with pytest.raises(RuntimeError, match="already in use"):
            mgr.launch("llamacpp", GGUF, {})
    finally:
        blocker.close()


def test_remove_dead_server(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
        "argv": [sys.executable, "-c", "pass"], "env": {}, "port": 45124,
    })
    srv = mgr.launch("llamacpp", GGUF, {})
    srv.process.wait(timeout=10)
    assert not srv.is_running()
    assert mgr.remove(srv.server_id)
    assert mgr.list() == []

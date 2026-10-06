"""Tests for the freeze watchdog: a llama-server whose work loop hangs gets stopped.

A frozen llama-server (2026-10-06, gpuhost) kept accepting connections and logging
"cancel task" for 18 minutes while its work loop never ran again. /health answers
without the work loop; /slots needs it, so a /slots request that times out while
the process lives is the freeze signal.
"""
import sys

import httpx
import pytest

from local_llm_launcher import failures
from local_llm_launcher.engines.base import LocalServer
from local_llm_launcher.registry import ServerManager

GGUF = {
    "repo_id": "org/m-GGUF", "path": "/x/m-Q4.gguf", "format": "gguf",
    "size_bytes": 1, "source": "folder", "quant": "Q4_K_M", "config": {},
    "gguf_files": [{"filename": "m-Q4.gguf", "path": "/x/m-Q4.gguf", "size_bytes": 1, "quant": "Q4_K_M"}],
    "param_count_b": 8.0,
}


def _server(mgr, monkeypatch, state, minutes=3):
    """A registered llama.cpp server whose probe reports `state[0]`."""
    srv = LocalServer(server_id="s1", engine="llamacpp", model_label="org/m", port=45201,
                      argv=[], env={}, log_dir=mgr.log_dir, watchdog_minutes=minutes)
    monkeypatch.setattr(srv, "is_running", lambda: not state[1])
    monkeypatch.setattr(srv, "probe", lambda: state[0])
    stops = []

    def stop(timeout=15.0):
        stops.append(True)
        state[1] = True
        return True

    monkeypatch.setattr(srv, "stop", stop)
    mgr.servers[srv.server_id] = srv
    return srv, stops


def test_frozen_server_is_stopped_after_the_set_minutes(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    state = ["stuck", False]
    srv, stops = _server(mgr, monkeypatch, state)
    assert mgr.watchdog_tick(now=1000.0) == []
    assert mgr.watchdog_tick(now=1000.0 + 179) == []
    assert mgr.watchdog_tick(now=1000.0 + 180) == ["s1"]
    assert stops == [True]
    explanation = failures.translate(srv.tail_logs(5))
    assert explanation and "froze" in explanation and "3 minutes" not in explanation
    assert any("watchdog" in line and "3 minutes" in line for line in srv.tail_logs(5))


def test_any_answer_resets_the_clock(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    state = ["stuck", False]
    _, stops = _server(mgr, monkeypatch, state)
    mgr.watchdog_tick(now=0.0)
    state[0] = "ok"
    mgr.watchdog_tick(now=100.0)
    state[0] = "stuck"
    mgr.watchdog_tick(now=150.0)
    assert mgr.watchdog_tick(now=300.0) == []  # only 150 s stuck since the reset
    assert stops == []


@pytest.mark.parametrize("quiet", ["loading", "ok"])
def test_loading_or_healthy_servers_are_left_alone(tmp_path, monkeypatch, quiet):
    mgr = ServerManager(app_dir=tmp_path)
    _, stops = _server(mgr, monkeypatch, [quiet, False])
    for t in range(0, 3600, 30):
        assert mgr.watchdog_tick(now=float(t)) == []
    assert stops == []


def test_watchdog_off_never_stops(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    _, stops = _server(mgr, monkeypatch, ["stuck", False], minutes=0)
    mgr.watchdog_tick(now=0.0)
    assert mgr.watchdog_tick(now=10_000.0) == []
    assert stops == []


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.mark.parametrize("health, slots, expected", [
    (httpx.ConnectError("refused"), None, "loading"),  # still loading: not listening yet
    (_Response(503), None, "loading"),                    # listening, model loading
    (_Response(200), _Response(200), "ok"),
    (_Response(200), _Response(501), "ok"),               # /slots turned off: still an answer
    (_Response(200), httpx.ReadTimeout("no answer"), "stuck"),
    (httpx.ReadTimeout("no answer"), None, "stuck"),      # accepts connections, never answers
])
def test_probe_tells_a_freeze_from_loading(tmp_path, monkeypatch, health, slots, expected):
    srv = LocalServer(server_id="s1", engine="llamacpp", model_label="org/m", port=45202,
                      argv=[], env={}, log_dir=tmp_path)

    def get(url, timeout):
        answer = health if url.endswith("/health") else slots
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(httpx, "get", get)
    assert srv.probe() == expected


def test_llamacpp_launch_arms_the_watchdog_and_it_survives_a_restart(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
        "argv": [sys.executable, "-c", "import time; time.sleep(60)"], "env": {}, "port": 45203})
    default = mgr.launch("llamacpp", GGUF, {})
    custom = mgr.launch("llamacpp", GGUF, {"watchdog_minutes": 10})
    vllm = mgr.launch("vllm-native", GGUF, {})
    try:
        assert default.watchdog_minutes == 3
        assert custom.watchdog_minutes == 10
        assert vllm.watchdog_minutes == 0  # vLLM has no /slots; not covered
        again = ServerManager(app_dir=tmp_path)
        assert again.get(custom.server_id).watchdog_minutes == 10
    finally:
        mgr.stop_all()


def test_watchdog_setting_is_catalogued_and_never_a_raw_flag():
    from local_llm_launcher import catalog
    from local_llm_launcher.engines._args import build_args_and_env

    spec = {f["key"]: f for f in catalog.load_catalog("llamacpp")["flags"]}["watchdog_minutes"]
    assert spec["flag"] is None and spec["default"] == 3 and spec["min"] == 0
    argv, env, extra = build_args_and_env("llamacpp", {"watchdog_minutes": 5})
    assert argv == [] and env == {} and extra == []

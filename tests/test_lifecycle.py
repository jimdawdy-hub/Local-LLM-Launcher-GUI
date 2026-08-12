"""Tests for server lifecycle: PID identity across GUI restarts."""
import subprocess
import sys

import psutil

from local_llm_launcher.engines.base import LocalServer

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]

RECORD = lambda **kw: {  # noqa: E731
    "server_id": "srv1",
    "engine": "llamacpp",
    "model_label": "org/m",
    "port": 45140,
    "argv": list(SLEEP),
    "container_name": None,
    "pid": None,
    "pid_create_time": None,
    "started_at": None,
    "log_path": None,
    **kw,
}


def test_stale_recycled_pid_reported_not_running(tmp_path):
    proc = subprocess.Popen(SLEEP)
    try:
        live_ct = psutil.Process(proc.pid).create_time()
        # Record persists the OLD process identity, PID now held by another process.
        record = RECORD(pid=proc.pid, pid_create_time=live_ct - 5)
        srv = LocalServer.from_record(record, tmp_path)
        assert not srv.is_running()
    finally:
        proc.terminate()
        proc.wait()


def test_stop_refuses_to_kill_recycled_pid(tmp_path):
    proc = subprocess.Popen(SLEEP)
    try:
        live_ct = psutil.Process(proc.pid).create_time()
        record = RECORD(pid=proc.pid, pid_create_time=live_ct - 5)
        srv = LocalServer.from_record(record, tmp_path)
        assert srv.stop(timeout=2)
        assert proc.poll() is None  # innocent process survives
    finally:
        proc.terminate()
        proc.wait()


def test_genuine_record_reports_running(tmp_path):
    proc = subprocess.Popen(SLEEP)
    try:
        live_ct = psutil.Process(proc.pid).create_time()
        record = RECORD(pid=proc.pid, pid_create_time=live_ct)
        srv = LocalServer.from_record(record, tmp_path)
        assert srv.is_running()
    finally:
        proc.terminate()
        proc.wait()
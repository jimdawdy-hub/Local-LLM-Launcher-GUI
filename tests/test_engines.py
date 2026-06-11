"""Tests for engine command builders and the server lifecycle."""
import sys
import time

from local_llm_launcher.engines import llamacpp, vllm_docker, vllm_native
from local_llm_launcher.engines.base import LocalServer
from local_llm_launcher import failures

MODEL = {
    "repo_id": "org/model-8B", "path": "/snap/dir", "format": "safetensors",
    "size_bytes": 8 * 1024**3, "source": "hf-cache", "quant": None, "config": {},
    "gguf_files": [], "param_count_b": 8.0,
}

GGUF = {
    "repo_id": "org/model-GGUF", "path": "/models/m-Q4_K_M.gguf", "format": "gguf",
    "size_bytes": 5 * 1024**3, "source": "folder", "quant": "Q4_K_M", "config": {},
    "gguf_files": [{"filename": "m-Q4_K_M.gguf", "path": "/models/m-Q4_K_M.gguf",
                    "size_bytes": 5 * 1024**3, "quant": "Q4_K_M"}],
    "param_count_b": 8.0,
}


def test_vllm_native_basic_command():
    spec = vllm_native.build(MODEL, {
        "port": 8002, "tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
        "max_model_len": 8192, "trust_remote_code": True,
    })
    argv = spec["argv"]
    assert argv[:3] == ["vllm", "serve", "org/model-8B"]
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8002"
    assert "--tensor-parallel-size" in argv
    assert "--trust-remote-code" in argv  # bool: flag only, no value
    assert "--quantization" not in argv   # None values omitted
    assert spec["port"] == 8002


def test_vllm_native_env_handling():
    spec = vllm_native.build(MODEL, {"device_ids": "0,1", "hf_token": "hf_secret"})
    assert spec["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert spec["env"]["HF_TOKEN"] == "hf_secret"
    assert "device_ids" not in " ".join(spec["argv"])
    assert "hf_secret" not in " ".join(spec["argv"])


def test_vllm_native_extra_args():
    spec = vllm_native.build(MODEL, {"extra_args": "--seed 42 --foo"})
    assert spec["argv"][-3:] == ["--seed", "42", "--foo"]


def test_vllm_docker_command():
    spec = vllm_docker.build(MODEL, {"port": 8002, "tensor_parallel_size": 2,
                                     "hf_token": "hf_x", "device_ids": "1"})
    argv = spec["argv"]
    assert argv[:2] == ["docker", "run"]
    assert "--ipc=host" in argv
    assert "-p" in argv and "8002:8000" in argv  # host port maps to container 8000
    assert any(a.startswith('"device=1"') or a == '"device=1"' for a in argv) or "device=1" in " ".join(argv)
    assert "--model" in argv and "org/model-8B" in argv
    # HF token passed via -e, value not in plain argv after -e HF_TOKEN form
    assert "HF_TOKEN=hf_x" in argv
    assert spec["container_name"].startswith("llml-")


def test_llamacpp_command():
    spec = llamacpp.build(GGUF, {"port": 8080, "ctx_size": 8192, "n_gpu_layers": 999,
                                 "jinja": True, "flash_attn": "auto"},
                          binary="/usr/local/bin/llama-server")
    argv = spec["argv"]
    assert argv[0] == "/usr/local/bin/llama-server"
    assert "-m" in argv and "/models/m-Q4_K_M.gguf" in argv
    assert "--ctx-size" in argv and "8192" in argv
    assert "--jinja" in argv
    assert "--flash-attn" in argv and "auto" in argv
    assert spec["port"] == 8080


def test_llamacpp_picks_specific_gguf_file():
    model = dict(GGUF)
    model["gguf_files"] = [
        {"filename": "a-Q4_K_M.gguf", "path": "/x/a-Q4_K_M.gguf", "size_bytes": 1, "quant": "Q4_K_M"},
        {"filename": "a-Q8_0.gguf", "path": "/x/a-Q8_0.gguf", "size_bytes": 2, "quant": "Q8_0"},
    ]
    spec = llamacpp.build(model, {"gguf_file": "a-Q8_0.gguf"}, binary="llama-server")
    assert "/x/a-Q8_0.gguf" in spec["argv"]


# ---------- lifecycle ----------

def test_local_server_lifecycle(tmp_path):
    srv = LocalServer(
        server_id="test1", engine="vllm", model_label="org/model-8B", port=9999,
        argv=[sys.executable, "-c", "import time; print('hello log'); time.sleep(60)"],
        env={}, log_dir=tmp_path,
    )
    assert srv.start()
    try:
        assert srv.is_running()
        time.sleep(0.5)
        logs = srv.tail_logs(10)
        assert any("hello log" in line for line in logs)
        st = srv.status()
        assert st["running"] is True and st["port"] == 9999
    finally:
        assert srv.stop()
    assert not srv.is_running()


def test_local_server_detects_exit(tmp_path):
    srv = LocalServer(
        server_id="test2", engine="vllm", model_label="m", port=9998,
        argv=[sys.executable, "-c", "print('boom: CUDA out of memory'); raise SystemExit(1)"],
        env={}, log_dir=tmp_path,
    )
    srv.start()
    time.sleep(1.0)
    assert not srv.is_running()
    st = srv.status()
    assert st["running"] is False
    assert st["exit_code"] == 1


# ---------- failure translation ----------

def test_failure_translation_oom():
    msg = failures.translate(["something", "torch.OutOfMemoryError: CUDA out of memory"])
    assert msg is not None and "memory" in msg.lower()


def test_failure_translation_quant_mismatch():
    msg = failures.translate(["Quantization method specified in the model config (compressed-tensors) does not match"])
    assert msg is not None and "quantization" in msg.lower()


def test_failure_translation_unknown_returns_none():
    assert failures.translate(["all good here"]) is None

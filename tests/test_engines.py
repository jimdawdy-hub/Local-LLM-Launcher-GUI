"""Tests for engine command builders and the server lifecycle."""
import sys
import time

import pytest

from local_llm_launcher.engines import llamacpp, sglang, tgi, vllm_docker, vllm_native
from local_llm_launcher.engines.base import LocalServer
from local_llm_launcher import failures
from pathlib import Path

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
    spec = vllm_native.build(MODEL, {"port": 8002, "tensor_parallel_size": 2})
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


def test_only_explicit_config_is_emitted():
    """Regression: catalog defaults must NOT leak into the command line.

    vLLM removed --swap-space; emitting unset defaults breaks launches whenever
    an engine changes its options between versions.
    """
    spec = vllm_native.build(MODEL, {"port": 8002})
    argv = spec["argv"]
    assert "--swap-space" not in argv
    assert "--dtype" not in argv
    assert "--gpu-memory-utilization" not in argv
    assert "--enable-prefix-caching" not in argv

    spec = llamacpp.build(GGUF, {"port": 8080}, binary="llama-server")
    argv = spec["argv"]
    assert "--batch-size" not in argv
    assert "--n-gpu-layers" not in argv  # leave it to llama.cpp's auto-fit
    assert "--split-mode" not in argv


def test_explicit_false_disables_default_on_vllm_flags():
    for build in (vllm_native.build, vllm_docker.build):
        argv = build(MODEL, {"enable_prefix_caching": False,
                             "enable_chunked_prefill": False})["argv"]
        assert "--no-enable-prefix-caching" in argv
        assert "--no-enable-chunked-prefill" in argv
        assert "--enable-prefix-caching" not in argv
    assert "--trust-remote-code" not in vllm_native.build(MODEL, {"trust_remote_code": False})["argv"]


def test_vllm_native_env_handling():
    spec = vllm_native.build(MODEL, {"device_ids": "0,1", "hf_token": "hf_secret"})
    assert spec["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    # device_ids use nvidia-smi numbering; CUDA's default order is fastest-first.
    assert spec["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert spec["env"]["HF_TOKEN"] == "hf_secret"
    assert "device_ids" not in " ".join(spec["argv"])
    assert "hf_secret" not in " ".join(spec["argv"])


@pytest.mark.parametrize("raw, expected", [(" 0, 1 ", "0,1"), (0, "0")])
def test_vllm_builders_send_normalized_device_ids(raw, expected):
    assert vllm_native.build(MODEL, {"device_ids": raw})["env"]["CUDA_VISIBLE_DEVICES"] == expected
    argv = vllm_docker.build(MODEL, {"device_ids": raw})["argv"]
    assert argv[argv.index("--gpus") + 1] == f'"device={expected}"'


def test_vllm_native_always_uses_nvidia_smi_numbering():
    # An inherited CUDA_VISIBLE_DEVICES must mean the same cards nvidia-smi lists.
    assert vllm_native.build(MODEL, {})["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert llamacpp.build(GGUF, {})["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_vllm_native_extra_args():
    spec = vllm_native.build(MODEL, {"extra_args": "--seed 42 --foo"})
    assert spec["argv"][-3:] == ["--seed", "42", "--foo"]


def test_vllm_docker_command():
    spec = vllm_docker.build(MODEL, {"port": 8002, "tensor_parallel_size": 2,
                                     "hf_token": "hf_x", "device_ids": "1"})
    argv = spec["argv"]
    assert argv[:2] == ["docker", "run"]
    assert "--ipc=host" in argv
    assert "-p" in argv and "8002:8000" in argv[argv.index("-p") + 1]  # host port maps to container 8000
    assert any(a.startswith('"device=1"') or a == '"device=1"' for a in argv) or "device=1" in " ".join(argv)
    assert "--model" in argv and "org/model-8B" in argv
    # HF token passed via --env-file (not -e) to avoid ps exposure
    assert "--env-file" in argv
    assert "hf_x" not in " ".join(argv)  # token value not in argv
    assert spec["container_name"].startswith("llml-")


def test_vllm_docker_maps_uvm_when_present(monkeypatch):
    monkeypatch.setattr(vllm_docker.os.path, "exists", lambda path: path == "/dev/nvidia-uvm")
    assert "--device=/dev/nvidia-uvm" in vllm_docker.build(MODEL, {})["argv"]
    monkeypatch.setattr(vllm_docker.os.path, "exists", lambda path: False)
    assert "--device=/dev/nvidia-uvm" not in vllm_docker.build(MODEL, {})["argv"]


def test_vllm_docker_loopback_binding_by_default():
    spec = vllm_docker.build(MODEL, {"port": 8002})
    i = spec["argv"].index("-p")
    assert spec["argv"][i + 1] == "127.0.0.1:8002:8000"


def test_vllm_docker_lan_binding_when_enabled():
    spec = vllm_docker.build(MODEL, {"port": 8002, "host": "0.0.0.0"})
    i = spec["argv"].index("-p")
    assert spec["argv"][i + 1] == "0.0.0.0:8002:8000"


def test_vllm_docker_spec_carries_env_file_path(tmp_path):
    spec = vllm_docker.build(MODEL, {"port": 8002, "hf_token": "hf_x"})
    assert "--env-file" in spec["argv"]
    env_file = spec.get("env_file")
    assert env_file and Path(env_file).is_file()
    assert "hf_x" in Path(env_file).read_text()


def test_vllm_native_loopback_binding_by_default():
    spec = vllm_native.build(MODEL, {"port": 8002})
    i = spec["argv"].index("--host")
    assert spec["argv"][i + 1] == "127.0.0.1"


def test_vllm_native_lan_binding_when_enabled():
    spec = vllm_native.build(MODEL, {"port": 8002, "host": "0.0.0.0"})
    i = spec["argv"].index("--host")
    assert spec["argv"][i + 1] == "0.0.0.0"


def test_llamacpp_loopback_binding_by_default():
    spec = llamacpp.build(GGUF, {"port": 8080}, binary="llama-server")
    i = spec["argv"].index("--host")
    assert spec["argv"][i + 1] == "127.0.0.1"


def test_llamacpp_lan_binding_when_enabled():
    spec = llamacpp.build(GGUF, {"port": 8080, "host": "0.0.0.0"}, binary="llama-server")
    i = spec["argv"].index("--host")
    assert spec["argv"][i + 1] == "0.0.0.0"


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


# ---------- SGLang ----------

def test_sglang_basic_command():
    spec = sglang.build(MODEL, {
        "port": 30001, "tensor_parallel_size": 2, "mem_fraction_static": 0.88,
        "trust_remote_code": True,
    })
    argv = spec["argv"]
    assert argv[:3] == [sys.executable, "-m", "sglang.launch_server"]
    assert "--model-path" in argv and argv[argv.index("--model-path") + 1] == "org/model-8B"
    assert "--host" in argv and "127.0.0.1" in argv
    assert "--port" in argv and argv[argv.index("--port") + 1] == "30001"
    assert "--tensor-parallel-size" in argv
    assert "--trust-remote-code" in argv  # bool: flag only, no value
    assert "--quantization" not in argv   # None values omitted
    assert spec["port"] == 30001


def test_sglang_env_handling():
    spec = sglang.build(MODEL, {"device_ids": "0,1", "hf_token": "hf_secret"})
    assert spec["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert spec["env"]["HF_TOKEN"] == "hf_secret"
    assert "device_ids" not in " ".join(spec["argv"])
    assert "hf_secret" not in " ".join(spec["argv"])


def test_sglang_rejects_api_keys_that_upstream_would_log():
    for config in ({"api_key": "private"},
                   {"extra_args": "--api-key private"},
                   {"extra_args": "--admin-api-key=private"}):
        with pytest.raises(ValueError, match="appear in its logs"):
            sglang.build(MODEL, config)


def test_sglang_extra_args():
    spec = sglang.build(MODEL, {"extra_args": "--seed 42 --foo"})
    assert spec["argv"][-3:] == ["--seed", "42", "--foo"]


def test_sglang_only_explicit_config_is_emitted():
    """Regression: catalog defaults must NOT leak into the command line."""
    spec = sglang.build(MODEL, {"port": 30001})
    argv = spec["argv"]
    assert "--mem-fraction-static" not in argv
    assert "--dtype" not in argv
    assert "--schedule-policy" not in argv


def test_tgi_docker_command_keeps_secrets_out_of_process_args():
    spec = tgi.build(MODEL, {"port": 8089, "host": "127.0.0.1", "device_ids": "1",
                             "hf_token": "hf_test_secret", "api_key": "test_api_secret",
                             "max_total_tokens": 4096, "max_input_tokens": 3072})
    try:
        args = spec["argv"]
        assert "127.0.0.1:8089:80" in args
        assert "--gpus" in args and '"device=1"' in args
        assert args[-1] == "3072"
        assert "--model-id" in args and "org/model-8B" in args
        assert "hf_test_secret" not in " ".join(args)
        assert "test_api_secret" not in " ".join(args)
        secrets = Path(spec["env_file"]).read_text()
        assert "HF_TOKEN=hf_test_secret" in secrets
        assert "API_KEY=test_api_secret" in secrets
    finally:
        Path(spec["env_file"]).unlink(missing_ok=True)


def test_tgi_rejects_gguf_and_invalid_prompt_limit():
    with pytest.raises(ValueError, match="GGUF"):
        tgi.build(GGUF, {})
    with pytest.raises(ValueError, match="smaller"):
        tgi.build(MODEL, {"max_input_tokens": 4096, "max_total_tokens": 4096})


def test_tgi_podman_gpu_selection(monkeypatch):
    monkeypatch.setattr(tgi, "_uses_podman", lambda: True)
    spec = tgi.build(MODEL, {"device_ids": "1"})
    assert spec["argv"][spec["argv"].index("--device") + 1] == "nvidia.com/gpu=1"
    assert "--gpus" not in spec["argv"]
    multi = tgi.build(MODEL, {"device_ids": "0,1"})
    assert multi["argv"].count("--device") == 2
    assert "nvidia.com/gpu=0" in multi["argv"]
    assert "nvidia.com/gpu=1" in multi["argv"]


def test_tgi_rejects_multiline_secret():
    with pytest.raises(ValueError, match="single-line"):
        tgi.build(MODEL, {"api_key": "first\nSECOND=secret"})


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


def test_local_server_env_file_deleted_on_stop(tmp_path):
    srv = LocalServer(
        server_id="env1", engine="vllm-docker", model_label="org/model-8B", port=9997,
        argv=[sys.executable, "-c", "import time; time.sleep(60)"],
        env={}, log_dir=tmp_path, env_file=str(tmp_path / "llml-x.env"),
    )
    env_file = Path(srv.env_file)
    env_file.write_text("HF_TOKEN=hf_x\n")
    assert srv.start()
    try:
        assert srv.is_running()
        assert env_file.is_file()
    finally:
        assert srv.stop()
    assert not env_file.exists()


def test_local_server_env_file_deleted_after_unexpected_exit(tmp_path):
    env_file = tmp_path / "llml-exit.env"
    env_file.write_text("HF_TOKEN=hf_x\n")
    srv = LocalServer(
        server_id="env-exit", engine="tgi", model_label="org/model-8B", port=9996,
        argv=[sys.executable, "-c", "raise SystemExit(1)"],
        env={}, log_dir=tmp_path, env_file=str(env_file),
    )
    assert srv.start()
    srv.process.wait(timeout=5)
    assert not srv.is_running()
    assert not env_file.exists()
    assert srv.to_record()["env_file"] is None


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


def test_catalog_new_flags_and_advanced_placement():
    from local_llm_launcher import catalog

    llama_flags = {f["key"]: f for f in catalog.load_catalog("llamacpp")["flags"]}
    vllm_flags = {f["key"]: f for f in catalog.load_catalog("vllm")["flags"]}

    # Non-advanced exceptions requested by user: temp, system_prompt, host, n_cpu_moe
    for key, expected_cat in (
        ("temp", "essential"),
        ("system_prompt", "essential"),
        ("host", "api"),
        ("n_cpu_moe", "performance"),
    ):
        assert key in llama_flags
        assert llama_flags[key]["category"] == expected_cat
        assert not llama_flags[key].get("advanced", False), f"{key} should not be in advanced"

    assert "host" in vllm_flags
    assert vllm_flags["host"]["category"] == "api"
    assert not vllm_flags["host"].get("advanced", False)

    # Advanced flags
    for key in (
        "ctx_shift",
        "slots",
        "lazy_mode",
        "moe_cache_mib",
        "n_cpu_ffn",
        "color",
        "system_prompt_file",
        "reasoning",
        "reasoning_format",
        "reasoning_effort",
        "reasoning_budget",
        "spec_draft_model",
        "spec_draft_ngl",
        "spec_draft_n_max",
        "spec_draft_n_min",
        "spec_draft_p_min",
    ):
        assert key in llama_flags, f"Missing {key} in llamacpp catalog"
        assert llama_flags[key].get("advanced") is True, f"{key} should be advanced"


def test_llamacpp_new_flags_emission(monkeypatch):
    from local_llm_launcher import hardware
    monkeypatch.setattr(hardware, "llama_capabilities", lambda _: {"load_mode": False,
                        "server_options": {"system_prompt": True, "color": True}})
    spec = llamacpp.build(
        GGUF,
        {
            "port": 8080,
            "host": "192.168.1.50",
            "temp": 0.6,
            "system_prompt": "You are concise.",
            "n_cpu_moe": 8,
            "moe_cache_mib": 2048,
            "n_cpu_ffn": 4,
            "lazy_mode": "on",
            "ctx_shift": True,
            "slots": False,
            "color": "on",
            "reasoning": "on",
            "reasoning_format": "deepseek",
            "reasoning_effort": "high",
            "reasoning_budget": 1024,
            "spec_draft_model": "/models/draft.gguf",
            "spec_draft_ngl": 99,
            "spec_draft_n_max": 16,
            "spec_draft_n_min": 2,
            "spec_draft_p_min": 0.75,
        },
        binary="llama-server",
    )
    argv = spec["argv"]
    assert argv.count("--host") == 1
    assert argv[argv.index("--host") + 1] == "192.168.1.50"
    assert argv[argv.index("--temp") + 1] == "0.6"
    assert argv[argv.index("--system-prompt") + 1] == "You are concise."
    assert argv[argv.index("--n-cpu-moe") + 1] == "8"
    assert argv[argv.index("--moe-cache-mib") + 1] == "2048"
    assert argv[argv.index("--n-cpu-ffn") + 1] == "4"
    assert argv[argv.index("--lazy-mode") + 1] == "on"
    assert "--context-shift" in argv
    assert "--no-slots" in argv and "--slots" not in argv
    assert argv[argv.index("--color") + 1] == "on"
    assert argv[argv.index("--reasoning") + 1] == "on"
    assert argv[argv.index("--reasoning-format") + 1] == "deepseek"
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    assert argv[argv.index("--reasoning-budget") + 1] == "1024"
    assert argv[argv.index("--spec-draft-model") + 1] == "/models/draft.gguf"
    assert argv[argv.index("--spec-draft-ngl") + 1] == "99"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "16"
    assert argv[argv.index("--spec-draft-n-min") + 1] == "2"
    assert argv[argv.index("--spec-draft-p-min") + 1] == "0.75"


def test_llamacpp_negative_ctx_shift_and_alias():
    argv_false = llamacpp.build(GGUF, {"ctx_shift": False}, binary="llama-server")["argv"]
    assert "--no-context-shift" in argv_false
    argv_alias = llamacpp.build(GGUF, {"context_shift": True}, binary="llama-server")["argv"]
    assert "--context-shift" in argv_alias


def test_host_emitted_once_across_all_builders():
    v_native = vllm_native.build(MODEL, {"host": "10.0.0.5", "port": 8000})["argv"]
    assert v_native.count("--host") == 1
    assert v_native[v_native.index("--host") + 1] == "10.0.0.5"

    v_docker = vllm_docker.build(MODEL, {"host": "10.0.0.5", "port": 8000})["argv"]
    assert v_docker[v_docker.index("-p") + 1] == "10.0.0.5:8000:8000"
    assert v_docker.count("--host") == 1
    assert v_docker[v_docker.index("--host") + 1] == "0.0.0.0"


def test_placement_validation_for_new_flags():
    with pytest.raises(ValueError, match="listen address"):
        llamacpp.build(GGUF, {"host": "127.0.0.1 8080"})
    with pytest.raises(ValueError, match="system prompt"):
        llamacpp.build(GGUF, {"system_prompt": "hi", "system_prompt_file": "/tmp/p.txt"})
    with pytest.raises(ValueError, match="speculative draft"):
        llamacpp.build(GGUF, {"spec_draft_n_min": 10, "spec_draft_n_max": 5})
    with pytest.raises(ValueError, match="reasoning_budget"):
        llamacpp.build(GGUF, {"reasoning_budget": -2})
    # -1 and 0 are valid for reasoning_budget
    assert "--reasoning-budget" in llamacpp.build(GGUF, {"reasoning_budget": -1})["argv"]
    assert "--reasoning-budget" in llamacpp.build(GGUF, {"reasoning_budget": 0})["argv"]
    with pytest.raises(ValueError, match="Lazy mode"):
        llamacpp.build(GGUF, {"lazy_mode": "invalid"})
    with pytest.raises(ValueError, match="Reasoning effort"):
        llamacpp.build(GGUF, {"reasoning_effort": "ultra"})

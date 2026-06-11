"""Tests for the advisor (traffic-light) engine.

The advisor works on plain dicts (the JSON shapes produced by Hardware.to_dict()
and LocalModel.to_dict()) so it is easy to call from the API layer.
"""
from local_llm_launcher import advisor

GB = 1024**3

DUAL_5060TI = {
    "gpus": [
        {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_total_mb": 16311, "vram_free_mb": 15090,
         "compute_capability": "12.0", "index": 0},
        {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_total_mb": 16311, "vram_free_mb": 16100,
         "compute_capability": "12.0", "index": 1},
    ],
    "apple_silicon": None, "cpu_cores": 24, "ram_gb": 31.0, "disk_free_gb": 500.0,
    "total_vram_mb": 32622,
    "engines": {"vllm_native": True, "vllm_docker": True, "llamacpp_path": "/usr/local/bin/llama-server"},
}

APPLE_M3 = {
    "gpus": [], "apple_silicon": {"chip": "Apple M3 Pro", "memory_gb": 36},
    "cpu_cores": 12, "ram_gb": 36.0, "disk_free_gb": 500.0, "total_vram_mb": 0,
    "engines": {"vllm_native": False, "vllm_docker": False, "llamacpp_path": "/opt/homebrew/bin/llama-server"},
}

CPU_ONLY = {
    "gpus": [], "apple_silicon": None, "cpu_cores": 8, "ram_gb": 16.0,
    "disk_free_gb": 100.0, "total_vram_mb": 0,
    "engines": {"vllm_native": False, "vllm_docker": False, "llamacpp_path": "/usr/bin/llama-server"},
}


def safetensors_model(size_gb=15.0, params=27.0, quant="awq", config=None):
    return {
        "repo_id": "test/model", "path": "/x", "format": "safetensors",
        "size_bytes": int(size_gb * GB), "source": "hf-cache", "quant": quant,
        "config": config or {}, "gguf_files": [], "param_count_b": params,
    }


def gguf_model(size_gb=5.0, params=8.0):
    return {
        "repo_id": "test/model-GGUF", "path": "/x/model-Q4_K_M.gguf", "format": "gguf",
        "size_bytes": int(size_gb * GB), "source": "folder", "quant": "Q4_K_M",
        "config": {}, "gguf_files": [{"filename": "model-Q4_K_M.gguf", "path": "/x/model-Q4_K_M.gguf",
                                       "size_bytes": int(size_gb * GB), "quant": "Q4_K_M"}],
        "param_count_b": params,
    }


# ---------- overall fit ----------

def test_small_model_fits_green():
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 8192, "max_num_seqs": 1}, DUAL_5060TI)
    assert a["overall"]["level"] == "green"
    assert a["budget"]["available_gb"] > 0


def test_huge_model_red():
    a = advisor.advise("vllm", safetensors_model(size_gb=70.0, params=70.0, quant=None),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 8192}, DUAL_5060TI)
    assert a["overall"]["level"] == "red"


def test_tight_model_yellow():
    # ~25GB weights on ~26GB usable budget -> yellow zone
    a = advisor.advise("vllm", safetensors_model(size_gb=21.5, params=35.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 4096, "max_num_seqs": 1}, DUAL_5060TI)
    assert a["overall"]["level"] == "yellow"


# ---------- hard blockers ----------

def test_gguf_on_vllm_is_red():
    a = advisor.advise("vllm", gguf_model(), {"tensor_parallel_size": 1}, DUAL_5060TI)
    assert a["overall"]["level"] == "red"
    assert "llama.cpp" in a["overall"]["headline"]


def test_safetensors_on_llamacpp_is_red():
    a = advisor.advise("llamacpp", safetensors_model(), {}, DUAL_5060TI)
    assert a["overall"]["level"] == "red"


def test_vllm_on_apple_silicon_red():
    a = advisor.advise("vllm", safetensors_model(size_gb=4.0, params=7.0), {}, APPLE_M3)
    assert a["overall"]["level"] == "red"


def test_vllm_no_gpu_red():
    a = advisor.advise("vllm", safetensors_model(size_gb=4.0, params=7.0), {}, CPU_ONLY)
    assert a["overall"]["level"] == "red"


# ---------- vLLM flag rules ----------

def test_util_exceeding_free_vram_red():
    # GPU 0 has 15090/16311 free => max safe util ~0.925; ask 0.95
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.95,
                        "max_model_len": 4096}, DUAL_5060TI)
    assert a["flags"]["gpu_memory_utilization"]["level"] == "red"


def test_util_high_on_consumer_yellow():
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.91,
                        "max_model_len": 4096}, DUAL_5060TI)
    assert a["flags"]["gpu_memory_utilization"]["level"] == "yellow"


def test_tp_more_than_gpus_red():
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0),
                       {"tensor_parallel_size": 4}, DUAL_5060TI)
    assert a["flags"]["tensor_parallel_size"]["level"] == "red"


def test_tp1_when_model_needs_both_gpus_yellow():
    a = advisor.advise("vllm", safetensors_model(size_gb=20.0, params=35.0),
                       {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.85,
                        "max_model_len": 4096}, DUAL_5060TI)
    assert a["flags"]["tensor_parallel_size"]["level"] in ("yellow", "red")


def test_explicit_quantization_yellow():
    a = advisor.advise("vllm", safetensors_model(),
                       {"tensor_parallel_size": 2, "quantization": "awq"}, DUAL_5060TI)
    assert a["flags"]["quantization"]["level"] == "yellow"


def test_cpu_offload_yellow():
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0),
                       {"tensor_parallel_size": 2, "cpu_offload_gb": 8}, DUAL_5060TI)
    assert a["flags"]["cpu_offload_gb"]["level"] == "yellow"
    assert "slow" in a["flags"]["cpu_offload_gb"]["message"].lower()


def test_max_model_len_above_model_limit_yellow():
    cfg = {"max_position_embeddings": 32768}
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0, config=cfg),
                       {"tensor_parallel_size": 2, "max_model_len": 65536}, DUAL_5060TI)
    assert a["flags"]["max_model_len"]["level"] == "yellow"


def test_reasoning_parser_hint_for_qwen3():
    m = safetensors_model(size_gb=8.0, params=8.0)
    m["repo_id"] = "Qwen/Qwen3-8B"
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2}, DUAL_5060TI)
    assert a["flags"]["reasoning_parser"]["level"] == "yellow"


# ---------- llama.cpp flag rules ----------

def test_llamacpp_green_fit():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"n_gpu_layers": 999, "ctx_size": 8192}, DUAL_5060TI)
    assert a["overall"]["level"] == "green"


def test_llamacpp_zero_gpu_layers_yellow_with_gpu_present():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"n_gpu_layers": 0, "ctx_size": 8192}, DUAL_5060TI)
    assert a["flags"]["n_gpu_layers"]["level"] == "yellow"


def test_llamacpp_v_cache_quant_without_flash_attn_red():
    a = advisor.advise("llamacpp", gguf_model(),
                       {"cache_type_v": "q8_0", "flash_attn": "off"}, DUAL_5060TI)
    assert a["flags"]["cache_type_v"]["level"] == "red"


def test_llamacpp_threads_above_cores_yellow():
    a = advisor.advise("llamacpp", gguf_model(), {"threads": 64}, DUAL_5060TI)
    assert a["flags"]["threads"]["level"] == "yellow"


def test_llamacpp_apple_silicon_green():
    a = advisor.advise("llamacpp", gguf_model(size_gb=10.0, params=13.0),
                       {"n_gpu_layers": 999, "ctx_size": 8192}, APPLE_M3)
    assert a["overall"]["level"] == "green"


def test_llamacpp_cpu_only_big_model_red():
    a = advisor.advise("llamacpp", gguf_model(size_gb=20.0, params=35.0),
                       {"ctx_size": 8192}, CPU_ONLY)
    assert a["overall"]["level"] == "red"


# ---------- presets ----------

def test_presets_exist_and_are_advised():
    m = safetensors_model(size_gb=15.0, params=27.0)
    ps = advisor.presets("vllm", m, DUAL_5060TI)
    names = [p["name"] for p in ps]
    assert "Safe (recommended)" in names
    for p in ps:
        assert isinstance(p["config"], dict)
        assert p["config"].get("tensor_parallel_size") == 2  # uses both GPUs


def test_presets_llamacpp():
    ps = advisor.presets("llamacpp", gguf_model(), APPLE_M3)
    assert len(ps) >= 2


# ---------- kv cache math ----------

def test_kv_cache_from_config_is_sane():
    cfg = {"num_hidden_layers": 32, "hidden_size": 4096,
           "num_attention_heads": 32, "num_key_value_heads": 8}
    gb = advisor.estimate_kv_gb(cfg, None, max_len=8192, seqs=1, dtype_bytes=2)
    assert 0.9 < gb < 1.1  # 32 layers * 8 kv heads * 128 dim * 2 * 2B * 8192 = 1.0 GB

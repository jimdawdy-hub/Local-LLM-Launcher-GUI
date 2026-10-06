"""Tests for the advisor (traffic-light) engine.

The advisor works on plain dicts (the JSON shapes produced by Hardware.to_dict()
and LocalModel.to_dict()) so it is easy to call from the API layer.
"""
import json
import struct

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


def safetensors_model(size_gb=15.0, params=27.0, quant="awq", config=None, multimodal=False):
    return {
        "repo_id": "test/model", "path": "/x", "format": "safetensors",
        "size_bytes": int(size_gb * GB), "source": "hf-cache", "quant": quant,
        "config": config or {}, "gguf_files": [], "param_count_b": params,
        "multimodal": multimodal,
    }


def gguf_model(size_gb=5.0, params=8.0, multimodal=False):
    return {
        "repo_id": "test/model-GGUF", "path": "/x/model-Q4_K_M.gguf", "format": "gguf",
        "size_bytes": int(size_gb * GB), "source": "folder", "quant": "Q4_K_M",
        "config": {}, "gguf_files": [{"filename": "model-Q4_K_M.gguf", "path": "/x/model-Q4_K_M.gguf",
                                       "size_bytes": int(size_gb * GB), "quant": "Q4_K_M"}],
        "param_count_b": params, "multimodal": multimodal,
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


def test_memory_only_red_can_be_overridden():
    # Too big for the cards is a warning the user may ignore; the button
    # turns into a red "you have been warned" instead of refusing.
    a = advisor.advise("vllm", safetensors_model(size_gb=70.0, params=70.0, quant=None),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 8192}, DUAL_5060TI)
    assert a["overall"]["level"] == "red"
    assert a["overall"]["override"] is True


def test_load_headroom_red_can_be_overridden():
    a = advisor.advise("vllm", safetensors_model(size_gb=23.2, params=31.0),
                       {"tensor_parallel_size": 2, "max_model_len": 4096},
                       busy_gpu0_hw(free0_mb=10240))
    assert a["overall"]["level"] == "red"
    assert a["overall"]["override"] is True


def test_hard_blocker_cannot_be_overridden():
    a = advisor.advise("vllm", gguf_model(), {"tensor_parallel_size": 1}, DUAL_5060TI)
    assert a["overall"]["level"] == "red"
    assert a["overall"].get("override") is not True


def test_fitting_model_has_no_override():
    a = advisor.advise("vllm", safetensors_model(size_gb=8.0, params=8.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 8192, "max_num_seqs": 1}, DUAL_5060TI)
    assert a["overall"].get("override") is not True


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


def test_reasoning_parser_family_mismatch_red():
    # Real-world failure: gemma-4 model launched with the qwen3 parser aborts
    # instantly ("could not locate think start/end tokens").
    m = safetensors_model(size_gb=23.0, params=31.0)
    m["repo_id"] = "QuantTrio/gemma-4-31B-it-AWQ-6Bit"
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2, "reasoning_parser": "qwen3"},
                       DUAL_5060TI)
    assert a["flags"]["reasoning_parser"]["level"] == "red"
    assert "gemma4" in a["flags"]["reasoning_parser"]["message"]
    # A red flag must also gate the overall verdict.
    assert a["overall"]["level"] in ("yellow", "red")


def test_reasoning_parser_matching_family_ok():
    m = safetensors_model(size_gb=23.0, params=31.0)
    m["repo_id"] = "QuantTrio/gemma-4-31B-it-AWQ-6Bit"
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2, "reasoning_parser": "gemma4"},
                       DUAL_5060TI)
    assert a["flags"].get("reasoning_parser") is None


def test_reasoning_parser_unknown_family_yellow():
    m = safetensors_model(size_gb=8.0, params=8.0)
    m["repo_id"] = "mistralai/Mistral-7B-Instruct"
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2, "reasoning_parser": "qwen3"},
                       DUAL_5060TI)
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


# ---------- per-GPU load headroom (the "display tax") ----------

def busy_gpu0_hw(free0_mb):
    """Dual 16GB cards where the desktop occupies part of GPU 0."""
    hw = dict(DUAL_5060TI)
    hw["gpus"] = [
        {"name": "RTX 5060 Ti", "vram_total_mb": 15866, "vram_free_mb": free0_mb,
         "compute_capability": "12.0", "index": 0},
        {"name": "RTX 5060 Ti", "vram_total_mb": 15888, "vram_free_mb": 15737,
         "compute_capability": "12.0", "index": 1},
    ]
    return hw


def test_load_headroom_warns_when_display_gpu_is_busy():
    # Real failure: 23 GiB AWQ model at TP=2 needs ~14.6 GB free per card;
    # GPU 0 had only ~13.5 GB free (desktop using ~2 GB) and OOMed by ~400 MB.
    hw = busy_gpu0_hw(free0_mb=13824)  # 13.5 GiB free
    a = advisor.advise("vllm", safetensors_model(size_gb=23.2, params=31.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.86,
                        "max_model_len": 4096, "max_num_seqs": 1,
                        "kv_cache_dtype": "fp8"}, hw)
    assert a["overall"]["level"] in ("yellow", "red")
    details = " ".join(a["overall"]["details"])
    assert "GPU 0" in details
    assert "Close GPU-heavy apps" in details


def test_load_headroom_quiet_when_gpu_is_free():
    hw = busy_gpu0_hw(free0_mb=15400)  # desktop closed, ~15 GiB free
    a = advisor.advise("vllm", safetensors_model(size_gb=23.2, params=31.0),
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.86,
                        "max_model_len": 4096, "max_num_seqs": 1,
                        "kv_cache_dtype": "fp8"}, hw)
    assert not any("Close GPU-heavy apps" in d for d in a["overall"]["details"])


def test_load_headroom_red_when_far_short():
    hw = busy_gpu0_hw(free0_mb=10240)  # 10 GiB free — hopeless
    a = advisor.advise("vllm", safetensors_model(size_gb=23.2, params=31.0),
                       {"tensor_parallel_size": 2, "max_model_len": 4096}, hw)
    assert a["overall"]["level"] == "red"


# ---------- multimodal / text-only ----------

def test_multimodal_vllm_suggests_text_only():
    m = safetensors_model(size_gb=8.0, params=8.0, multimodal=True)
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2}, DUAL_5060TI)
    assert a["flags"]["language_model_only"]["level"] == "yellow"
    assert "Text-only" in a["flags"]["language_model_only"]["message"]


def test_multimodal_vllm_text_only_on_is_green():
    m = safetensors_model(size_gb=8.0, params=8.0, multimodal=True)
    a = advisor.advise("vllm", m,
                       {"tensor_parallel_size": 2, "language_model_only": True}, DUAL_5060TI)
    assert a["flags"]["language_model_only"]["level"] == "green"


def test_non_multimodal_vllm_no_text_only_flag():
    m = safetensors_model(size_gb=8.0, params=8.0, multimodal=False)
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2}, DUAL_5060TI)
    assert "language_model_only" not in a["flags"]


def test_multimodal_tight_fit_recommends_text_only_in_verdict():
    # The real case: 31B vision model, tight on 2x16GB, text-only not yet on.
    m = safetensors_model(size_gb=21.5, params=31.0, multimodal=True)
    a = advisor.advise("vllm", m,
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 4096, "max_num_seqs": 1}, DUAL_5060TI)
    assert a["overall"]["level"] in ("yellow", "red")
    assert any("Text-only" in d for d in a["overall"]["details"])


def test_multimodal_text_only_on_no_verdict_nag():
    m = safetensors_model(size_gb=21.5, params=31.0, multimodal=True)
    a = advisor.advise("vllm", m,
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 4096, "max_num_seqs": 1,
                        "language_model_only": True}, DUAL_5060TI)
    assert not any("turn on Text-only" in d for d in a["overall"]["details"])


def test_multimodal_llamacpp_explains_vision_part_is_never_loaded():
    # The launcher starts llama-server with -m and no --mmproj, so the image projector
    # never loads; Text-only must not promise to save memory or change the estimate.
    m = gguf_model(size_gb=5.0, params=8.0, multimodal=True)
    for text_only in (False, True):
        a = advisor.advise("llamacpp", m, {"n_gpu_layers": 999, "ctx_size": 8192,
                                           "no_mmproj": text_only}, DUAL_5060TI)
        assert a["flags"]["no_mmproj"]["level"] == "green"
        assert "never loaded" in a["flags"]["no_mmproj"]["message"]
        assert a["budget"]["needed_gb"] == advisor.advise(
            "llamacpp", m, {"n_gpu_layers": 999, "ctx_size": 8192}, DUAL_5060TI)["budget"]["needed_gb"]


# ---------- KV-cache gate (weights fit but conversation memory starves) ----------

def test_kv_gate_flags_weights_fit_but_no_kv_room():
    # The real failure: 31B weights load (11.6 GB/card on 2x16GB at util 0.85),
    # leaving almost nothing for KV cache at a 4096 context.
    m = safetensors_model(size_gb=23.2, params=31.0)
    a = advisor.advise("vllm", m,
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 4096, "max_num_seqs": 1,
                        "kv_cache_dtype": "fp8"}, DUAL_5060TI)
    joined = " ".join(a["overall"]["details"])
    assert "conversation memory" in joined or "KV cache" in joined
    assert a["overall"]["level"] in ("yellow", "red")


def test_kv_gate_quiet_when_plenty_of_room():
    # Small model leaves loads of room for KV — no KV-gate warning.
    m = safetensors_model(size_gb=8.0, params=8.0)
    a = advisor.advise("vllm", m,
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 4096, "max_num_seqs": 1}, DUAL_5060TI)
    assert not any("conversation memory (KV cache) and" in d for d in a["overall"]["details"])


def test_kv_gate_suggests_lower_context():
    m = safetensors_model(size_gb=23.2, params=31.0)
    a = advisor.advise("vllm", m,
                       {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                        "max_model_len": 8192, "max_num_seqs": 1,
                        "kv_cache_dtype": "fp8"}, DUAL_5060TI)
    joined = " ".join(a["overall"]["details"])
    # Should mention a remedy: lower context, raise util, or fp8.
    assert any(w in joined for w in ("context window", "memory usage limit", "fp8"))


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


def test_llamacpp_cache_precision_accounts_for_k_and_v_independently():
    model = gguf_model()
    model["config"] = {"num_hidden_layers": 32, "hidden_size": 4096,
                       "num_attention_heads": 32, "num_key_value_heads": 8}
    for k_type, v_type, expected_gb in (
        ("f16", "bf16", 1.0),
        ("f32", "bf16", 1.5),
        ("bf16", "f32", 1.5),
        ("f32", "f32", 2.0),
    ):
        result = advisor.advise("llamacpp", model, {
            "ctx_size": 8192, "cache_type_k": k_type, "cache_type_v": v_type,
        }, CPU_ONLY)
        assert result["budget"]["kv_cache_gb"] == expected_gb, (k_type, v_type)


def test_llamacpp_f32_cache_changes_fit_when_memory_is_insufficient():
    model = gguf_model(size_gb=5.0)
    model["config"] = {"num_hidden_layers": 32, "hidden_size": 4096,
                       "num_attention_heads": 32, "num_key_value_heads": 8}
    half = advisor.advise("llamacpp", model, {
        "ctx_size": 32768, "cache_type_k": "f16", "cache_type_v": "bf16",
    }, CPU_ONLY)
    full = advisor.advise("llamacpp", model, {
        "ctx_size": 32768, "cache_type_k": "f32", "cache_type_v": "f32",
    }, CPU_ONLY)
    assert half["budget"]["needed_gb"] < half["budget"]["available_gb"]
    assert half["overall"]["level"] != "red"
    assert full["budget"]["needed_gb"] > full["budget"]["available_gb"]
    assert full["overall"]["level"] == "red"


# ---------- --no-kv-offload (KV cache to system RAM) ----------

def test_no_kv_offload_red_flag():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"n_gpu_layers": 999, "ctx_size": 8192, "no_kv_offload": True},
                       DUAL_5060TI)
    assert a["flags"]["no_kv_offload"]["level"] == "red"
    assert "HEAVY SPEED PENALTY" in a["flags"]["no_kv_offload"]["message"]


def test_no_kv_offload_long_context_extra_warning():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"n_gpu_layers": 999, "ctx_size": 132000, "no_kv_offload": True},
                       DUAL_5060TI)
    msg = a["flags"]["no_kv_offload"]["message"]
    assert "worst-case combination" in msg


def test_no_kv_offload_zeroes_kv_in_budget():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"n_gpu_layers": 999, "ctx_size": 8192, "no_kv_offload": True},
                       DUAL_5060TI)
    assert a["budget"]["kv_cache_gb"] == 0.0


def test_no_kv_offload_reduces_needed_gb():
    normal = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                            {"n_gpu_layers": 999, "ctx_size": 8192}, DUAL_5060TI)
    offloaded = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                               {"n_gpu_layers": 999, "ctx_size": 8192, "no_kv_offload": True},
                               DUAL_5060TI)
    assert offloaded["budget"]["needed_gb"] < normal["budget"]["needed_gb"]


def test_no_kv_offload_false_has_no_flag():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"n_gpu_layers": 999, "ctx_size": 8192, "no_kv_offload": False},
                       DUAL_5060TI)
    assert "no_kv_offload" not in a["flags"]


def test_selected_gpus_follow_inherited_mask_when_device_ids_unset():
    gpus = [{"index": 0, "vram_total_mb": 8000}, {"index": 1, "vram_total_mb": 16000}]
    hw = {"gpus": gpus, "cuda_visible_devices": "1"}
    assert advisor._selected_gpus(hw, {}) == [gpus[1]]
    assert advisor._selected_gpus(hw, {"device_ids": "0"}) == [gpus[0]]


# ---------- exact sizes read from the model files ----------

def _write_gguf(path, metadata):
    """Minimal GGUF v3 header: uint32, float32, string, and int32/bool arrays (tuples)."""
    def string(text):
        data = text.encode()
        return struct.pack('<Q', len(data)) + data
    body = b''
    for key, value in metadata.items():
        if isinstance(value, tuple):
            inner, fmt = (7, '?') if isinstance(value[0], bool) else (5, 'i')
            body += (string(key) + struct.pack('<IIQ', 9, inner, len(value))
                     + struct.pack(f'<{len(value)}{fmt}', *value))
        elif isinstance(value, str):
            body += string(key) + struct.pack('<I', 8) + string(value)
        elif isinstance(value, float):
            body += string(key) + struct.pack('<I', 6) + struct.pack('<f', value)
        else:
            body += string(key) + struct.pack('<I', 4) + struct.pack('<I', value)
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(metadata)) + body)
    return str(path)


def _gguf_file_model(path, size_gb, extra_files=()):
    files = [{"filename": path.rsplit('/', 1)[-1], "path": path, "size_bytes": int(size_gb * GB)}]
    files += [{"filename": f.rsplit('/', 1)[-1], "path": f, "size_bytes": int(s * GB)} for f, s in extra_files]
    return {
        "repo_id": "test/model-GGUF", "path": path.rsplit('/', 1)[0], "format": "gguf",
        "size_bytes": sum(f["size_bytes"] for f in files), "source": "hf-cache", "quant": None,
        "config": {}, "gguf_files": files, "param_count_b": 27.0, "multimodal": False,
    }


QWEN35_27B = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65, "qwen35.nextn_predict_layers": 1,
    "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
    "qwen35.embedding_length": 5120, "qwen35.full_attention_interval": 4,
}


def test_llamacpp_kv_from_gguf_counts_only_attention_layers(tmp_path):
    # The real case: Qwen3.8 27B keeps a KV cache in 16 of its 64 layers. At 200k
    # context with q8_0 K and q4_0 V, llama.cpp reserved ~5 GB, not the 11.7 GB guessed.
    path = _write_gguf(tmp_path / "qwen.gguf", QWEN35_27B)
    a = advisor.advise("llamacpp", _gguf_file_model(path, 15.75),
                       {"ctx_size": 200000, "flash_attn": "auto",
                        "cache_type_k": "q8_0", "cache_type_v": "q4_0"}, DUAL_5060TI)
    expected = 16 * 4 * 256 * (34 / 32 + 18 / 32) * 200000 / GB
    assert a["budget"]["kv_cache_gb"] == round(expected, 1) == 5.0
    # Two cards each reserve a working buffer, including an f16 copy of one layer's K and V.
    compute = 2 * advisor._compute_bytes(200000, 512, 200000 * 4 * 512 * 2) / GB
    assert a["budget"]["needed_gb"] == round(15.75 + expected + compute, 1)


def test_llamacpp_kv_from_gguf_dense_model_f16(tmp_path):
    meta = {"general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096}
    path = _write_gguf(tmp_path / "llama.gguf", meta)
    a = advisor.advise("llamacpp", _gguf_file_model(path, 5.0), {"ctx_size": 8192}, DUAL_5060TI)
    # head width defaults to embedding / heads = 128; K and V each 2 bytes per value.
    assert a["budget"]["kv_cache_gb"] == round(32 * 8 * 128 * 4 * 8192 / GB, 1)


def test_llamacpp_kv_falls_back_to_guess_when_file_unreadable():
    a = advisor.advise("llamacpp", gguf_model(size_gb=5.0, params=8.0),
                       {"ctx_size": 8192}, DUAL_5060TI)
    assert a["budget"]["kv_cache_gb"] == round(0.125 * 8192 / 1024, 1)


def test_llamacpp_weights_count_only_the_file_that_loads(tmp_path):
    # A GGUF repo can hold several quants plus an image file (mmproj); llama-server
    # loads just one model file, and never the mmproj without --mmproj.
    path = _write_gguf(tmp_path / "Model-Q4_K_M.gguf", QWEN35_27B)
    other = _write_gguf(tmp_path / "Model-Q8_0.gguf", QWEN35_27B)
    mmproj = str(tmp_path / "mmproj-F16.gguf")
    m = _gguf_file_model(path, 10.0, [(other, 20.0), (mmproj, 1.0)])
    a = advisor.advise("llamacpp", m, {"ctx_size": 8192}, DUAL_5060TI)
    assert a["budget"]["weights_gb"] == 10.0
    b = advisor.advise("llamacpp", m, {"ctx_size": 8192, "gguf_file": "Model-Q8_0.gguf"}, DUAL_5060TI)
    assert b["budget"]["weights_gb"] == 20.0


def _write_safetensors(path, tensors):
    """Header-only safetensors file: {name: byte length}, laid out back to back."""
    header, offset = {}, 0
    for name, size in tensors.items():
        header[name] = {"dtype": "BF16", "shape": [size // 2], "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data)


def test_vllm_text_only_subtracts_vision_encoder_weights(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.visual.blocks.0.attn.qkv.weight": int(1.5 * GB),
        "model.multi_modal_projector.linear.weight": int(0.5 * GB),
        "model.language_model.layers.0.mlp.up_proj.weight": int(6 * GB),
    })
    m = safetensors_model(size_gb=8.0, params=8.0, multimodal=True)
    m["path"] = str(tmp_path)
    off = advisor.advise("vllm", m, {"tensor_parallel_size": 2}, DUAL_5060TI)
    on = advisor.advise("vllm", m, {"tensor_parallel_size": 2, "language_model_only": True}, DUAL_5060TI)
    assert off["budget"]["weights_gb"] == 8.0
    assert on["budget"]["weights_gb"] == 6.0
    assert on["budget"]["needed_gb"] == round(off["budget"]["needed_gb"] - 2.0, 1)


def test_vllm_text_only_unchanged_when_encoder_size_unknown():
    m = safetensors_model(size_gb=8.0, params=8.0, multimodal=True)
    a = advisor.advise("vllm", m, {"tensor_parallel_size": 2, "language_model_only": True}, DUAL_5060TI)
    assert a["budget"]["weights_gb"] == 8.0


def test_llamacpp_never_picks_the_mmproj_file_as_the_model(tmp_path):
    # Sorted by name, a lowercase "mmproj-F16.gguf" comes before "qwen-Q4_K_M.gguf".
    mmproj = str(tmp_path / "mmproj-F16.gguf")
    path = _write_gguf(tmp_path / "qwen-Q4_K_M.gguf", QWEN35_27B)
    m = _gguf_file_model(mmproj, 1.0, [(path, 10.0)])
    a = advisor.advise("llamacpp", m, {"ctx_size": 8192}, DUAL_5060TI)
    assert a["budget"]["weights_gb"] == 10.0


GEMMA4_12B = {
    # Real header: five sliding-window layers (8 KV heads, width 256, 1,024-token window)
    # for every global layer (1 KV head, width 512); per-layer values are arrays.
    "general.architecture": "gemma4", "gemma4.block_count": 48,
    "gemma4.attention.head_count": 16, "gemma4.embedding_length": 3840,
    "gemma4.attention.head_count_kv": (8, 8, 8, 8, 8, 1) * 8,
    "gemma4.attention.sliding_window_pattern": (True, True, True, True, True, False) * 8,
    "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
    "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
    "gemma4.attention.sliding_window": 1024, "gemma4.attention.shared_kv_layers": 0,
}


def test_llamacpp_kv_from_gguf_sliding_window_and_per_layer_heads(tmp_path):
    path = _write_gguf(tmp_path / "gemma.gguf", GEMMA4_12B)
    a = advisor.advise("llamacpp", _gguf_file_model(path, 11.8),
                       {"ctx_size": 200000, "flash_attn": "auto", "ubatch_size": 512,
                        "cache_type_k": "q8_0", "cache_type_v": "q4_0"}, DUAL_5060TI)
    per_value = 34 / 32 + 18 / 32
    full = 8 * 1 * 512 * per_value * 200000
    # llama-server's default 4 slots share the cache, each keeping its own window.
    window = 40 * 8 * 256 * per_value * (1024 * 4 + 512)
    assert a["budget"]["kv_cache_gb"] == round((full + window) / GB, 1) == 1.8


def test_llamacpp_kv_shared_layers_keep_no_cache(tmp_path):
    meta = {"general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096, "llama.attention.shared_kv_layers": 16}
    path = _write_gguf(tmp_path / "shared.gguf", meta)
    a = advisor.advise("llamacpp", _gguf_file_model(path, 5.0), {"ctx_size": 8192}, DUAL_5060TI)
    assert a["budget"]["kv_cache_gb"] == round(16 * 8 * 128 * 4 * 8192 / GB, 1)


def test_llamacpp_draft_mtp_adds_its_own_cache_and_buffers(tmp_path):
    # llama.cpp's draft-MTP context has the same context length, caching only the MTP
    # layer at f16 (the draft cache type), plus its own working buffer on one card.
    path = _write_gguf(tmp_path / "qwen.gguf", QWEN35_27B)
    m = _gguf_file_model(path, 15.75)
    base = {"ctx_size": 200000, "flash_attn": "auto", "cache_type_k": "q8_0", "cache_type_v": "q4_0"}
    off = advisor.advise("llamacpp", m, base, DUAL_5060TI)["budget"]
    on = advisor.advise("llamacpp", m, {**base, "use_mtp": True}, DUAL_5060TI)["budget"]
    mtp = (1 * 4 * 256 * 2 * 2 * 200000 + advisor._compute_bytes(200000, 512)) / GB
    assert off["mtp_gb"] == 0.0
    assert on["mtp_gb"] == round(mtp, 1)
    assert abs(on["needed_gb"] - (off["needed_gb"] + mtp)) <= 0.1
    raw = advisor.advise("llamacpp", m, {**base, "extra_args": "--spec-type draft-mtp --spec-draft-n-max 1"},
                         DUAL_5060TI)["budget"]
    assert raw["mtp_gb"] == on["mtp_gb"]


def test_llamacpp_draft_mtp_adds_nothing_without_mtp_layers(tmp_path):
    meta = {"general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096}
    path = _write_gguf(tmp_path / "llama.gguf", meta)
    a = advisor.advise("llamacpp", _gguf_file_model(path, 5.0), {"ctx_size": 8192, "use_mtp": True}, DUAL_5060TI)
    assert a["budget"]["mtp_gb"] == 0.0


def test_llamacpp_draft_mtp_counts_a_separate_head_file(tmp_path):
    path = _write_gguf(tmp_path / "gemma-Q8_0.gguf", GEMMA4_12B)
    head_meta = {"general.architecture": "gemma4-assistant", "gemma4-assistant.block_count": 2,
                 "gemma4-assistant.attention.head_count": 16, "gemma4-assistant.attention.head_count_kv": 8,
                 "gemma4-assistant.embedding_length": 2048}
    head = _write_gguf(tmp_path / "mtp-gemma.gguf", head_meta)
    m = _gguf_file_model(path, 11.8, [(head, 0.5)])
    a = advisor.advise("llamacpp", m, {"ctx_size": 8192, "use_mtp": True}, DUAL_5060TI)["budget"]
    head_kv = 2 * 8 * 128 * 4 * 8192 / GB
    assert a["weights_gb"] == 11.8
    assert a["mtp_gb"] == round(0.5 + head_kv + advisor._compute_bytes(8192, 512) / GB, 1)


def test_llamacpp_draft_mtp_model_named_mtp_is_not_its_own_head(tmp_path):
    # The real file name "Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf" matches llama.cpp's
    # "mtp-" head rule, but it is the model being loaded, not a separate head.
    path = _write_gguf(tmp_path / "Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf", QWEN35_27B)
    a = advisor.advise("llamacpp", _gguf_file_model(path, 15.75),
                       {"ctx_size": 200000, "cache_type_k": "q8_0", "cache_type_v": "q4_0",
                        "flash_attn": "auto", "use_mtp": True}, DUAL_5060TI)
    mtp = (1 * 4 * 256 * 2 * 2 * 200000 + advisor._compute_bytes(200000, 512)) / GB
    assert a["budget"]["mtp_gb"] == round(mtp, 1) == 1.1


# ---------- llama.cpp buffers measured on gpuhost (2026-10-06) ----------
# llama.cpp build 1537a0a8, two RTX 5060 Ti, flash attention on, `--fit off -lv 4`;
# each figure is llama.cpp's own "buffer size" report, in MiB.

MB = 1024**2

QWEN35_27B_SSM = {**QWEN35_27B,
                  "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.state_size": 128,
                  "qwen35.ssm.group_count": 16, "qwen35.ssm.time_step_rank": 48,
                  "qwen35.ssm.inner_size": 6144}


def _buffers(tmp_path, meta, **cfg):
    path = _write_gguf(tmp_path / "m.gguf", meta)
    cfg = {"flash_attn": "on", **cfg}
    return advisor._llamacpp_buffers(path, cfg, devices=2)


def test_llamacpp_compute_buffer_grows_with_ubatch_like_llamacpp(tmp_path):
    q8 = {"ctx_size": 262144, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}
    for ubatch, measured_per_card in ((128, 1108.27), (512, 1360.28), (1024, 1696.30)):
        compute = _buffers(tmp_path, QWEN35_27B_SSM, ubatch_size=ubatch, **q8)["compute"] / MB
        assert 2 * measured_per_card <= compute <= 2 * measured_per_card * 1.06, ubatch
    short = _buffers(tmp_path, QWEN35_27B_SSM, ubatch_size=512, ctx_size=65536,
                     cache_type_k="q8_0", cache_type_v="q8_0")["compute"] / MB
    assert 2 * 400.28 <= short <= 2 * 400.28 * 1.15
    # An f16 cache needs no conversion copy, so Gemma's buffer is the micro-batch part only.
    gemma = _buffers(tmp_path, GEMMA4_12B, ubatch_size=512, ctx_size=131072)["compute"] / MB
    assert 2 * 253.52 <= gemma <= 2 * 253.52 * 1.05


def test_llamacpp_recurrent_state_is_counted_per_conversation_slot(tmp_path):
    q8 = {"ctx_size": 65536, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "ubatch_size": 512}
    default = _buffers(tmp_path, QWEN35_27B_SSM, **q8)  # llama-server opens 4 slots
    one = _buffers(tmp_path, QWEN35_27B_SSM, parallel=1, **q8)
    assert round(default["recurrent"] / MB, 1) == round(311.72 + 286.78, 1)
    assert round(one["recurrent"] / MB, 1) == round(77.93 + 71.70, 1)
    assert round(default["kv"] / MB) == 2 * 1088
    assert _buffers(tmp_path, QWEN35_27B, **q8)["recurrent"] == 0  # no ssm keys: no state


def test_llamacpp_sliding_window_cache_holds_a_window_per_slot(tmp_path):
    shared = {"ctx_size": 32768, "ubatch_size": 512}
    default = _buffers(tmp_path, GEMMA4_12B, **shared)
    assert round(default["kv"] / MB) == 256 + 256 + 756 + 684
    one = _buffers(tmp_path, GEMMA4_12B, parallel=1, **shared)
    full = 8 * 1 * 512 * 2 * 2 * 32768
    assert one["kv"] == full + 40 * 8 * 256 * 2 * 2 * 1536  # window + micro-batch, padded to 256


def test_llamacpp_draft_mtp_keeps_an_f16_cache_and_its_own_state(tmp_path):
    q8 = {"ctx_size": 262144, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "ubatch_size": 128}
    off = _buffers(tmp_path, QWEN35_27B_SSM, **q8)
    on = _buffers(tmp_path, QWEN35_27B_SSM, use_mtp=True, **q8)
    assert off["mtp"] == 0
    # Measured: the draft context's KV 1024 MiB (f16 although the main cache is q8_0),
    # a second copy of the recurrent state, and an 81 MiB compute buffer on one card.
    extra = on["mtp"] / MB
    assert 1024 + 598.5 + 81.0 <= extra <= (1024 + 598.5 + 81.0) * 1.05


def test_llamacpp_estimate_brackets_measured_gpu_use(tmp_path):
    # nvidia-smi totals across both cards after loading the real Qwen3.8 file
    # (16,912,387,392 bytes). The estimate counts the whole file although llama.cpp
    # keeps the token table (1.26 GB) in system RAM, so it should land a little above.
    path = _write_gguf(tmp_path / "Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf", QWEN35_27B_SSM)
    model = _gguf_file_model(path, 16912387392 / GB)
    for ubatch, mtp, measured_mib in ((128, False, 26246), (512, False, 26750),
                                      (1024, False, 27422), (128, True, 28378)):
        cfg = {"ctx_size": 262144, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
               "flash_attn": "on", "ubatch_size": ubatch, "use_mtp": mtp}
        needed = advisor.advise("llamacpp", model, cfg, DUAL_5060TI)["budget"]["needed_gb"]
        assert measured_mib / 1024 <= needed <= measured_mib / 1024 + 1.8, (ubatch, mtp, needed)


def test_llamacpp_failed_load_is_not_called_a_fit(tmp_path):
    # Measured: ubatch 512 with draft-MTP ran out of memory on the second card.
    path = _write_gguf(tmp_path / "Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf", QWEN35_27B_SSM)
    cfg = {"ctx_size": 262144, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
           "flash_attn": "on", "ubatch_size": 512, "use_mtp": True}
    a = advisor.advise("llamacpp", _gguf_file_model(path, 16912387392 / GB), cfg, DUAL_5060TI)
    assert a["overall"]["level"] == "red"


def test_llamacpp_near_full_warns_about_silent_cpu_offload(tmp_path):
    path = _write_gguf(tmp_path / "qwen.gguf", QWEN35_27B_SSM)
    cfg = {"ctx_size": 262144, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
           "flash_attn": "on", "ubatch_size": 512}
    tight = advisor.advise("llamacpp", _gguf_file_model(path, 16912387392 / GB), cfg, DUAL_5060TI)
    assert tight["budget"]["pct"] > 0.9
    assert any("CPU" in d and "slower" in d for d in tight["overall"]["details"])
    roomy = advisor.advise("llamacpp", _gguf_file_model(path, 16912387392 / GB),
                           {**cfg, "ctx_size": 32768}, DUAL_5060TI)
    assert not any("CPU" in d for d in roomy["overall"]["details"])


def test_llamacpp_unset_parallel_means_llama_servers_own_four_slots(tmp_path):
    # The launcher sends --parallel only when the user sets it; the catalog's displayed
    # 1 is then not what runs, so the estimate must not use it.
    path = _write_gguf(tmp_path / "qwen.gguf", QWEN35_27B_SSM)
    cfg = {"ctx_size": 65536, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"}
    unset = advisor.advise("llamacpp", _gguf_file_model(path, 15.75), cfg, DUAL_5060TI)["budget"]
    one = advisor.advise("llamacpp", _gguf_file_model(path, 15.75), {**cfg, "parallel": 1}, DUAL_5060TI)["budget"]
    assert unset["recurrent_state_gb"] == 0.6
    assert one["recurrent_state_gb"] == 0.1

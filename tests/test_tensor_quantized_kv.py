"""Regression: tensor splitting and quantized KV are not mutually exclusive."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_llm_launcher import advisor, api
from local_llm_launcher.app import create_app
from local_llm_launcher.engines import llamacpp
from tests.test_engines import GGUF
from tests.test_advisor import DUAL_5060TI, safetensors_model


@pytest.mark.parametrize("weight_quant", ["Q5_K_M", "NVFP4"])
@pytest.mark.parametrize("flash_attn", ["on", "auto"])
@pytest.mark.parametrize("k,v", [("q8_0", "q8_0"), ("q8_0", "f16"),
                                 ("f16", "q8_0"), ("q4_0", "q4_0")])
def test_tensor_quantized_cache_preserved_in_command_and_advice(weight_quant, flash_attn, k, v):
    model = deepcopy(GGUF)
    model["quant"] = weight_quant
    model["gguf_files"][0]["quant"] = weight_quant
    config = {"split_mode": "tensor", "tensor_split": "1,1", "cache_type_k": k,
              "cache_type_v": v, "flash_attn": flash_attn}
    argv = llamacpp.build(model, config)["argv"]
    for flag, value in [("--split-mode", "tensor"), ("--tensor-split", "1,1"),
                        ("--cache-type-k", k), ("--cache-type-v", v),
                        ("--flash-attn", flash_attn)]:
        assert argv[argv.index(flag) + 1] == value
    report = advisor.advise("llamacpp", model, config, DUAL_5060TI)
    assert report["overall"]["level"] == "yellow"
    assert report["budget"]["fit_unknown"] is True
    warning = report["flags"]["split_mode"]
    assert warning["level"] == "yellow"
    assert "#23792" in warning["message"] and "Older builds" in warning["message"]
    assert "NVFP4" in warning["message"]


@pytest.mark.parametrize("cache", ["f16", "bf16", "f32", "q8_0", "q4_0"])
def test_tensor_still_requires_flash_attention(cache):
    config = {"split_mode": "tensor", "cache_type_k": cache,
              "cache_type_v": cache, "flash_attn": "off"}
    with pytest.raises(ValueError, match="flash attention"):
        llamacpp.build(GGUF, config)
    with pytest.raises(ValueError, match="flash attention"):
        advisor.advise("llamacpp", GGUF, config, DUAL_5060TI)


@pytest.mark.parametrize("cache", ["f16", "bf16", "f32"])
def test_full_precision_tensor_remains_available(cache):
    config = {"split_mode": "tensor", "cache_type_k": cache,
              "cache_type_v": cache, "flash_attn": "on"}
    assert "--split-mode" in llamacpp.build(GGUF, config)["argv"]
    report = advisor.advise("llamacpp", GGUF, config, DUAL_5060TI)
    assert "#23792" not in report["flags"]["split_mode"]["message"]


def test_nvfp4_does_not_make_safetensors_loadable_in_llamacpp():
    model = safetensors_model(size_gb=5.0, params=8.0, quant="NVFP4")
    report = advisor.advise("llamacpp", model,
        {"split_mode": "tensor", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"},
        DUAL_5060TI)
    assert report["overall"]["level"] == "red"
    assert "safetensors" in report["overall"]["headline"]


@pytest.mark.parametrize("weight_quant", ["Q5_K_M", "NVFP4"])
def test_api_advice_accepts_quantized_tensor_and_launch_passes_original_config(monkeypatch, weight_quant):
    model = deepcopy(GGUF); model["quant"] = weight_quant
    config = {"split_mode": "tensor", "tensor_split": "1,1", "cache_type_k": "q8_0",
              "cache_type_v": "q8_0", "flash_attn": "on"}
    monkeypatch.setattr(api, "find_model", lambda _: model)
    monkeypatch.setattr(api, "get_hardware", lambda: DUAL_5060TI)
    captured = {}
    def launch(engine_mode, actual_model, actual_config, **kwargs):
        captured.update(engine_mode=engine_mode, model=actual_model, config=actual_config)
        llamacpp.build(actual_model, actual_config)
        return SimpleNamespace(status=lambda: {"id": "synthetic-server", "status": "starting"})
    monkeypatch.setattr(api.servers, "launch", launch)
    client = TestClient(create_app(), base_url="http://127.0.0.1")
    response = client.post("/api/advise", json={"engine": "llamacpp", "repo_id": model["repo_id"], "config": config})
    assert response.status_code == 200, response.text
    assert response.json()["overall"]["level"] == "yellow"
    response = client.post("/api/servers", json={"engine_mode": "llamacpp", "repo_id": model["repo_id"], "config": config})
    assert response.status_code == 200, response.text
    assert captured["config"] == {**config, "host": "127.0.0.1"}
    assert captured["engine_mode"] == "llamacpp"

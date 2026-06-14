"""Tests for the flag catalog loader."""
from local_llm_launcher import catalog


def test_load_both_catalogs():
    for engine in ("vllm", "llamacpp", "sglang"):
        cat = catalog.load_catalog(engine)
        assert cat["engine"] == engine
        assert len(cat["flags"]) >= 10


def test_every_flag_has_required_fields():
    for engine in ("vllm", "llamacpp", "sglang"):
        for f in catalog.load_catalog(engine)["flags"]:
            assert f["key"], f
            assert f["label"], f
            assert f["type"] in ("int", "float", "bool", "choice", "string"), f
            assert len(f["help"]) > 30, f"help too short for {f['key']}"
            assert f["category"] in ("essential", "performance", "api", "advanced"), f
            if f["type"] == "choice":
                assert "choices" in f and len(f["choices"]) >= 2, f


def test_defaults_returns_flat_config():
    d = catalog.defaults("vllm")
    assert d["port"] == 8000
    assert d["gpu_memory_utilization"] == 0.85
    assert "quantization" not in d or d["quantization"] is None


def test_unknown_engine_raises():
    import pytest
    with pytest.raises(ValueError):
        catalog.load_catalog("ollama")

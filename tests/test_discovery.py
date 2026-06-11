"""Tests for installed-model discovery against a fake HuggingFace cache layout."""
import json

from local_llm_launcher import discovery


def make_hf_model(hub_dir, org, name, files, config=None):
    snap = hub_dir / f"models--{org}--{name}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for fname, size in files.items():
        (snap / fname).write_bytes(b"\0" * size)
    if config is not None:
        (snap / "config.json").write_text(json.dumps(config))
    return snap


def test_scan_hf_cache_safetensors(tmp_path):
    hub = tmp_path / "hub"
    make_hf_model(
        hub, "meta-llama", "Llama-3.1-8B-Instruct",
        {"model-00001-of-00002.safetensors": 5000, "model-00002-of-00002.safetensors": 3000,
         "tokenizer.json": 100},
        config={"num_hidden_layers": 32, "hidden_size": 4096,
                "num_key_value_heads": 8, "num_attention_heads": 32,
                "torch_dtype": "bfloat16"},
    )
    models = discovery.scan_hf_cache(hub)
    assert len(models) == 1
    m = models[0]
    assert m.repo_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert m.format == "safetensors"
    assert m.size_bytes == 8000
    assert m.quant is None
    assert m.config["num_hidden_layers"] == 32


def test_scan_hf_cache_detects_quant_from_config(tmp_path):
    hub = tmp_path / "hub"
    make_hf_model(
        hub, "cyankiwi", "Qwen3.6-27B-AWQ-INT4",
        {"model.safetensors": 1000},
        config={"quantization_config": {"quant_method": "awq"}},
    )
    models = discovery.scan_hf_cache(hub)
    assert models[0].quant == "awq"


def test_scan_hf_cache_gguf_repo(tmp_path):
    hub = tmp_path / "hub"
    make_hf_model(
        hub, "bartowski", "Llama-3.1-8B-Instruct-GGUF",
        {"Llama-3.1-8B-Instruct-Q4_K_M.gguf": 4000,
         "Llama-3.1-8B-Instruct-Q8_0.gguf": 8000},
    )
    models = discovery.scan_hf_cache(hub)
    assert len(models) == 1
    m = models[0]
    assert m.format == "gguf"
    assert len(m.gguf_files) == 2
    quants = {f["quant"] for f in m.gguf_files}
    assert quants == {"Q4_K_M", "Q8_0"}


def test_scan_hf_cache_skips_datasets_and_junk(tmp_path):
    hub = tmp_path / "hub"
    (hub / "datasets--nguha--legalbench").mkdir(parents=True)
    (hub / "CACHEDIR.TAG").parent.mkdir(parents=True, exist_ok=True)
    (hub / "CACHEDIR.TAG").write_text("x")
    assert discovery.scan_hf_cache(hub) == []


def test_scan_hf_cache_missing_dir():
    assert discovery.scan_hf_cache("/nonexistent/path") == []


def test_scan_gguf_folder(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "mistral-7b-instruct-Q5_K_M.gguf").write_bytes(b"\0" * 5000)
    (folder / "notes.txt").write_text("hi")
    models = discovery.scan_gguf_folder(folder)
    assert len(models) == 1
    assert models[0].format == "gguf"
    assert models[0].gguf_files[0]["quant"] == "Q5_K_M"
    assert models[0].size_bytes == 5000


def test_guess_quant_from_filename():
    assert discovery.guess_gguf_quant("foo-Q4_K_M.gguf") == "Q4_K_M"
    assert discovery.guess_gguf_quant("foo.IQ2_XS.gguf") == "IQ2_XS"
    assert discovery.guess_gguf_quant("foo-f16.gguf") == "F16"
    assert discovery.guess_gguf_quant("mystery.gguf") is None


def test_is_multimodal_detects_conditional_generation():
    assert discovery.is_multimodal_config(
        {"architectures": ["Gemma4ForConditionalGeneration"]}
    )


def test_is_multimodal_detects_vision_config():
    assert discovery.is_multimodal_config(
        {"architectures": ["SomeForCausalLM"], "vision_config": {"hidden_size": 1024}}
    )


def test_is_multimodal_detects_audio_and_image_token():
    assert discovery.is_multimodal_config({"audio_config": {}})
    assert discovery.is_multimodal_config({"image_token_id": 256000})


def test_is_multimodal_false_for_text_model():
    assert not discovery.is_multimodal_config(
        {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 32}
    )
    assert not discovery.is_multimodal_config({})


def test_scan_marks_multimodal_model(tmp_path):
    hub = tmp_path / "hub"
    make_hf_model(
        hub, "google", "gemma-4-12B",
        {"model.safetensors": 1000},
        config={"architectures": ["Gemma4ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152}, "text_config": {}},
    )
    models = discovery.scan_hf_cache(hub)
    assert models[0].multimodal is True
    assert models[0].to_dict()["multimodal"] is True


def test_scan_text_model_not_multimodal(tmp_path):
    hub = tmp_path / "hub"
    make_hf_model(
        hub, "meta-llama", "Llama-3.1-8B",
        {"model.safetensors": 1000},
        config={"architectures": ["LlamaForCausalLM"]},
    )
    assert discovery.scan_hf_cache(hub)[0].multimodal is False


def test_guess_param_count():
    assert discovery.guess_param_count_b("meta-llama/Llama-3.1-8B-Instruct") == 8.0
    assert discovery.guess_param_count_b("Qwen3.6-27B-AWQ") == 27.0
    assert discovery.guess_param_count_b("Qwen3.6-35B-A3B-AWQ-4bit") == 35.0  # MoE total, not active
    assert discovery.guess_param_count_b("gemma-2-2b-it") == 2.0
    assert discovery.guess_param_count_b("no-size-here") is None

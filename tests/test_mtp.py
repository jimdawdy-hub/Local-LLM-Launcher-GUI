"""One MTP checkbox becomes whatever the installed engine accepts, or a clear update message."""
import json
import subprocess
from pathlib import Path
import venv

import pytest
from fastapi.testclient import TestClient

from local_llm_launcher import api
from local_llm_launcher.app import create_app
from local_llm_launcher.engines import mtp, vllm_capabilities, vllm_docker, vllm_native

DEEPSEEK = {"repo_id": "deepseek-ai/DeepSeek-V3", "format": "safetensors", "size_bytes": 1024**3,
            "param_count_b": 1, "config": {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"],
                                           "num_nextn_predict_layers": 1}}
QWEN_NEXT = {**DEEPSEEK, "config": {"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"],
                                    "num_nextn_predict_layers": 1}}
GLM = {**DEEPSEEK, "config": {"model_type": "glm4_moe", "architectures": ["Glm4MoeForCausalLM"],
                              "num_nextn_predict_layers": 1}}
# Qwen3.5 keeps its MTP layer count in the text part of a multimodal config.
QWEN35 = {**DEEPSEEK, "config": {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                                 "text_config": {"model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 2}}}
LLAMA = {**DEEPSEEK, "config": {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}}

FLAGS = ["--model", "--dtype", "--speculative-config"]
# Shapes read from the real vLLM sources of each release.
V030 = {"version": "0.30.0", "flags": FLAGS,
        "mtp": {"methods": ["deepseek_mtp", "mtp", "qwen3_5_mtp", "qwen3_next_mtp"],
                "models": ["deepseek_v3", "qwen3_5", "qwen3_5_moe", "qwen3_next", "Glm4MoeForCausalLM"]}}
V0102 = {"version": "0.10.2", "flags": FLAGS,
         "mtp": {"methods": ["deepseek_mtp", "ernie_mtp", "qwen3_next_mtp"],
                 "models": ["deepseek_v3", "Glm4MoeForCausalLM", "MiMoForCausalLM", "ernie4_5_moe", "qwen3_next"]}}
V092 = {"version": "0.9.2", "flags": FLAGS,
        "mtp": {"methods": ["deepseek_mtp"], "models": ["deepseek_v3", "MiMoForCausalLM"]}}
V085 = {"version": "0.8.5", "flags": FLAGS, "mtp": {"methods": [], "models": ["deepseek_v3"]}}
ON = {"use_mtp": True}


def spec(result):
    assert result["args"][0] == "--speculative-config"
    return json.loads(result["args"][1])


@pytest.mark.parametrize("model, evidence, method", [
    (DEEPSEEK, V030, "mtp"), (QWEN_NEXT, V030, "mtp"), (GLM, V030, "mtp"),
    # Before the generic name, each family had its own; GLM and MiMo reused DeepSeek's.
    (DEEPSEEK, V0102, "deepseek_mtp"), (QWEN_NEXT, V0102, "qwen3_next_mtp"),
    (GLM, V0102, "deepseek_mtp"), (DEEPSEEK, V092, "deepseek_mtp"),
])
def test_method_follows_installed_runtime(model, evidence, method):
    result = mtp.vllm("vllm-native", model, ON, evidence)
    assert result["level"] == "green"
    assert spec(result) == {"method": method, "num_speculative_tokens": 1}


def test_layer_count_read_from_multimodal_text_config():
    assert spec(mtp.vllm("vllm-native", QWEN35, ON, V030))["num_speculative_tokens"] == 2


@pytest.mark.parametrize("model, evidence, expected", [
    (QWEN_NEXT, V092, "vLLM 0.9.2 does not support MTP for qwen3_next models"),
    (QWEN35, V0102, "vLLM 0.10.2 does not support MTP for qwen3_5_moe models"),
    (DEEPSEEK, V085, "vLLM 0.8.5 cannot run MTP"),
    (DEEPSEEK, {**V030, "flags": ["--model", "--dtype"]}, "vLLM 0.30.0 has no --speculative-config option"),
])
def test_old_runtime_is_refused_with_update_advice(model, evidence, expected):
    result = mtp.vllm("vllm-native", model, ON, evidence)
    assert result["level"] == "red" and result["args"] == []
    assert result["message"].startswith(expected)
    assert "pip install -U vllm" in result["message"]
    assert result["message"].endswith("Or turn off MTP to run the model without it.")


def test_model_without_mtp_layers_is_refused():
    assert "does not list any MTP layers" in mtp.vllm("vllm-native", LLAMA, ON, V030)["message"]
    zero = {**DEEPSEEK, "config": {**DEEPSEEK["config"], "num_nextn_predict_layers": 0}}
    assert mtp.vllm("vllm-native", zero, ON, V030)["level"] == "red"


def test_unreadable_runtime_assumes_generic_name_and_says_so():
    result = mtp.vllm("vllm-docker", DEEPSEEK, ON, {"version": None, "flags": None, "mtp": None})
    assert result["level"] == "yellow"
    assert spec(result)["method"] == "mtp"
    assert "0.11 or newer" in result["message"]
    red = mtp.vllm("vllm-docker", DEEPSEEK, ON, {**V085, "version": None})
    assert "docker pull vllm/vllm-openai:latest. Or turn off MTP" in red["message"]


def test_raw_speculative_config_is_noted_and_off_means_nothing():
    result = mtp.vllm("vllm-native", DEEPSEEK, {**ON, "extra_args": "--speculative_config={}"}, V030)
    assert result["level"] == "yellow" and "take priority" in result["message"]
    assert mtp.vllm("vllm-native", DEEPSEEK, {"use_mtp": False}, V030) == {"level": "green", "message": "", "args": []}


@pytest.mark.parametrize("builder", [vllm_native.build, vllm_docker.build])
def test_builders_place_mtp_before_raw_flags(builder):
    argv = builder(DEEPSEEK, {"use_mtp": True, "_mtp_args": ["--speculative-config", "{}"],
                              "extra_args": "--seed 1"})["argv"]
    assert argv.index("--speculative-config") < argv.index("--seed")
    assert "--speculative-config" not in builder(DEEPSEEK, {"use_mtp": True})["argv"]


def test_probe_reads_mtp_names_from_source_without_importing(tmp_path):
    environment = tmp_path / "engine"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    site = Path(subprocess.check_output([str(python), "-I", "-c",
                                         'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    (site / "vllm-0.30.0.dist-info").mkdir()
    (site / "vllm-0.30.0.dist-info/METADATA").write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.30.0\n")
    (site / "vllm/config").mkdir(parents=True)
    (site / "vllm/__init__.py").write_text('raise RuntimeError("vllm must not be imported")\n')
    (site / "vllm/config/speculative.py").write_text(
        'raise RuntimeError("must not import")\n'
        'MTPModelTypes = Literal[\n    "deepseek_mtp",\n    "mtp",\n]\n'
        'class SpeculativeConfig:\n'
        '    @staticmethod\n'
        '    def hf_config_override(hf_config):\n'
        '        if hf_config.model_type == "deepseek_v3":\n'
        '            hf_config.model_type = "deepseek_mtp"\n'
        '        return hf_config\n\n'
        '    def other(self):\n        return "not_a_model"\n')
    binary = environment / "bin/vllm"
    binary.write_text(f'#!{python}\nprint("--model --dtype --speculative-config")\n')
    binary.chmod(0o755)
    vllm_capabilities._cache.clear()
    result = vllm_capabilities.probe("vllm-native", str(binary))
    assert result["mtp"] == {"methods": ["deepseek_mtp", "mtp"], "models": ["deepseek_mtp", "deepseek_v3"]}
    assert "--speculative-config" in result["flags"]


@pytest.fixture
def client(monkeypatch):
    hw = {"gpus": [{"index": 0, "name": "RTX 5060 Ti", "compute_capability": "12.0",
                    "vram_total_mb": 16384, "vram_free_mb": 15000}],
          "engines": {"vllm_native": True, "vllm_docker": True}, "ram_gb": 64, "cpu_cores": 32, "apple_silicon": None}
    monkeypatch.setattr(api, "find_model", lambda _: QWEN_NEXT)
    monkeypatch.setattr(api, "get_hardware", lambda: hw)
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_advice_and_launch_share_the_decision(client, monkeypatch):
    monkeypatch.setattr(vllm_capabilities, "probe", lambda *a, **kw: V092)
    body = {"engine": "vllm", "engine_mode": "vllm-native", "repo_id": "x", "config": ON}
    advice = client.post("/api/advise", json=body).json()
    assert advice["overall"]["level"] == "red"
    assert advice["flags"]["use_mtp"]["level"] == "red"
    assert advice["overall"].get("override") is not True  # launch refuses, so no warned button
    launch = client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "x", "config": ON})
    assert launch.status_code == 400
    assert launch.json()["detail"] == advice["overall"]["headline"]

    seen = {}
    monkeypatch.setattr(vllm_capabilities, "probe", lambda *a, **kw: V0102)
    monkeypatch.setattr(api.servers, "build_spec", lambda _mode, _model, config, *a: seen.update(config) or (_ for _ in ()).throw(RuntimeError("stop")))
    assert client.post("/api/advise", json=body).json()["flags"]["use_mtp"]["level"] == "green"
    client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "x", "config": ON})
    assert json.loads(seen["_mtp_args"][1])["method"] == "qwen3_next_mtp"


# ---------------------------------------------------------------- llama.cpp

import struct

from local_llm_launcher import hardware
from local_llm_launcher.engines import llamacpp


def write_gguf(path, arch, blocks=4, nextn=1, tensors=None):
    """A header-only GGUF v3, laid out as llama.cpp's gguf-py writes it."""
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    kvs = [(f'general.architecture', 8, string(arch)),
           (f'{arch}.block_count', 4, struct.pack('<I', blocks)),
           # A string array like the tokenizer's, which the reader must skip.
           ('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, 3) + b''.join(string(t) for t in ('a', 'bb', 'ccc')))]
    if nextn is not None:
        kvs.append((f'{arch}.nextn_predict_layers', 4, struct.pack('<I', nextn)))
    if tensors is None:
        tensors = ['token_embd.weight', f'blk.{blocks - 1}.nextn.eh_proj.weight'] if nextn else ['token_embd.weight']
    body = b''.join(string(k) + struct.pack('<I', kind) + value for k, kind, value in kvs)
    body += b''.join(string(t) + struct.pack('<I', 2) + struct.pack('<QQ', 8, 4) + struct.pack('<IQ', 1, 0) for t in tensors)
    Path(path).write_bytes(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kvs)) + body)
    return str(path)


def gguf_model(*paths):
    files = [{"filename": Path(p).name, "path": str(p), "size_bytes": 1, "quant": None} for p in paths]
    return {"repo_id": "org/m-GGUF", "path": str(Path(paths[0]).parent), "format": "gguf",
            "size_bytes": 1, "config": {}, "gguf_files": files, "param_count_b": 8.0}


NEW = {"mtp": True, "build": 11235}


def test_help_and_version_probe_real_output_formats(tmp_path, monkeypatch):
    from types import SimpleNamespace
    help_text = ('----- speculative params -----\n\n--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,ngram-cache\n'
                 '                                        comma-separated list of types of speculative decoding to use (default: none)\n')
    outputs = {'--help': (help_text, ''), '--list-devices': ('', ''),
               '--version': ('', 'version: 0.5.0-dev (build 11235, commit 6c7a87f7e)\nbuilt with cc for x86_64\n')}
    monkeypatch.setattr(hardware.subprocess, 'run', lambda argv, **kw: SimpleNamespace(
        returncode=0, stdout=outputs[argv[-1]][0], stderr=outputs[argv[-1]][1]))
    binary = tmp_path / 'llama-server'
    binary.write_text('x')
    hardware._llama_capabilities_cached.cache_clear()
    assert hardware.llama_capabilities(str(binary)) | {'devices': []} == {'load_mode': False, 'devices': [], 'mtp': True, 'build': 11235}
    outputs['--help'] = ('--spec-type none,draft-simple,draft-eagle3,ngram-simple\n', '')
    outputs['--version'] = ('', 'version: 9131 (634275f)\n')
    binary.write_text('older')
    hardware._llama_capabilities_cached.cache_clear()
    assert hardware.llama_capabilities(str(binary))['mtp'] is False
    assert hardware.llama_capabilities(str(binary))['build'] == 9131
    # A shallow clone counts few commits; that build number means nothing.
    outputs['--help'] = (help_text, '')
    outputs['--version'] = ('', 'version: 0.5.0-dev (build 50, commit abc)\n')
    binary.write_text('shallow')
    hardware._llama_capabilities_cached.cache_clear()
    assert hardware.llama_capabilities(str(binary))['build'] is None


def test_embedded_mtp_layers_enable_draft_mtp(tmp_path):
    path = write_gguf(tmp_path / 'Qwen3.5-Q4_K_M.gguf', 'qwen35moe')
    result = mtp.llamacpp(gguf_model(path), ON, NEW)
    assert result['level'] == 'green'
    assert result['args'] == ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']


def test_sharded_model_finds_mtp_layers_in_a_later_split(tmp_path):
    first = write_gguf(tmp_path / 'm-00001-of-00002.gguf', 'qwen3next', tensors=['token_embd.weight'])
    write_gguf(tmp_path / 'm-00002-of-00002.gguf', 'qwen3next', tensors=['blk.3.nextn.eh_proj.weight'])
    assert mtp.llamacpp(gguf_model(first), ON, NEW)['level'] == 'green'


@pytest.mark.parametrize("capabilities, arch, expected", [
    ({"mtp": False, "build": 8000}, 'qwen35', 'This llama.cpp (build 8000) has no MTP support'),
    ({"mtp": True, "build": 10000}, 'glm4moe', 'This llama.cpp (build 10000) predates MTP support for glm4moe models, which arrived in build 10603'),
    # The layers exist, but llama.cpp has no MTP graph for this architecture.
    (NEW, 'glm4', "llama.cpp can't use MTP with glm4 models yet"),
])
def test_llama_refuses_with_update_advice(tmp_path, capabilities, arch, expected):
    result = mtp.llamacpp(gguf_model(write_gguf(tmp_path / 'm.gguf', arch)), ON, capabilities)
    assert result['level'] == 'red' and result['args'] == []
    assert result['message'].startswith(expected) and 'Update llama.cpp' in result['message']
    assert result['message'].endswith('Or turn off MTP to run the model without it.')


def test_newer_build_than_checked_is_only_a_warning(tmp_path):
    result = mtp.llamacpp(gguf_model(write_gguf(tmp_path / 'm.gguf', 'glm4')), ON, {"mtp": True, "build": 12000})
    assert result['level'] == 'yellow' and result['args'][:2] == ['--spec-type', 'draft-mtp']


def test_llama_model_without_mtp_layers_is_refused(tmp_path):
    for name, nextn, tensors in (('plain.gguf', None, None), ('no-tensor.gguf', 1, ['token_embd.weight'])):
        result = mtp.llamacpp(gguf_model(write_gguf(tmp_path / name, 'qwen35', nextn=nextn, tensors=tensors)), ON, NEW)
        assert result['level'] == 'red' and 'no MTP layers' in result['message'] and 'Update' not in result['message']


def test_separate_head_file_is_passed_and_never_picked_as_the_model(tmp_path):
    main = write_gguf(tmp_path / 'gemma-4-Q4_K_M.gguf', 'gemma4', nextn=None)
    assert 'separate file' in mtp.llamacpp(gguf_model(main), ON, NEW)['message']
    head = write_gguf(tmp_path / 'mtp-gemma-4-Q4_K_M.gguf', 'gemma4-assistant', nextn=None)
    write_gguf(tmp_path / 'mtp-gemma-4-Q8_0.gguf', 'gemma4-assistant', nextn=None)
    model = gguf_model(head, main)
    assert llamacpp.pick_gguf_path(model, {}) == main
    result = mtp.llamacpp(model, ON, NEW)
    assert result['level'] == 'green'
    assert result['args'][-2:] == ['--spec-draft-model', head]


def test_llama_builder_places_mtp_before_raw_flags(tmp_path):
    path = write_gguf(tmp_path / 'm.gguf', 'qwen35')
    argv = llamacpp.build(gguf_model(path), {"_mtp_args": ["--spec-type", "draft-mtp"], "extra_args": "--seed 1"},
                          binary="llama-server")["argv"]
    assert argv.index("--spec-type") < argv.index("--seed")


def test_llama_advice_uses_installed_binary(tmp_path, monkeypatch):
    model = gguf_model(write_gguf(tmp_path / 'm.gguf', 'glm4moe'))
    hw = {"gpus": [], "engines": {"vllm_native": False, "vllm_docker": False, "llamacpp_path": "/x/llama-server"},
          "ram_gb": 64, "cpu_cores": 32, "apple_silicon": None}
    monkeypatch.setattr(api, "find_model", lambda _: model)
    monkeypatch.setattr(api, "get_hardware", lambda: hw)
    seen = []
    monkeypatch.setattr(hardware, "llama_capabilities", lambda binary: seen.append(binary) or {"mtp": True, "build": 10000})
    client = TestClient(create_app(), base_url="http://127.0.0.1")
    advice = client.post("/api/advise", json={"engine": "llamacpp", "repo_id": "x", "config": ON}).json()
    assert seen == ["/x/llama-server"]
    assert advice["overall"]["level"] == "red" and "10603" in advice["flags"]["use_mtp"]["message"]


def test_head_choice_prefers_same_folder_then_same_quant(tmp_path):
    # 'zz' sorts after 'mtp-a', so only the folder preference picks the nearby head.
    (tmp_path / 'zz').mkdir()
    main = write_gguf(tmp_path / 'zz/model-Q4_K_M.gguf', 'gemma4', nextn=None)
    elsewhere = write_gguf(tmp_path / 'mtp-a-Q4_K_M.gguf', 'gemma4-assistant', nextn=None)
    other_quant = write_gguf(tmp_path / 'zz/mtp-a-Q8_0.gguf', 'gemma4-assistant', nextn=None)
    same_quant = write_gguf(tmp_path / 'zz/mtp-b-Q4_K_M.gguf', 'gemma4-assistant', nextn=None)
    model = gguf_model(main, elsewhere, other_quant, same_quant)
    assert mtp.llamacpp(model, {**ON, "gguf_file": "model-Q4_K_M.gguf"}, NEW)['args'][-1] == same_quant
    Path(same_quant).unlink()
    Path(other_quant).unlink()
    assert mtp.llamacpp(gguf_model(main, elsewhere), ON, NEW)['args'][-1] == elsewhere


def test_old_llama_with_plain_model_is_told_to_turn_mtp_off(tmp_path):
    result = mtp.llamacpp(gguf_model(write_gguf(tmp_path / 'plain.gguf', 'llama', nextn=None)), ON, {"mtp": False, "build": 8000})
    assert 'no MTP layers' in result['message'] and 'Update' not in result['message']


# ------------------------------------------------ review fixes (PR #21 review)

def test_unrelated_head_in_a_shared_folder_is_ignored(tmp_path):
    # A flat folder: a Qwen3.5 file without MTP layers next to another model's head.
    main = write_gguf(tmp_path / 'Qwen3.5-9B-Q4_K_M.gguf', 'qwen35', nextn=None)
    write_gguf(tmp_path / 'mtp-gemma-4-31B-Q4_K_M.gguf', 'gemma4-assistant', nextn=None)
    result = mtp.llamacpp(gguf_model(main), ON, NEW)
    assert result['level'] == 'red' and result['args'] == []
    # A head converted from the same architecture is still used.
    own = write_gguf(tmp_path / 'mtp-Qwen3.5-9B-Q4_K_M.gguf', 'qwen35', nextn=1)
    assert mtp.llamacpp(gguf_model(main), ON, NEW)['args'][-2:] == ['--spec-draft-model', own]


def test_head_named_with_mtp_in_the_middle_is_found_and_not_picked_as_model(tmp_path):
    (tmp_path / 'Q4_K_M').mkdir()
    main = write_gguf(tmp_path / 'Q4_K_M/step-3.7-Q4_K_M.gguf', 'step35', nextn=None)
    head = write_gguf(tmp_path / 'step-3.7-mtp-Q8_0.gguf', 'step35', nextn=1)
    model = gguf_model(head, main)
    assert llamacpp.pick_gguf_path(model, {}) == main
    assert mtp.llamacpp(model, ON, NEW)['args'][-2:] == ['--spec-draft-model', head]


def test_raw_spec_type_replaces_the_launchers_own(tmp_path):
    path = write_gguf(tmp_path / 'm.gguf', 'qwen35')
    result = mtp.llamacpp(gguf_model(path), {**ON, 'extra_args': '--spec-type draft-mtp,ngram-mod'}, NEW)
    assert '--spec-type' not in result['args'] and result['level'] == 'yellow'
    assert 'include draft-mtp there' in result['message']
    last_wins = mtp.llamacpp(gguf_model(path), {**ON, 'extra_args': '--spec-draft-n-max 5'}, NEW)
    assert last_wins['args'][:2] == ['--spec-type', 'draft-mtp'] and 'take priority' in last_wins['message']


@pytest.mark.parametrize('corrupt', ['huge-string', 'huge-array', 'deep-nesting', 'truncated'])
def test_malformed_gguf_header_gives_a_red_verdict(tmp_path, corrupt):
    path = tmp_path / 'bad.gguf'
    head = b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + struct.pack('<Q', 1) + b'k'
    if corrupt == 'huge-string':
        body = struct.pack('<I', 8) + struct.pack('<Q', 1 << 62)
    elif corrupt == 'huge-array':
        body = struct.pack('<I', 9) + struct.pack('<IQ', 8, 1 << 60)
    elif corrupt == 'deep-nesting':
        body = struct.pack('<I', 9) + struct.pack('<IQ', 9, 1) * 5000
    else:
        body = struct.pack('<I', 4)
    path.write_bytes(head + body)
    result = mtp.llamacpp(gguf_model(str(path)), ON, NEW)
    assert result['level'] == 'red' and result['message'].startswith('Could not read the selected GGUF file')


@pytest.mark.parametrize('capabilities, note', [
    ({"mtp": True, "build": None}, 'Could not read the llama.cpp build number; MTP for glm4moe needs build 10603'),
    ({"mtp": None, "build": None}, 'Could not check the installed llama-server; MTP for glm4moe needs build 10603'),
])
def test_llama_partial_evidence_is_a_warning(tmp_path, capabilities, note):
    result = mtp.llamacpp(gguf_model(write_gguf(tmp_path / 'm.gguf', 'glm4moe')), ON, capabilities)
    assert result['level'] == 'yellow' and note in result['message']
    assert result['args'][:2] == ['--spec-type', 'draft-mtp']


def test_vllm_unknown_family_list_is_a_warning():
    evidence = {**V030, "mtp": {"methods": V030["mtp"]["methods"], "models": None}}
    result = mtp.vllm("vllm-native", DEEPSEEK, ON, evidence)
    assert result["level"] == "yellow" and spec(result)["method"] == "mtp"
    assert "Could not confirm this vLLM supports MTP for this model family" in result["message"]


def test_yellow_decision_reaches_advice_and_launch(client, monkeypatch):
    evidence = {**V030, "mtp": {"methods": V030["mtp"]["methods"], "models": None}}
    monkeypatch.setattr(vllm_capabilities, "probe", lambda *a, **kw: evidence)
    advice = client.post("/api/advise", json={"engine": "vllm", "engine_mode": "vllm-native",
                                              "repo_id": "x", "config": ON}).json()
    assert advice["flags"]["use_mtp"]["level"] == "yellow"
    assert advice["overall"]["level"] in ("yellow", "red")
    assert any("Could not confirm" in d for d in advice["overall"]["details"])
    seen = {}
    monkeypatch.setattr(api.servers, "build_spec", lambda _mode, _model, config, *a: seen.update(config) or (_ for _ in ()).throw(RuntimeError("stop")))
    client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "x", "config": ON})
    assert seen["_mtp_args"][0] == "--speculative-config"

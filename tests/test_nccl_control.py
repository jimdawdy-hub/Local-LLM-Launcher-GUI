"""Per-launch AllReduce selection uses the environment, not CLI flags."""
import pytest
from local_llm_launcher import catalog
from local_llm_launcher.engines import llamacpp
from local_llm_launcher.engines.base import LocalServer

MODEL = {"path": "/models/test.gguf", "gguf_files": []}

@pytest.mark.parametrize("mode", ["nccl", "internal"])
def test_explicit_mode_overrides_inherited_environment(mode, monkeypatch, tmp_path):
    monkeypatch.setenv("GGML_CUDA_ALLREDUCE", "nccl" if mode == "internal" else "internal")
    spec = llamacpp.build(MODEL, {"split_mode": "tensor", "cuda_allreduce": mode})
    assert spec["env"]["GGML_CUDA_ALLREDUCE"] == mode
    assert mode not in spec["argv"]
    assert "--cuda-allreduce" not in spec["argv"]
    # Exercise real process launch, including environment merge.
    import sys
    server = LocalServer(server_id="nccl-test", engine="llamacpp", model_label="test",
                         port=45140, argv=[sys.executable, "-c",
                         "import os; print(os.environ['GGML_CUDA_ALLREDUCE'], flush=True)"],
                         env=spec["env"], log_dir=tmp_path)
    server.start()
    server.process.wait(timeout=10)
    assert mode in server.log_path.read_text()

@pytest.mark.parametrize("config", [{}, {"cuda_allreduce": None}])
def test_automatic_preserves_engine_or_inherited_default(config):
    assert "GGML_CUDA_ALLREDUCE" not in llamacpp.build(MODEL, config)["env"]

@pytest.mark.parametrize("mode", ["none", "off", "false", False, 1])
def test_invalid_mode_refused(mode):
    with pytest.raises(ValueError, match="CUDA AllReduce"):
        llamacpp.build(MODEL, {"cuda_allreduce": mode})

def test_gui_catalog_has_both_modes_and_automatic():
    flag = next(f for f in catalog.load_catalog("llamacpp")["flags"] if f["key"] == "cuda_allreduce")
    assert flag["choices"] == [None, "nccl", "internal"]
    assert flag["flag"] is None
    assert flag["env_var"] == "GGML_CUDA_ALLREDUCE"

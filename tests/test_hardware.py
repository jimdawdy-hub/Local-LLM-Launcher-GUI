"""Tests for hardware detection. Parse functions are pure so we feed them canned output."""
from local_llm_launcher import hardware


NVIDIA_SMI_CSV = (
    "NVIDIA GeForce RTX 5060 Ti, 16311, 15090, 12.0, 0\n"
    "NVIDIA GeForce RTX 5060 Ti, 16311, 16100, 12.0, 1\n"
)


def test_parse_nvidia_smi_two_gpus():
    gpus = hardware.parse_nvidia_smi(NVIDIA_SMI_CSV)
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 5060 Ti"
    assert gpus[0].vram_total_mb == 16311
    assert gpus[0].vram_free_mb == 15090
    assert gpus[0].compute_capability == "12.0"
    assert gpus[0].index == 0
    assert gpus[1].index == 1


def test_parse_nvidia_smi_empty():
    assert hardware.parse_nvidia_smi("") == []


def test_parse_nvidia_smi_garbage_line_skipped():
    gpus = hardware.parse_nvidia_smi("not,a,gpu\n" + NVIDIA_SMI_CSV)
    assert len(gpus) == 2


def test_hardware_summary_dual_gpu():
    gpus = hardware.parse_nvidia_smi(NVIDIA_SMI_CSV)
    hw = hardware.Hardware(
        gpus=gpus, apple_silicon=None, cpu_cores=24, ram_gb=31.0, disk_free_gb=500.0,
        engines=hardware.EngineAvailability(vllm_native=False, vllm_docker=True, llamacpp_path=None),
    )
    assert hw.total_vram_mb == 32622
    s = hw.summary()
    assert "2x NVIDIA GeForce RTX 5060 Ti" in s
    assert "32 GB" in s  # total VRAM rounded


def test_hardware_summary_apple():
    hw = hardware.Hardware(
        gpus=[], apple_silicon=hardware.AppleSilicon(chip="Apple M3 Pro", memory_gb=36),
        cpu_cores=12, ram_gb=36.0, disk_free_gb=500.0,
        engines=hardware.EngineAvailability(vllm_native=False, vllm_docker=False, llamacpp_path="/opt/homebrew/bin/llama-server"),
    )
    s = hw.summary()
    assert "Apple M3 Pro" in s and "36 GB" in s


def test_hardware_summary_cpu_only():
    hw = hardware.Hardware(
        gpus=[], apple_silicon=None, cpu_cores=8, ram_gb=16.0, disk_free_gb=100.0,
        engines=hardware.EngineAvailability(vllm_native=False, vllm_docker=False, llamacpp_path=None),
    )
    assert "no GPU" in hw.summary().lower() or "cpu" in hw.summary().lower()


def test_to_dict_roundtrip():
    gpus = hardware.parse_nvidia_smi(NVIDIA_SMI_CSV)
    hw = hardware.Hardware(
        gpus=gpus, apple_silicon=None, cpu_cores=24, ram_gb=31.0, disk_free_gb=500.0,
        engines=hardware.EngineAvailability(vllm_native=True, vllm_docker=False, llamacpp_path=None),
    )
    d = hw.to_dict()
    assert d["gpus"][0]["name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert d["total_vram_mb"] == 32622
    assert d["engines"]["vllm_native"] is True
    assert isinstance(d["summary"], str)


def test_detect_hardware_runs_without_crashing():
    """Integration: on any machine this must return a Hardware object, never raise."""
    hw = hardware.detect_hardware()
    assert isinstance(hw, hardware.Hardware)
    assert hw.cpu_cores >= 1
    assert hw.ram_gb > 0

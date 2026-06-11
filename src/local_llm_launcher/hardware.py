"""Hardware detection: NVIDIA GPUs, Apple Silicon, CPU, RAM, disk, and engine availability.

GPU detection approach adapted from vllm-cli (https://github.com/Chen-zexi/vllm-cli)
by Chen-zexi, MIT license.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

_SMI_QUERY = "gpu_name,memory.total,memory.free,compute_cap,index"


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: str
    index: int


@dataclass
class AppleSilicon:
    chip: str
    memory_gb: float


@dataclass
class EngineAvailability:
    vllm_native: bool
    vllm_docker: bool
    llamacpp_path: Optional[str]


@dataclass
class Hardware:
    gpus: List[GpuInfo]
    apple_silicon: Optional[AppleSilicon]
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    engines: EngineAvailability
    notes: List[str] = field(default_factory=list)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpus)

    def summary(self) -> str:
        if self.gpus:
            names: Dict[str, int] = {}
            for g in self.gpus:
                names[g.name] = names.get(g.name, 0) + 1
            parts = [f"{n}x {name}" for name, n in names.items()]
            total_gb = round(self.total_vram_mb / 1024)
            ceiling = _model_ceiling_text(self.total_vram_mb)
            return (
                f"{', '.join(parts)} — {total_gb} GB of GPU memory (VRAM) total. {ceiling}"
            )
        if self.apple_silicon:
            a = self.apple_silicon
            return (
                f"{a.chip} with {a.memory_gb:.0f} GB unified memory. "
                f"llama.cpp with Metal acceleration is the recommended engine."
            )
        return (
            f"No GPU detected — CPU only ({self.cpu_cores} cores, {self.ram_gb:.0f} GB RAM). "
            f"Small models via llama.cpp will work, but expect slow generation."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpus": [vars(g) for g in self.gpus],
            "apple_silicon": vars(self.apple_silicon) if self.apple_silicon else None,
            "cpu_cores": self.cpu_cores,
            "ram_gb": round(self.ram_gb, 1),
            "disk_free_gb": round(self.disk_free_gb, 1),
            "total_vram_mb": self.total_vram_mb,
            "engines": vars(self.engines),
            "summary": self.summary(),
            "notes": self.notes,
        }


def _model_ceiling_text(total_vram_mb: int) -> str:
    gb = total_vram_mb / 1024
    if gb >= 70:
        return "Good for models up to ~70B at 4-bit quantization."
    if gb >= 44:
        return "Good for models up to ~70B at 4-bit quantization (tight) or ~35B comfortably."
    if gb >= 28:
        return "Good for models up to ~35B at 4-bit quantization."
    if gb >= 14:
        return "Good for models up to ~14B at 4-bit quantization."
    if gb >= 7:
        return "Good for models up to ~8B at 4-bit quantization."
    return "Best suited to small models (up to ~3B)."


def parse_nvidia_smi(output: str) -> List[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output."""
    gpus: List[GpuInfo] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    name=parts[0],
                    vram_total_mb=int(float(parts[1])),
                    vram_free_mb=int(float(parts[2])),
                    compute_capability=parts[3],
                    index=int(parts[4]),
                )
            )
        except ValueError:
            continue
    return gpus


def _detect_nvidia() -> List[GpuInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        return parse_nvidia_smi(out.stdout)
    except (OSError, subprocess.SubprocessError):
        return []


def _detect_apple_silicon() -> Optional[AppleSilicon]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    chip = "Apple Silicon"
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            chip = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    mem_gb = psutil.virtual_memory().total / (1024**3)
    return AppleSilicon(chip=chip, memory_gb=round(mem_gb))


def _vllm_native_available() -> bool:
    if shutil.which("vllm"):
        return True
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        return False


def _vllm_docker_available() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        out = subprocess.run(
            [docker, "image", "ls", "--format", "{{.Repository}}"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return False
        return any("vllm" in line for line in out.stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return False


_LLAMA_LOCATIONS = [
    "/usr/local/bin/llama-server",
    "/opt/homebrew/bin/llama-server",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
    os.path.expanduser("~/.local/bin/llama-server"),
]


def find_llamacpp(extra_path: Optional[str] = None) -> Optional[str]:
    if extra_path and os.path.isfile(extra_path) and os.access(extra_path, os.X_OK):
        return extra_path
    found = shutil.which("llama-server")
    if found:
        return found
    for candidate in _LLAMA_LOCATIONS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def detect_hardware(llamacpp_hint: Optional[str] = None) -> Hardware:
    """Detect everything. Must never raise — degrade gracefully on any failure."""
    notes: List[str] = []
    try:
        gpus = _detect_nvidia()
    except Exception:
        gpus = []
    apple = None
    try:
        apple = _detect_apple_silicon()
    except Exception:
        pass
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = 1.0
    try:
        disk_free_gb = shutil.disk_usage(os.path.expanduser("~")).free / (1024**3)
    except Exception:
        disk_free_gb = 0.0

    engines = EngineAvailability(
        vllm_native=_vllm_native_available(),
        vllm_docker=_vllm_docker_available(),
        llamacpp_path=find_llamacpp(llamacpp_hint),
    )
    if apple and (engines.vllm_native or engines.vllm_docker):
        notes.append("vLLM has no Apple Silicon GPU support — llama.cpp is recommended here.")

    return Hardware(
        gpus=gpus,
        apple_silicon=apple,
        cpu_cores=os.cpu_count() or 1,
        ram_gb=ram_gb,
        disk_free_gb=disk_free_gb,
        engines=engines,
        notes=notes,
    )

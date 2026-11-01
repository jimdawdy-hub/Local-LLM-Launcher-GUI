"""Hardware detection: NVIDIA GPUs, Apple Silicon, CPU, RAM, disk, and engine availability.

GPU detection approach adapted from vllm-cli (https://github.com/Chen-zexi/vllm-cli)
by Chen-zexi, MIT license.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
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
    sglang: bool
    tgi: bool = False


@dataclass
class Hardware:
    gpus: List[GpuInfo]
    apple_silicon: Optional[AppleSilicon]
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    engines: EngineAvailability
    notes: List[str] = field(default_factory=list)
    numa: Dict[str, Any] = field(default_factory=dict)
    llama_devices: List[Dict[str, str]] = field(default_factory=list)

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
            "numa": self.numa,
            "llama_devices": self.llama_devices,
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


def _sglang_available() -> bool:
    try:
        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):
        return False


def _tgi_available() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        out = subprocess.run([docker, "image", "inspect",
                              "ghcr.io/huggingface/text-generation-inference:3.3.7"],
                             capture_output=True, text=True, timeout=10)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# Source builds (usually CUDA/Metal-optimized) take priority over generic
# prebuilt binaries, which are often CPU-only.
_LLAMA_LOCATIONS = [
    os.path.expanduser("~/Projects/llama.cpp/build-cuda/bin/llama-server"),
    os.path.expanduser("~/Projects/llama.cpp/build/bin/llama-server"),
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
    "/usr/local/bin/llama-server",
    "/opt/homebrew/bin/llama-server",
    os.path.expanduser("~/.local/bin/llama-server"),
    os.path.expanduser("~/.local/opt/llama.cpp/llama-server"),
    os.path.expanduser("~/.local/opt/llama.cpp/bin/llama-server"),
]


def find_llamacpp(extra_path: Optional[str] = None) -> Optional[str]:
    if extra_path and os.path.isfile(extra_path) and os.access(extra_path, os.X_OK):
        return extra_path
    for candidate in _LLAMA_LOCATIONS[:3]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("llama-server")
    if found:
        return found
    for candidate in _LLAMA_LOCATIONS[3:]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def detect_hardware(llamacpp_hint: Optional[str] = None, vllm_hint: Optional[str] = None) -> Hardware:
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
        vllm_native=(bool(vllm_hint and os.path.isfile(vllm_hint) and os.access(vllm_hint, os.X_OK))
                     if vllm_hint else _vllm_native_available()),
        vllm_docker=_vllm_docker_available(),
        llamacpp_path=find_llamacpp(llamacpp_hint),
        sglang=_sglang_available(),
        tgi=_tgi_available(),
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
        numa=detect_numa(),
        llama_devices=llama_capabilities(engines.llamacpp_path)["devices"] if engines.llamacpp_path else [],
    )


def _node_set(value: str) -> set[int]:
    nodes = set()
    for part in value.strip().split(','):
        bounds = [int(n) for n in part.split('-')]
        if len(bounds) > 2 or min(bounds) < 0 or max(bounds) > 65535:
            raise ValueError('Invalid node list')
        nodes.update(range(bounds[0], bounds[-1] + 1))
    return nodes


def detect_numa(online=None, status=None) -> Dict[str, Any]:
    from pathlib import Path
    linux = platform.system() == 'Linux'
    nodes = None
    if linux:
        try:
            visible = _node_set(Path(online or '/sys/devices/system/node/online').read_text())
            lines = Path(status or '/proc/self/status').read_text().splitlines()
            allowed = next(line.split(':', 1)[1] for line in lines if line.startswith('Mems_allowed_list:'))
            nodes = sorted(visible & _node_set(allowed))
        except (OSError, ValueError, StopIteration):
            pass
    return {'nodes': nodes, 'node_count': len(nodes) if nodes is not None else None,
            'numactl_path': shutil.which('numactl') if linux else None, 'linux': linux}


@lru_cache(maxsize=16)
def _llama_capabilities_cached(binary: str, mtime: int) -> Dict[str, Any]:
    import re
    result = {'load_mode': False, 'devices': [], 'mtp': None, 'build': None, 'server_options': {}}
    env = dict(os.environ)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    directory = os.path.dirname(binary)
    if directory:
        env['LD_LIBRARY_PATH'] = directory + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    for option in ('--help', '--list-devices', '--version'):
        try:
            out = subprocess.run([binary, option], capture_output=True, text=True, timeout=8, env=env)
            if out.returncode != 0:
                continue
            output = out.stdout + '\n' + out.stderr
            if option == '--help':
                result['load_mode'] = bool(re.search(r'(?<![\w-])--load-mode(?=\s|=|$)', output))
                for key in ('system_prompt', 'system_prompt_file', 'color'):
                    flag = '--' + key.replace('_', '-')
                    result['server_options'][key] = bool(re.search(r'(?<![\w-])' + re.escape(flag) + r'(?=\s|=|$)', output))
                # MTP is one value of --spec-type's list (build 9180+).
                result['mtp'] = bool(re.search(r'(?<![\w-])--spec-type\s+\S*(?<![\w-])draft-mtp(?![\w-])', output))
            elif option == '--version':
                # "version: 0.5.0-dev (build 11235, commit …)" or older "version: 9180 (2555826)".
                match = re.search(r'\(build (\d+)', output) or re.search(r'version: (\d+) \(', output)
                result['build'] = int(match[1]) if match else None
            else:
                result['devices'] = [{'name': m.group(1), 'description': m.group(2)}
                                     for line in output.splitlines()
                                     if (m := re.match(r'^\s*([A-Za-z][A-Za-z0-9_]*\d+):\s+(.+)$', line))]
        except (OSError, subprocess.SubprocessError):
            pass
    # The build number counts git commits: 0 without git, tiny in a shallow clone.
    if result['build'] is not None and result['mtp'] and result['build'] < 9180:
        result['build'] = None
    return result


def llama_capabilities(binary: str) -> Dict[str, Any]:
    resolved = shutil.which(binary) or binary
    try:
        stamp = os.stat(resolved).st_mtime_ns
    except OSError:
        stamp = 0
    return _llama_capabilities_cached(resolved, stamp)

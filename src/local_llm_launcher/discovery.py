"""Discover locally installed models: HuggingFace cache repos and loose GGUF folders.

Model discovery approach adapted from vllm-cli (https://github.com/Chen-zexi/vllm-cli)
by Chen-zexi, MIT license.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

_WEIGHT_EXTS = (".safetensors", ".bin", ".gguf", ".pt")

# Matches "8B", "27b", "3.5B" preceded by - _ . or start; avoids matching e.g. "A3B" active-params suffix.
_PARAM_RE = re.compile(r"(?:^|[-_./ ])(\d{1,3}(?:\.\d)?)\s?[bB](?:$|[-_. ])")
_GGUF_QUANT_RE = re.compile(r"(?:^|[-_.])((?:I?Q\d[A-Z0-9_]*)|(?:[Ff](?:16|32))|(?:[Bb][Ff]16))(?:$|[-_.])")


@dataclass
class LocalModel:
    repo_id: str               # "org/name" for HF cache, file stem for loose GGUF
    path: str                  # snapshot dir or gguf file path
    format: str                # "safetensors" | "gguf"
    size_bytes: int
    source: str                # "hf-cache" | "folder"
    quant: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    gguf_files: List[Dict[str, Any]] = field(default_factory=list)
    param_count_b: Optional[float] = None
    multimodal: bool = False  # has a vision/audio encoder (can be skipped for text-only)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(vars(self))
        d["size_gb"] = round(self.size_bytes / (1024**3), 2)
        return d


def guess_gguf_quant(filename: str) -> Optional[str]:
    m = _GGUF_QUANT_RE.search(Path(filename).stem)
    return m.group(1).upper() if m else None


def is_multimodal_config(config: Dict[str, Any]) -> bool:
    """Detect a vision/audio encoder from a HuggingFace config.json.

    Multimodal models carry encoder weights (vision tower, etc.) that consume
    GPU memory even when you only want text. Markers: a *ForConditionalGeneration
    architecture (the HF convention for multimodal LMs), or a nested
    vision_config / audio_config / image-token field.
    """
    archs = config.get("architectures") or []
    if any(
        "ConditionalGeneration" in a or "ForVision" in a or "VisionText" in a
        for a in archs
    ):
        return True
    return any(
        key in config
        for key in ("vision_config", "audio_config", "image_token_id", "vision_tower")
    )


def guess_param_count_b(name: str) -> Optional[float]:
    candidates = [float(m) for m in _PARAM_RE.findall(name)]
    if not candidates:
        return None
    # MoE names like "35B-A3B" embed both total and active counts; total is what
    # determines weight memory, so take the largest match.
    return max(candidates)


def _latest_snapshot(repo_dir: Path) -> Optional[Path]:
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [d for d in snaps.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _read_config(snap: Path) -> Dict[str, Any]:
    cfg = snap / "config.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text())
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _file_size(p: Path) -> int:
    try:
        # st_size of a symlink target (HF cache snapshots symlink into blobs/)
        return p.resolve().stat().st_size
    except OSError:
        return 0


_NOT_FOR_INFERENCE = ("training_args.bin", "optimizer.bin", "optimizer.pt", "scheduler.pt", "scaler.pt")


def _loaded_weights(snap: Path, files: List[Path]) -> List[Path]:
    """The files vLLM's automatic loader reads (v0.30 DefaultModelLoader), so a repo
    that ships the same weights twice (.bin and .safetensors, or Mistral's
    consolidated copy beside the sharded one) is counted once."""
    # vLLM globs only the top folder; keep nested files only when nothing is on top.
    files = [f for f in files if f.parent == snap] or files
    safetensors = [f for f in files if f.suffix.lower() == ".safetensors"]
    consolidated = [f for f in safetensors if f.name.startswith("consolidated")]
    chosen = consolidated or safetensors
    if chosen:
        index = "consolidated.safetensors.index.json" if consolidated else "model.safetensors.index.json"
        try:
            # The map lists every tensor; only its few distinct file names matter.
            listed = set(json.loads((snap / index).read_text())["weight_map"].values())
            return [f for f in chosen if f.relative_to(snap).as_posix() in listed] or chosen
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return chosen
    for suffix in (".bin", ".pt"):
        chosen = [f for f in files if f.suffix.lower() == suffix and f.name not in _NOT_FOR_INFERENCE]
        if chosen:
            return chosen
    return []


def _scan_repo(repo_dir: Path) -> Optional[LocalModel]:
    name = repo_dir.name
    if not name.startswith("models--"):
        return None
    repo_id = name[len("models--"):].replace("--", "/")
    snap = _latest_snapshot(repo_dir)
    if snap is None:
        return None

    gguf_files: List[Dict[str, Any]] = []
    other_weights: List[Path] = []
    weight_bytes = 0
    for f in sorted(snap.rglob("*")):
        if not f.is_file() and not f.is_symlink():
            continue
        suffix = f.suffix.lower()
        if suffix not in _WEIGHT_EXTS:
            continue
        if suffix == ".gguf":
            size = _file_size(f)
            weight_bytes += size
            gguf_files.append({
                "filename": f.name,
                "path": str(f),
                "size_bytes": size,
                "size_gb": round(size / (1024**3), 2),
                "quant": guess_gguf_quant(f.name),
            })
        else:
            other_weights.append(f)
    has_safetensors = any(f.suffix.lower() == ".safetensors" for f in other_weights)
    weight_bytes += sum(_file_size(f) for f in _loaded_weights(snap, other_weights))

    if weight_bytes == 0:
        return None

    config = _read_config(snap)
    fmt = "safetensors" if has_safetensors else ("gguf" if gguf_files else "safetensors")
    quant = None
    qc = config.get("quantization_config")
    if isinstance(qc, dict):
        quant = qc.get("quant_method")
    return LocalModel(
        repo_id=repo_id,
        path=str(snap),
        format=fmt,
        size_bytes=weight_bytes,
        source="hf-cache",
        quant=quant,
        config=config,
        gguf_files=gguf_files,
        param_count_b=guess_param_count_b(repo_id),
        multimodal=is_multimodal_config(config),
    )


def scan_hf_cache(hub_dir: Union[str, Path, None] = None) -> List[LocalModel]:
    hub = Path(hub_dir) if hub_dir else DEFAULT_HF_HUB
    if not hub.is_dir():
        return []
    models: List[LocalModel] = []
    for entry in sorted(hub.iterdir()):
        if not entry.is_dir():
            continue
        try:
            model = _scan_repo(entry)
        except OSError:
            continue
        if model:
            models.append(model)
    return models


def scan_gguf_folder(folder: Union[str, Path]) -> List[LocalModel]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    models: List[LocalModel] = []
    for f in sorted(root.rglob("*.gguf")):
        if not f.is_file():
            continue
        size = _file_size(f)
        models.append(LocalModel(
            repo_id=f.stem,
            path=str(f),
            format="gguf",
            size_bytes=size,
            source="folder",
            quant=guess_gguf_quant(f.name),
            gguf_files=[{
                "filename": f.name,
                "path": str(f),
                "size_bytes": size,
                "size_gb": round(size / (1024**3), 2),
                "quant": guess_gguf_quant(f.name),
            }],
            param_count_b=guess_param_count_b(f.stem),
        ))
    return models


def list_installed(extra_gguf_folders: Optional[List[str]] = None) -> List[LocalModel]:
    models = scan_hf_cache()
    for folder in extra_gguf_folders or []:
        models.extend(scan_gguf_folder(folder))
    return models


# Name segments of image/audio encoder tensors in multimodal checkpoints
# (Qwen-VL "visual", LLaVA/Mistral "vision_tower" + "multi_modal_projector",
# Gemma 3n "audio_tower"/"embed_vision"). Text-only mode skips loading them.
_ENCODER_SEGMENTS = frozenset({
    "visual", "vision_tower", "vision_model", "vision_encoder", "multi_modal_projector",
    "mm_projector", "audio_tower", "audio_model", "audio_encoder", "embed_vision", "embed_audio",
})
# A safetensors header is JSON metadata; anything this large is not a real header.
_MAX_SAFETENSORS_HEADER = 100 * 1024**2


def _safetensors_tensor_bytes(path: Path) -> Dict[str, int]:
    """{tensor name: byte length} from a safetensors header; tensor data is never read."""
    with open(path, "rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 0 < length <= _MAX_SAFETENSORS_HEADER:
            raise ValueError("not a safetensors header")
        header = json.loads(stream.read(length))
    return {name: int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])
            for name, entry in header.items() if name != "__metadata__"}


def encoder_weight_bytes(snapshot: Union[str, Path]) -> int:
    """Bytes of image/audio encoder weights that text-only mode skips, or 0 if unknown."""
    snap = Path(snapshot)
    try:
        files = _loaded_weights(snap, [f for f in snap.rglob("*.safetensors") if f.is_file()])
        total = 0
        for f in files:
            for name, size in _safetensors_tensor_bytes(f).items():
                if _ENCODER_SEGMENTS.intersection(name.split(".")):
                    total += size
        return total
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return 0

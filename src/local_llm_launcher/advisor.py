"""The advisor: rates overall model fit and every flag red/yellow/green for the
user's actual hardware, with plain-English explanations.

Heuristics informed by real-world failure modes on consumer GPUs (CUDA/NCCL
overhead, display "tax" on the GPU driving monitors, KV-cache starvation) and by
vllm-cli's configuration model (https://github.com/Chen-zexi/vllm-cli, MIT).

All inputs are plain dicts in the JSON shapes produced by Hardware.to_dict()
and LocalModel.to_dict().
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import catalog

GB = 1024**3
MB = 1024**2

# Memory vLLM/torch needs per GPU outside its managed pool (CUDA context, NCCL).
CUDA_OVERHEAD_GB = 1.7
# Temporary allocation headroom needed per GPU while weights stream in.
LOAD_BUFFER_GB = 1.3
# Working buffer inside the pool for activations/compile scratch.
WORKING_BUFFER_GB = 1.0
# Fraction of Apple unified memory it is sensible to give the model.
APPLE_USABLE_FRACTION = 0.75
# Fraction of system RAM usable for CPU-only inference.
CPU_RAM_FRACTION = 0.8

GREEN, YELLOW, RED = "green", "yellow", "red"
_RANK = {GREEN: 0, YELLOW: 1, RED: 2}

# Model-family patterns that need a reasoning parser to separate "thinking" text.
_REASONING_FAMILIES = [
    (re.compile(r"qwen3", re.I), "qwen3"),
    (re.compile(r"deepseek[-_]?r1", re.I), "deepseek_r1"),
    (re.compile(r"gemma[-_]?4", re.I), "gemma4"),
]


def _kv_mb_per_token_heuristic(param_b: Optional[float]) -> float:
    """Rough KV-cache cost when the model config is unavailable (e.g. GGUF)."""
    if param_b is None:
        return 0.125
    if param_b < 4:
        return 0.05
    if param_b < 10:
        return 0.125
    if param_b < 34:
        return 0.16
    return 0.2


def estimate_kv_gb(
    config: Dict[str, Any],
    param_b: Optional[float],
    max_len: int,
    seqs: int = 1,
    dtype_bytes: float = 2.0,
) -> float:
    """Estimate KV-cache size. Uses the model's config.json when available."""
    layers = config.get("num_hidden_layers")
    kv_heads = config.get("num_key_value_heads") or config.get("num_attention_heads")
    head_dim = config.get("head_dim")
    if head_dim is None and config.get("hidden_size") and config.get("num_attention_heads"):
        head_dim = config["hidden_size"] / config["num_attention_heads"]
    if layers and kv_heads and head_dim:
        per_token = 2 * layers * kv_heads * head_dim * dtype_bytes  # K and V
        return per_token * max_len * max(seqs, 1) / GB
    return _kv_mb_per_token_heuristic(param_b) * (dtype_bytes / 2.0) * max_len * max(seqs, 1) * MB / GB


def _selected_gpus(hw: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    gpus = hw.get("gpus") or []
    raw = config.get("device_ids")
    if raw:
        try:
            wanted = {int(x) for x in str(raw).replace(" ", "").split(",") if x != ""}
            picked = [g for g in gpus if g["index"] in wanted]
            if picked:
                return picked
        except ValueError:
            pass
    return gpus


class _Report:
    def __init__(self) -> None:
        self.flags: Dict[str, Dict[str, Optional[str]]] = {}
        self.blockers: List[str] = []
        self.details: List[str] = []
        self.min_level: Optional[str] = None  # floor for the overall verdict

    def escalate(self, level: str, detail: str) -> None:
        self.details.append(detail)
        if self.min_level is None or _RANK[level] > _RANK[self.min_level]:
            self.min_level = level

    def flag(self, key: str, level: str, message: Optional[str] = None) -> None:
        existing = self.flags.get(key)
        if existing and _RANK[existing["level"]] >= _RANK[level]:
            return
        self.flags[key] = {"level": level, "message": message}


def _merged_config(engine: str, config: Dict[str, Any]) -> Dict[str, Any]:
    merged = catalog.defaults(engine)
    merged.update({k: v for k, v in config.items() if v is not None})
    return merged


# --------------------------------------------------------------------------- vLLM

def _advise_vllm(model: Dict[str, Any], cfg: Dict[str, Any], hw: Dict[str, Any], rep: _Report) -> Dict[str, Any]:
    gpus = _selected_gpus(hw, cfg)
    weights_gb = model["size_bytes"] / GB

    # Hard blockers first.
    if model.get("format") == "gguf":
        rep.blockers.append(
            "This is a GGUF file — vLLM does not load GGUF reliably. Switch the engine to llama.cpp, "
            "which is built for GGUF."
        )
    if hw.get("apple_silicon"):
        rep.blockers.append(
            "vLLM cannot use the GPU in Apple Silicon Macs. Use llama.cpp instead — it supports "
            "Apple's Metal acceleration and will be much faster here."
        )
    elif not gpus:
        rep.blockers.append(
            "vLLM needs an NVIDIA GPU and none was detected. Use llama.cpp, which can run on CPU."
        )

    util = float(cfg.get("gpu_memory_utilization", 0.85))
    tp = int(cfg.get("tensor_parallel_size", 1))
    max_len = int(cfg.get("max_model_len", 8192))
    seqs = int(cfg.get("max_num_seqs", 4))

    # tensor_parallel_size rules
    if gpus:
        if tp > len(gpus):
            rep.flag("tensor_parallel_size", RED,
                     f"You asked to split across {tp} GPUs but only {len(gpus)} are available. "
                     f"Set this to {len(gpus)} or fewer.")
        elif len(gpus) % tp != 0:
            rep.flag("tensor_parallel_size", YELLOW,
                     f"{tp} doesn't divide evenly into your {len(gpus)} GPUs, so some GPUs would sit idle.")

    # Capacity: vLLM claims `util` of each GPU it uses; weights + KV + scratch live in that pool.
    used_gpus = gpus[: max(tp, 1)] if gpus else []
    capacity_gb = sum(g["vram_total_mb"] for g in used_gpus) / 1024 * util
    kv_dtype_bytes = 1.0 if cfg.get("kv_cache_dtype") == "fp8" else 2.0
    # vLLM's KV pool is shared across requests, not pre-reserved per request —
    # cap the multiplier at 2 so "needed" reflects a sane working minimum.
    kv_gb = estimate_kv_gb(model.get("config") or {}, model.get("param_count_b"),
                           max_len, seqs=min(seqs, 2), dtype_bytes=kv_dtype_bytes)
    needed_gb = weights_gb + kv_gb + WORKING_BUFFER_GB

    if gpus and tp < len(gpus) and needed_gb > capacity_gb:
        all_capacity = sum(g["vram_total_mb"] for g in gpus) / 1024 * util
        if needed_gb <= all_capacity:
            rep.flag("tensor_parallel_size", YELLOW,
                     f"This model needs ~{needed_gb:.0f} GB but {tp} GPU(s) only give "
                     f"{capacity_gb:.0f} GB. Set this to {len(gpus)} to use all your GPUs.")

    # gpu_memory_utilization rules — the pre-load headroom check uses FREE memory,
    # which the desktop may be eating into on the GPU driving the monitors.
    if used_gpus:
        worst_free_ratio = min(g["vram_free_mb"] / g["vram_total_mb"] for g in used_gpus)
        if util > worst_free_ratio - 0.005:
            busiest = min(used_gpus, key=lambda g: g["vram_free_mb"] / g["vram_total_mb"])
            in_use = (busiest["vram_total_mb"] - busiest["vram_free_mb"]) / 1024
            rep.flag("gpu_memory_utilization", RED,
                     f"Your desktop and other programs are already using ~{in_use:.1f} GB on GPU "
                     f"{busiest['index']}, so vLLM can't claim {util:.2f} of it — the launch will fail "
                     f"its memory check. Lower this to {max(worst_free_ratio - 0.02, 0.5):.2f} or close "
                     f"GPU-heavy apps (browsers, games).")
        elif util > 0.90 and all(g["vram_total_mb"] <= 24 * 1024 for g in used_gpus):
            rep.flag("gpu_memory_utilization", YELLOW,
                     "Above 0.90 on consumer cards the launch often fails its memory check because "
                     "the driver and desktop need some space. 0.85–0.90 is the safe zone.")

    # quantization override
    if cfg.get("quantization"):
        rep.flag("quantization", YELLOW,
                 "vLLM reads the compression format from the model files automatically, and refuses to "
                 "start if this setting doesn't match exactly. Leave it empty unless a model's "
                 "instructions say otherwise.")

    # cpu offload
    if float(cfg.get("cpu_offload_gb") or 0) > 0:
        rep.flag("cpu_offload_gb", YELLOW,
                 "This makes an oversized model run, but roughly 10x slower — the offloaded part is "
                 "re-sent over the PCIe slot on every word generated. Prefer a smaller or more "
                 "compressed model if one exists.")

    # enforce eager
    if cfg.get("enforce_eager"):
        rep.flag("enforce_eager", YELLOW,
                 "This disables CUDA graphs, which usually cost ~5-10x generation speed. Only keep it "
                 "on if the model fails at the final startup step otherwise.")

    # context above the model's trained limit
    max_pos = (model.get("config") or {}).get("max_position_embeddings")
    if max_pos and max_len > int(max_pos):
        rep.flag("max_model_len", YELLOW,
                 f"This model was trained for a maximum of {int(max_pos):,} tokens of context. "
                 f"Asking for more usually fails at startup or degrades answers.")

    # reasoning parser: suggest when empty, and hard-stop a family mismatch —
    # a wrong parser aborts the launch instantly ("could not locate think
    # start/end tokens in the tokenizer").
    detected = None
    for pattern, parser in _REASONING_FAMILIES:
        if pattern.search(model.get("repo_id", "")):
            detected = parser
            break
    chosen = cfg.get("reasoning_parser")
    if chosen and detected and chosen != detected:
        rep.flag("reasoning_parser", RED,
                 f"'{chosen}' is the wrong format for this model — it looks like a '{detected}' "
                 f"family model, and a mismatched parser makes the launch fail instantly. "
                 f"Pick '{detected}' (or leave this empty).")
    elif chosen and not detected:
        rep.flag("reasoning_parser", YELLOW,
                 f"Couldn't confirm this model uses the '{chosen}' thinking format. If the launch "
                 f"fails immediately, clear this setting.")
    elif detected and not chosen:
        rep.flag("reasoning_parser", YELLOW,
                 f"This looks like a '{detected}' family model that thinks step-by-step before "
                 f"answering. Set the thinking-model format to '{detected}' so apps get clean "
                 f"answers instead of raw thinking text.")

    # many simultaneous requests on consumer hardware
    if seqs > 32 and gpus and all(g["vram_total_mb"] <= 24 * 1024 for g in gpus):
        rep.flag("max_num_seqs", YELLOW,
                 "Each simultaneous request reserves conversation memory on the GPU. On consumer "
                 "cards, high values can starve the model of memory. 4-16 is plenty for personal use.")

    if cfg.get("kv_cache_dtype") == "fp8":
        rep.flag("kv_cache_dtype", GREEN,
                 "Good choice for tight fits — this roughly halves conversation memory with little "
                 "quality loss.")

    # Load-time headroom per GPU: total capacity can look fine while one card —
    # usually the one driving the monitors — lacks FREE memory for its share of
    # the weights right now. (Real failure: 31B AWQ at TP=2 OOMed on GPU 0 by
    # ~400 MB because the desktop held ~2 GB there.)
    if used_gpus and not rep.blockers:
        per_gpu_load_gb = weights_gb / max(tp, 1) + CUDA_OVERHEAD_GB + LOAD_BUFFER_GB
        worst = min(used_gpus, key=lambda g: g["vram_free_mb"])
        worst_free_gb = worst["vram_free_mb"] / 1024
        if per_gpu_load_gb > worst_free_gb:
            shortfall = per_gpu_load_gb - worst_free_gb
            in_use = (worst["vram_total_mb"] - worst["vram_free_mb"]) / 1024
            rep.escalate(
                RED if shortfall > 1.5 else YELLOW,
                f"Loading needs ~{per_gpu_load_gb:.1f} GB free on each card, but GPU "
                f"{worst['index']} only has {worst_free_gb:.1f} GB free right now — other "
                f"programs (usually your desktop and browser on the GPU driving your "
                f"monitors) are using {in_use:.1f} GB of it. Close GPU-heavy apps and "
                f"this should fit; you're about {shortfall:.1f} GB short.",
            )

    available_gb = capacity_gb
    return {
        "weights_gb": round(weights_gb, 1),
        "kv_cache_gb": round(kv_gb, 1),
        "working_buffer_gb": WORKING_BUFFER_GB,
        "overhead_gb": round(CUDA_OVERHEAD_GB * len(used_gpus), 1),
        "needed_gb": round(needed_gb, 1),
        "available_gb": round(available_gb, 1),
        "basis": f"{len(used_gpus)} GPU(s) x utilization {util:.2f}",
    }


# ----------------------------------------------------------------------- llama.cpp

def _advise_llamacpp(model: Dict[str, Any], cfg: Dict[str, Any], hw: Dict[str, Any], rep: _Report) -> Dict[str, Any]:
    if model.get("format") == "safetensors":
        rep.blockers.append(
            "This model is in safetensors format — llama.cpp only loads GGUF files. Either switch "
            "the engine to vLLM, or download a GGUF version of this model (search for "
            f"'{model.get('repo_id', '')} GGUF')."
        )

    gpus = hw.get("gpus") or []
    apple = hw.get("apple_silicon")
    ngl = int(cfg.get("n_gpu_layers", 999))
    ctx = int(cfg.get("ctx_size", 8192))
    weights_gb = model["size_bytes"] / GB

    kv_gb_f16 = estimate_kv_gb(model.get("config") or {}, model.get("param_count_b"), ctx, seqs=1)
    mult = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}
    k_mult = mult.get(str(cfg.get("cache_type_k", "f16")), 1.0)
    v_mult = mult.get(str(cfg.get("cache_type_v", "f16")), 1.0)
    kv_gb = kv_gb_f16 * (k_mult + v_mult) / 2
    needed_gb = weights_gb + kv_gb + 0.7  # compute buffers

    if apple:
        available_gb = float(apple["memory_gb"]) * APPLE_USABLE_FRACTION
        basis = f"unified memory x {APPLE_USABLE_FRACTION:.0%}"
    elif gpus and ngl > 0:
        available_gb = sum(g["vram_total_mb"] for g in gpus) / 1024 - 1.0 * len(gpus)
        basis = f"{len(gpus)} GPU(s) minus driver overhead"
    else:
        available_gb = float(hw.get("ram_gb", 0)) * CPU_RAM_FRACTION
        basis = f"system RAM x {CPU_RAM_FRACTION:.0%}"

    # flag rules
    if gpus and ngl == 0:
        rep.flag("n_gpu_layers", YELLOW,
                 "You have a GPU but this setting puts every layer on the CPU, which is far slower. "
                 "Set it to 999 to use the GPU fully.")
    if gpus and not apple and ngl > 0 and needed_gb > available_gb:
        ram_gb = float(hw.get("ram_gb", 0))
        if weights_gb < ram_gb * CPU_RAM_FRACTION:
            rep.flag("n_gpu_layers", YELLOW,
                     f"The whole model (~{needed_gb:.0f} GB) doesn't fit in GPU memory "
                     f"(~{available_gb:.0f} GB usable). Lower this number to put some layers in RAM — "
                     f"it still runs, just slower. Try reducing until it loads.")

    if str(cfg.get("cache_type_v", "f16")) in ("q8_0", "q4_0") and str(cfg.get("flash_attn", "auto")) == "off":
        rep.flag("cache_type_v", RED,
                 "Compressing the V cache requires flash attention to be on. Either set flash "
                 "attention to 'auto'/'on', or set this back to f16.")

    threads = cfg.get("threads")
    if threads and int(threads) > int(hw.get("cpu_cores", 1)):
        rep.flag("threads", YELLOW,
                 f"You asked for {threads} threads but this machine has {hw.get('cpu_cores')} CPU "
                 f"cores. More threads than cores makes things slower, not faster.")

    if int(cfg.get("parallel", 1)) > 1:
        rep.flag("parallel", GREEN,
                 f"Heads up: the context window is split between slots, so each of the "
                 f"{cfg.get('parallel')} requests gets {ctx // int(cfg.get('parallel', 1)):,} tokens.")

    if cfg.get("mlock") and weights_gb > float(hw.get("ram_gb", 0)) * 0.8:
        rep.flag("mlock", YELLOW,
                 "Locking the model in RAM needs the whole file to fit in memory with room to spare — "
                 "this model is too big for that on this machine.")

    if ctx > 131072:
        rep.flag("ctx_size", YELLOW,
                 "Very few models support context this long, and conversation memory grows with it. "
                 "Check the model's card before going past 131,072.")

    return {
        "weights_gb": round(weights_gb, 1),
        "kv_cache_gb": round(kv_gb, 1),
        "working_buffer_gb": 0.7,
        "overhead_gb": 0.0,
        "needed_gb": round(needed_gb, 1),
        "available_gb": round(available_gb, 1),
        "basis": basis,
    }


# ------------------------------------------------------------------------- public

def advise(engine: str, model: Dict[str, Any], config: Dict[str, Any], hw: Dict[str, Any]) -> Dict[str, Any]:
    """Rate a launch configuration. Returns overall fit, memory budget, per-flag ratings."""
    cfg = _merged_config(engine, config)
    rep = _Report()

    if engine == "vllm":
        budget = _advise_vllm(model, cfg, hw, rep)
    elif engine == "llamacpp":
        budget = _advise_llamacpp(model, cfg, hw, rep)
    else:
        raise ValueError(f"Unknown engine '{engine}'")

    if rep.blockers:
        overall = {"level": RED, "headline": rep.blockers[0], "details": rep.blockers[1:]}
        pct = None
    else:
        pct = budget["needed_gb"] / budget["available_gb"] if budget["available_gb"] > 0 else 99.0
        if pct <= 0.75:
            level, head = GREEN, (
                f"Fits comfortably — needs ~{budget['needed_gb']} GB of the ~{budget['available_gb']} GB available."
            )
        elif pct <= 0.95:
            level, head = YELLOW, (
                f"Tight fit — needs ~{budget['needed_gb']} GB of the ~{budget['available_gb']} GB available. "
                f"It should load, but consider a shorter context window or compressed conversation memory."
            )
        else:
            level, head = RED, (
                f"Won't fit — needs ~{budget['needed_gb']} GB but only ~{budget['available_gb']} GB is "
                f"available. Try a smaller or more compressed (quantized) version of this model."
            )
        red_flags = [k for k, v in rep.flags.items() if v["level"] == RED]
        if red_flags and level != RED:
            level = YELLOW
            head += " One or more settings below need attention first."
        if rep.min_level and _RANK[rep.min_level] > _RANK[level]:
            level = rep.min_level
        overall = {"level": level, "headline": head, "details": rep.details}

    budget["pct"] = round(pct, 2) if pct is not None else None
    return {"overall": overall, "budget": budget, "flags": rep.flags, "engine": engine,
            "effective_config": cfg}


def _max_context_that_fits(engine: str, model: Dict[str, Any], hw: Dict[str, Any],
                           base: Dict[str, Any], kv_dtype_bytes: float) -> int:
    """Largest context length keeping the fit under ~90% of budget, rounded to 1024."""
    best = 2048
    for length in range(2048, 262145, 2048):
        cfg = dict(base)
        key = "max_model_len" if engine == "vllm" else "ctx_size"
        cfg[key] = length
        a = advise(engine, model, cfg, hw)
        if a["budget"]["pct"] is not None and a["budget"]["pct"] <= 0.9:
            best = length
        else:
            break
    max_pos = (model.get("config") or {}).get("max_position_embeddings")
    if max_pos:
        best = min(best, int(max_pos))
    return best


def presets(engine: str, model: Dict[str, Any], hw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Hardware-computed launch presets, not static templates."""
    gpus = hw.get("gpus") or []
    out: List[Dict[str, Any]] = []

    if engine == "vllm":
        tp = max(len(gpus), 1)
        safe = {"tensor_parallel_size": tp, "gpu_memory_utilization": 0.85,
                "max_model_len": 8192, "max_num_seqs": 4}
        out.append({"name": "Safe (recommended)", "config": safe,
                    "description": "Conservative settings that almost always start on the first try."})

        ctx_base = {"tensor_parallel_size": tp, "gpu_memory_utilization": 0.88,
                    "max_num_seqs": 1, "kv_cache_dtype": "fp8"}
        max_ctx = _max_context_that_fits("vllm", model, hw, ctx_base, 1.0)
        out.append({"name": "Max context", "config": {**ctx_base, "max_model_len": max_ctx},
                    "description": f"Longest document window that fits: ~{max_ctx:,} tokens, "
                                   "with compressed conversation memory."})

        out.append({"name": "Max speed", "config": {
            "tensor_parallel_size": tp, "gpu_memory_utilization": 0.85,
            "max_model_len": 4096, "max_num_seqs": 8,
            "enable_prefix_caching": True, "enable_chunked_prefill": True},
            "description": "Short context, more simultaneous requests, all speed features on."})

        if gpus:
            worst_free = min(g["vram_free_mb"] / g["vram_total_mb"] for g in gpus)
            tight_util = round(max(min(worst_free - 0.02, 0.95), 0.5), 2)
            out.append({"name": "Tight fit", "config": {
                "tensor_parallel_size": tp, "gpu_memory_utilization": tight_util,
                "max_model_len": 4096, "max_num_seqs": 1, "kv_cache_dtype": "fp8"},
                "description": "For models that barely fit: claims as much GPU memory as is "
                               "actually free right now and minimizes everything else."})
    else:
        out.append({"name": "Safe (recommended)", "config": {
            "ctx_size": 8192, "flash_attn": "auto", "jinja": True},
            "description": "Sensible context; llama.cpp automatically fits as much of the "
                           "model onto the GPU as possible."})
        ctx_base = {"flash_attn": "on",
                    "cache_type_k": "q8_0", "cache_type_v": "q8_0", "jinja": True}
        max_ctx = _max_context_that_fits("llamacpp", model, hw, ctx_base, 1.0)
        out.append({"name": "Max context", "config": {**ctx_base, "ctx_size": max_ctx},
                    "description": f"Longest document window that fits: ~{max_ctx:,} tokens, "
                                   "with compressed conversation memory."})
        out.append({"name": "CPU fallback", "config": {
            "n_gpu_layers": 0, "ctx_size": 4096, "jinja": True},
            "description": "Runs entirely on the CPU — slow but works anywhere."})

    return out

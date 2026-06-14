"""REST API for the GUI."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import __version__, advisor, catalog, discovery, failures, hardware
from .config import Settings
from .downloads import DownloadManager, repo_files, search_hub
from .openwebui import OpenWebUIManager
from .registry import ServerManager

router = APIRouter(prefix="/api")

settings = Settings()
servers = ServerManager()
downloads = DownloadManager()
openwebui = OpenWebUIManager()

_hw_cache: Dict[str, Any] = {"at": 0.0, "data": None}
_hw_lock = threading.Lock()


def get_hardware(max_age: float = 5.0) -> Dict[str, Any]:
    now = time.time()
    with _hw_lock:
        if _hw_cache["data"] is None or now - _hw_cache["at"] > max_age:
            hw = hardware.detect_hardware(llamacpp_hint=settings.data.get("llamacpp_path"))
            _hw_cache["data"] = hw.to_dict()
            _hw_cache["at"] = now
        return _hw_cache["data"]


def installed_models() -> List[Dict[str, Any]]:
    models = discovery.list_installed(settings.data.get("gguf_folders") or [])
    return [m.to_dict() for m in models]


def find_model(repo_id: str) -> Dict[str, Any]:
    for m in installed_models():
        if m["repo_id"] == repo_id:
            return m
    raise HTTPException(404, f"Model '{repo_id}' is not installed.")


def recommended_engine(model: Dict[str, Any], hw: Dict[str, Any]) -> str:
    if model["format"] == "gguf":
        return "llamacpp"
    if hw.get("apple_silicon") or not hw.get("gpus"):
        return "llamacpp"
    if hw["engines"].get("sglang"):
        return "sglang"
    if hw["engines"].get("vllm_native"):
        return "vllm-native"
    if hw["engines"].get("vllm_docker"):
        return "vllm-docker"
    return "vllm-native"


# ----------------------------------------------------------------------- system

@router.get("/hardware")
def api_hardware():
    return get_hardware()


@router.get("/about")
def api_about():
    return {
        "name": "Local-LLM-Launcher-GUI",
        "version": __version__,
        "credits": [
            {
                "name": "vllm-cli by Chen-zexi",
                "url": "https://github.com/Chen-zexi/vllm-cli",
                "note": "This app adapts vllm-cli's flag catalog, configuration profiles, "
                        "server lifecycle management, and model discovery concepts (MIT license).",
            },
            {"name": "vLLM", "url": "https://github.com/vllm-project/vllm"},
            {"name": "llama.cpp", "url": "https://github.com/ggml-org/llama.cpp"},
            {"name": "SGLang", "url": "https://github.com/sgl-project/sglang"},
        ],
    }


@router.get("/settings")
def api_get_settings():
    return settings.public()


class SettingsUpdate(BaseModel):
    hf_token: Optional[str] = None
    gguf_folders: Optional[List[str]] = None
    llamacpp_path: Optional[str] = None


_UNSET = object()


class SettingsPatch(BaseModel):
    hf_token: Optional[str] = _UNSET
    gguf_folders: Optional[List[str]] = _UNSET
    llamacpp_path: Optional[str] = _UNSET


@router.put("/settings")
def api_put_settings(body: SettingsUpdate):
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if changes.get("hf_token") == "********":
        changes.pop("hf_token")  # masked value bounced back — keep stored token
    settings.update(changes)
    _hw_cache["data"] = None  # llamacpp_path may have changed
    return settings.public()


@router.patch("/settings")
def api_patch_settings(body: Dict[str, Any]):
    changes = {}
    for key in ("hf_token", "gguf_folders", "llamacpp_path"):
        if key in body:
            val = body[key]
            if key == "hf_token" and val == "********":
                continue  # masked value bounced back
            changes[key] = val
    settings.update(changes)
    _hw_cache["data"] = None
    return settings.public()


# ----------------------------------------------------------------------- models

@router.get("/models")
def api_models():
    hw = get_hardware()
    out = []
    rank = {"green": 0, "yellow": 1, "red": 2}
    for m in installed_models():
        engine = recommended_engine(m, hw)
        if engine == "llamacpp":
            adv_engine = "llamacpp"
        elif engine == "sglang":
            adv_engine = "sglang"
        else:
            adv_engine = "vllm"
        # The badge answers "can this run on my hardware at all?" — so report the
        # best outcome across presets, not just the conservative default.
        best = None
        for preset in advisor.presets(adv_engine, m, hw):
            advice = advisor.advise(adv_engine, m, preset["config"], hw)
            if best is None or rank[advice["overall"]["level"]] < rank[best["overall"]["level"]]:
                best = advice
            if best["overall"]["level"] == "green":
                break
        out.append({
            **m,
            "recommended_engine": engine,
            "fit": best["overall"]["level"],
            "fit_headline": best["overall"]["headline"],
        })
    return {"models": out}


@router.get("/models/search")
def api_models_search(q: str):
    try:
        return {"results": search_hub(q, token=settings.data.get("hf_token"))}
    except Exception:
        raise HTTPException(502, "Search failed — check your internet connection and try again.")


@router.get("/models/repo/{repo_id:path}")
def api_repo_detail(repo_id: str):
    try:
        return repo_files(repo_id, token=settings.data.get("hf_token"))
    except Exception:
        raise HTTPException(502, "Could not load repository details — check the name and try again.")


class DownloadRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None  # set for single-GGUF-file downloads


@router.post("/downloads")
def api_start_download(body: DownloadRequest):
    try:
        job = downloads.start(body.repo_id, body.filename, token=settings.data.get("hf_token"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return job.to_dict()


@router.get("/downloads")
def api_downloads():
    return {"downloads": downloads.list()}


# ---------------------------------------------------------------- advise/launch

@router.get("/catalog/{engine}")
def api_catalog(engine: str):
    try:
        return catalog.load_catalog(engine)
    except ValueError as e:
        raise HTTPException(404, str(e))


class AdviseRequest(BaseModel):
    engine: str  # "vllm" | "llamacpp" | "sglang"
    repo_id: str
    config: Dict[str, Any] = {}


@router.post("/advise")
def api_advise(body: AdviseRequest):
    model = find_model(body.repo_id)
    try:
        return advisor.advise(body.engine, model, body.config, get_hardware())
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/presets")
def api_presets(engine: str, repo_id: str):
    model = find_model(repo_id)
    return {"presets": advisor.presets(engine, model, get_hardware())}


class LaunchRequest(BaseModel):
    engine_mode: str  # "vllm-native" | "vllm-docker" | "llamacpp" | "sglang"
    repo_id: str
    config: Dict[str, Any] = {}


@router.post("/servers")
def api_launch(body: LaunchRequest):
    model = find_model(body.repo_id)
    config = dict(body.config)
    if body.engine_mode.startswith("vllm") and settings.data.get("hf_token") and not config.get("hf_token"):
        config["hf_token"] = settings.data["hf_token"]
    if body.engine_mode == "sglang" and settings.data.get("hf_token") and not config.get("hf_token"):
        config["hf_token"] = settings.data["hf_token"]
    hw = get_hardware()
    if body.engine_mode == "llamacpp" and not hw["engines"].get("llamacpp_path"):
        raise HTTPException(400, "llama.cpp (llama-server) was not found on this computer. "
                                 "See Settings for install instructions.")
    if body.engine_mode == "sglang" and not hw["engines"].get("sglang"):
        raise HTTPException(400, "SGLang is not installed on this computer. "
                                 "Install it with: pip install sglang")
    try:
        srv = servers.launch(body.engine_mode, model, config,
                             llamacpp_binary=hw["engines"].get("llamacpp_path"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return srv.status()


@router.get("/servers")
def api_servers():
    return {"servers": [_enrich(s) for s in servers.list()]}


def _enrich(status: Dict[str, Any]) -> Dict[str, Any]:
    srv = servers.get(status["id"])
    if srv is None:
        return status
    if status["running"]:
        status["healthy"] = srv.health()
    else:
        friendly = failures.translate(srv.tail_logs(300))
        if friendly:
            status["failure_explanation"] = friendly
    return status


@router.get("/servers/{server_id}")
def api_server(server_id: str):
    srv = servers.get(server_id)
    if not srv:
        raise HTTPException(404, "No such server.")
    return _enrich(srv.status())


@router.get("/servers/{server_id}/logs")
def api_server_logs(server_id: str, n: int = 200):
    srv = servers.get(server_id)
    if not srv:
        raise HTTPException(404, "No such server.")
    return {"lines": srv.tail_logs(n)}


@router.post("/servers/{server_id}/stop")
def api_server_stop(server_id: str):
    if not servers.stop(server_id):
        raise HTTPException(404, "No such server.")
    return {"ok": True}


@router.delete("/servers/{server_id}")
def api_server_remove(server_id: str):
    if not servers.remove(server_id):
        raise HTTPException(404, "No such server.")
    return {"ok": True}


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7


@router.post("/servers/{server_id}/chat")
def api_chat(server_id: str, body: ChatRequest):
    srv = servers.get(server_id)
    if not srv:
        raise HTTPException(404, "No such server.")
    if not srv.is_running():
        raise HTTPException(400, "That server is not running.")
    try:
        r = httpx.post(
            f"http://127.0.0.1:{srv.port}/v1/chat/completions",
            json={"model": srv.model_label, "messages": body.messages,
                  "max_tokens": body.max_tokens, "temperature": body.temperature,
                  "stream": False},
            timeout=600.0,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return {"content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or ""}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"The model server didn't answer: {e}")


# --------------------------------------------------------------- Open WebUI

@router.get("/openwebui")
def api_openwebui_status():
    return openwebui.status()


@router.post("/openwebui/launch")
def api_openwebui_launch():
    # Pre-connect Open WebUI to every model that's currently running, so they
    # show up ready to chat with the moment it opens.
    connect_urls = [
        f"http://localhost:{s['port']}/v1"
        for s in servers.list()
        if s.get("running")
    ]
    try:
        return openwebui.launch(connect_urls=connect_urls)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.post("/openwebui/stop")
def api_openwebui_stop():
    openwebui.stop()
    return openwebui.status()

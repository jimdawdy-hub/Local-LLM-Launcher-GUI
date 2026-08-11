"""FastAPI application: API routes + the built frontend."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import router


def create_app() -> FastAPI:
    app = FastAPI(title="Local-LLM-Launcher-GUI")
    # DNS-rebinding protection: only loopback hosts may reach the API.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.include_router(router)

    static_dir = Path(str(resources.files("local_llm_launcher") / "static"))
    index = static_dir / "index.html"

    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if index.is_file():
            return FileResponse(index)
        return JSONResponse(
            {"error": "Frontend not built. Run: cd frontend && npm install && npm run build"},
            status_code=503,
        )

    return app

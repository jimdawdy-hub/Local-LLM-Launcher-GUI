"""Entry point: `local-llm-launcher` starts the server and opens the browser."""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="local-llm-launcher",
        description="GUI for downloading and launching local LLMs (vLLM + llama.cpp).",
    )
    parser.add_argument("--port", type=int, default=8765, help="GUI port (default 8765)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    import uvicorn

    from .app import create_app

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Local-LLM-Launcher-GUI running at {url}  (Ctrl+C to quit)")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

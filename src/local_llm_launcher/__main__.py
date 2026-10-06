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
    parser.add_argument(
        "--allow-host", action="append", default=[], metavar="NAME",
        help="also accept requests addressed to this exact host name, e.g. a "
             "`tailscale serve` address (repeatable; the server still listens "
             "on 127.0.0.1 only)",
    )
    args = parser.parse_args()

    import uvicorn

    from . import app as app_module
    from .api import servers
    from .registry import find_free_port

    port = find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is in use, using {port} instead.")

    servers.start_watchdog()
    url = f"http://127.0.0.1:{port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Local-LLM-Launcher-GUI running at {url}  (Ctrl+C to quit)")
    uvicorn.run(app_module.create_app(extra_allowed_hosts=args.allow_host), host="127.0.0.1", port=port, log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()

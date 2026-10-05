"""Extra trusted host names, e.g. a Tailscale `tailscale serve` address."""
import pytest
from fastapi.testclient import TestClient

from local_llm_launcher.app import create_app

TAILNET = "gpuhost.tail048ee2.ts.net"


def _status(app, host):
    with TestClient(app, base_url=f"https://{host}") as client:
        return client.get("/").status_code


def test_unlisted_host_is_still_rejected_by_default():
    assert _status(create_app(), TAILNET) == 400


def test_an_allowed_host_reaches_the_app_and_others_stay_blocked():
    app = create_app(extra_allowed_hosts=[TAILNET])
    assert _status(app, TAILNET) != 400
    assert _status(app, f"{TAILNET}:8765") != 400
    assert _status(app, "127.0.0.1") != 400
    assert _status(app, "evil.example.com") == 400


@pytest.mark.parametrize("bad", ["*", "*.ts.net", "", "  "])
def test_wildcards_and_blanks_are_refused(bad):
    with pytest.raises(ValueError):
        create_app(extra_allowed_hosts=[bad])


def test_cli_passes_allow_host_through(monkeypatch):
    import uvicorn

    from local_llm_launcher import __main__ as cli
    from local_llm_launcher import app as app_module

    seen = {}
    monkeypatch.setattr(app_module, "create_app",
                        lambda extra_allowed_hosts=(): seen.setdefault("hosts", list(extra_allowed_hosts)))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.setdefault("host", kw.get("host")))
    monkeypatch.setattr("sys.argv", ["local-llm-launcher", "--no-browser", "--port", "18765",
                                     "--allow-host", TAILNET, "--allow-host", "other.ts.net"])
    cli.main()
    assert seen["hosts"] == [TAILNET, "other.ts.net"]
    # Still bound to loopback only; the tailnet reaches it through tailscale serve.
    assert seen["host"] == "127.0.0.1"

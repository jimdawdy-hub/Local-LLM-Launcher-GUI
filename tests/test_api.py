"""API smoke tests with FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

from local_llm_launcher.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_hardware(client):
    r = client.get("/api/hardware")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and "engines" in body


def test_about_credits_vllm_cli(client):
    r = client.get("/api/about")
    assert r.status_code == 200
    credits = r.json()["credits"]
    assert any("Chen-zexi" in c["name"] for c in credits)


def test_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    for m in r.json()["models"]:
        assert m["fit"] in ("green", "yellow", "red")
        assert m["recommended_engine"] in ("vllm-native", "vllm-docker", "llamacpp")


def test_catalog_routes(client):
    for engine in ("vllm", "llamacpp"):
        r = client.get(f"/api/catalog/{engine}")
        assert r.status_code == 200
        assert len(r.json()["flags"]) > 5
    assert client.get("/api/catalog/bogus").status_code == 404


def test_advise_unknown_model_404(client):
    r = client.post("/api/advise", json={"engine": "vllm", "repo_id": "no/such-model", "config": {}})
    assert r.status_code == 404


def test_servers_empty_or_list(client):
    r = client.get("/api/servers")
    assert r.status_code == 200
    assert isinstance(r.json()["servers"], list)


def test_settings_roundtrip_masks_token(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    r2 = client.put("/api/settings", json={"gguf_folders": []})
    assert r2.status_code == 200
    assert "hf_token_set" in r2.json()


def test_settings_roundtrip_lan_access_toggle(client):
    r = client.patch("/api/settings", json={"lan_access": True})
    assert r.status_code == 200
    assert r.json()["lan_access"] is True
    r2 = client.put("/api/settings", json={"lan_access": False})
    assert r2.status_code == 200
    assert r2.json()["lan_access"] is False


def test_launch_injects_loopback_host_by_default(client, monkeypatch):
    import local_llm_launcher.api as api

    captured = {}
    monkeypatch.setattr(api, "find_model", lambda repo_id: {**FAKE_MODEL, "repo_id": repo_id})
    monkeypatch.setattr(api, "get_hardware", lambda: {"engines": {"llamacpp_path": None, "vllm_native": True}})
    monkeypatch.setattr(api.servers, "launch",
                        lambda mode, model, config, llamacpp_binary=None: _record(captured, mode, config))

    client.patch("/api/settings", json={"lan_access": False})
    client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "org/model-8B", "config": {}})
    assert captured["host"] == "127.0.0.1"


def test_launch_injects_lan_host_when_enabled(client, monkeypatch):
    import local_llm_launcher.api as api

    captured = {}
    monkeypatch.setattr(api, "find_model", lambda repo_id: {**FAKE_MODEL, "repo_id": repo_id})
    monkeypatch.setattr(api, "get_hardware", lambda: {"engines": {"llamacpp_path": None, "vllm_native": True}})
    monkeypatch.setattr(api.servers, "launch",
                        lambda mode, model, config, llamacpp_binary=None: _record(captured, mode, config))

    client.patch("/api/settings", json={"lan_access": True})
    client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "org/model-8B", "config": {}})
    assert captured["host"] == "0.0.0.0"


def _record(captured, mode, config):
    captured["host"] = config.get("host")
    return _StubServer()


class _StubServer:
    def status(self):
        return {"server_id": "x", "running": False, "port": 1}


FAKE_MODEL = {
    "repo_id": "org/model-8B", "path": "/snap/dir", "format": "safetensors",
    "size_bytes": 8 * 1024**3, "source": "hf-cache", "quant": None, "config": {},
    "gguf_files": [], "param_count_b": 8.0,
}


def test_rejects_non_loopback_host(client):
    r = client.get("/api/about", headers={"Host": "evil.example"})
    assert r.status_code == 400


def test_spa_fallback(client):
    r = client.get("/")
    assert r.status_code in (200, 503)  # 503 until frontend is built


def test_launch_validates_engine(client):
    r = client.post("/api/servers", json={"engine_mode": "warp-drive", "repo_id": "no/model"})
    assert r.status_code in (400, 404)


def test_stop_unknown_server_is_404(client):
    r = client.post("/api/servers/no-such-id/stop")
    assert r.status_code == 404


def test_stop_failure_is_not_404(client, monkeypatch):
    import local_llm_launcher.api as api

    monkeypatch.setattr(api.servers, "get", lambda server_id: object())  # exists
    monkeypatch.setattr(api.servers, "stop", lambda server_id: False)     # but won't die
    r = client.post("/api/servers/known/stop")
    assert r.status_code in (409, 500)
    assert r.status_code != 404
    assert "log" in r.json().get("detail", "").lower()

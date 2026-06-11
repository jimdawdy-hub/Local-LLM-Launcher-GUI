"""API smoke tests with FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

from local_llm_launcher.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


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


def test_spa_fallback(client):
    r = client.get("/")
    assert r.status_code in (200, 503)  # 503 until frontend is built


def test_launch_validates_engine(client):
    r = client.post("/api/servers", json={"engine_mode": "warp-drive", "repo_id": "no/model"})
    assert r.status_code in (400, 404)

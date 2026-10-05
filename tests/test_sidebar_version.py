"""The sidebar version must come from the server, never a hand-typed string."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "src" / "App.jsx"
ASSETS = ROOT / "src" / "local_llm_launcher" / "static" / "assets"

HARDCODED = re.compile(r"Launcher v\d+\.\d+")


def test_app_source_has_no_hardcoded_version():
    assert not HARDCODED.search(APP.read_text())


def test_app_reads_version_from_about_endpoint():
    assert "api.about()" in APP.read_text()


def test_built_bundle_has_no_hardcoded_version():
    bundles = list(ASSETS.glob("index-*.js"))
    assert bundles, "frontend bundle missing; run npm run build"
    for bundle in bundles:
        assert not HARDCODED.search(bundle.read_text()), bundle.name

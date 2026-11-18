"""Static integration guards for the actual source-built React app (not a fake preview).

These tests do not claim to replace a real npm build or browser interaction test.
"""
from pathlib import Path
from fastapi.testclient import TestClient
import main

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return (ROOT / path).read_text(encoding="utf8")


def test_only_actual_tanstack_frontend_is_packaged():
    assert (ROOT / "src/components/layout/AppShell.tsx").exists()
    assert (ROOT / "src/routes/index.tsx").exists()
    assert not (ROOT / "frontend-preview").exists()
    assert not (ROOT / "src/lib/mock-data.ts").exists()
    for path in ("/ui", "/preview", "/research", "/research/research.js"):
        assert TestClient(main.app).get(path).status_code == 404


def test_original_pixel_visual_components_present_in_real_source():
    css = read("src/styles.css")
    shell = read("src/components/layout/AppShell.tsx")
    overview = read("src/routes/index.tsx")
    for font in ("pixelify-sans", "silkscreen", "press-start-2p"):
        assert f'@fontsource/{font}' in css
    assert "PixelLandscapeBackground" in shell
    assert "app-header" in shell and "nav-tab" in shell
    for page in ("Overview", "Graph Explorer", "Transactions", "Risk Analysis", "Crash Lab", "GNN Model", "Settings"):
        assert page in shell
    assert "overview-hero" in overview and "overview-skyline" in overview
    assert "@tailwindcss/vite" in read("vite.config.ts")


def test_real_api_client_has_no_fake_fallback():
    api = read("src/services/api.ts")
    assert 'BACKEND_MODE = "LIVE"' in api
    assert 'VITE_DEMO_MODE' not in api
    assert 'mock-data' not in api
    for endpoint in ("/api/health", "/api/graph", "/api/risk/history", "/api/crash/simulate"):
        assert endpoint in api
    assert "getGraph" in api and "createFile" in api and "linkFile" in api and "syncFile" in api


def test_attention_graph_and_inspection_controls_retained():
    s = read("src/components/vfsim/ResearchGraph.tsx")
    for marker in (
        "ReactFlow", "MiniMap", "fitView", "minimum", "Minimum attention", "Show all edges",
        "Incoming effective attention", "Raw input features", "Full embedding vector",
        "model", "graphScope", "faultlabTrace", "filesystemTrace",
    ):
        assert marker.lower() in s.lower()
    assert 'useState(true)' in s  # all edges visible by default
    assert "selectedIncoming" in s
    graph = read("src/components/vfsim/GraphExplorer.tsx")
    assert 'node.id.startsWith("dirent-")' in graph
    assert "onNodeClick" in graph and "fitView" in graph
    assert "hardlink" in graph

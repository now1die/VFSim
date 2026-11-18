"""Smoke checks for the preserved backend-to-React data contract."""
from fastapi.testclient import TestClient
from main import app


def test_research_attention_trace_contract():
    with TestClient(app) as client:
        response = client.get(
            "/api/research/hardmode/trace?family=package_upgrade&version=v09_s11"
        )
        assert response.status_code == 200
        trace = response.json()
        assert trace["model"]
        assert trace["nodes"] and trace["edges"]
        assert trace["feature_width"] > 0
        assert trace["embedding_width"] > 0
        node = trace["nodes"][0]
        assert len(node["features"]) == trace["feature_width"]
        assert len(node["embedding"]["layer2"]) == trace["embedding_width"]
        assert all("layer1" in edge and "layer2" in edge and "layer3" in edge for edge in trace["edges"])



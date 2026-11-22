from fastapi.testclient import TestClient
import pytest

import main
import sessions


def test_production_temporary_sessions_and_restart(monkeypatch):
    token = "test-token-" + "a" * 32
    monkeypatch.setattr(main, "production", True)
    monkeypatch.setenv("VFSIM_PROXY_TOKEN", token)
    headers = {"X-VFSim-Proxy-Token": token}
    origin = "https://vfsim.test"

    with TestClient(main.app, base_url=origin) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["storage"] == "temporary"
        assert client.get("/api/health").status_code == 403
        assert client.get("/api/graph", headers=headers).status_code == 401
        start = client.post("/api/session", headers=headers)
        assert start.status_code == 204
        assert all(flag in start.headers["set-cookie"] for flag in ("HttpOnly", "Secure", "SameSite=lax", "Path=/"))
        original_cookie = client.cookies[sessions.SESSION_COOKIE]
        assert client.post("/api/session", headers=headers).status_code == 204
        assert client.cookies[sessions.SESSION_COOKIE] == original_cookie

        operations = (
            ("create", {"path": "/report.txt"}),
            ("write", {"path": "/report.txt", "size": 256}),
            ("link", {"existing_path": "/report.txt", "new_path": "/alias.txt"}),
            ("fsync", {"path": "/"}),
            ("rename", {"old_path": "/report.txt", "new_path": "/later.txt"}),
        )
        for name, payload in operations:
            result = client.post(f"/api/vfs/{name}", headers=headers, json=payload)
            assert result.status_code == 200, result.text
        transaction_id = result.json()["id"]

        graph = client.get("/api/graph", headers=headers).json()
        assert any(node["path"] == "/later.txt" for node in graph["nodes"])
        assert any(edge["relation"] == "hardlink" for edge in graph["edges"])
        assert len(client.get("/api/transactions", headers=headers).json()) == len(operations)
        recovery = client.post("/api/crash/simulate", headers=headers, json={
            "transactionId": transaction_id,
            "scenarioId": "crash-last-sync",
        }).json()
        assert "/report.txt" in recovery["recoveredPaths"]
        assert "/later.txt" not in recovery["recoveredPaths"]
        assert client.get(f"/api/risk/{transaction_id}", headers=headers).json()["score"] is not None
        assert client.get("/api/research/gnn?scope=full", headers=headers).status_code == 200

        database = main.app.state.sessions.database
        assert database.is_file() and database.parent.stat().st_mode & 0o777 == 0o700
        client.cookies.clear()
        assert client.post("/api/session", headers=headers).status_code == 204
        assert client.cookies[sessions.SESSION_COOKIE] != original_cookie
        assert client.get("/api/transactions", headers=headers).json() == []
        assert [node["path"] for node in client.get("/api/graph", headers=headers).json()["nodes"]] == ["/"]

    assert not database.parent.exists()
    with TestClient(main.app, base_url=origin, headers=headers) as restarted:
        assert main.app.state.sessions.database != database
        restarted.cookies.set(sessions.SESSION_COOKIE, original_cookie, domain="vfsim.test", path="/")
        assert restarted.get("/api/graph").status_code == 401
        assert restarted.post("/api/session").status_code == 204
        assert restarted.cookies[sessions.SESSION_COOKIE] != original_cookie
        assert restarted.get("/api/transactions").json() == []


def test_production_refuses_to_start_without_proxy_token(monkeypatch):
    monkeypatch.setattr(main, "production", True)
    monkeypatch.delenv("VFSIM_PROXY_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="required for hosted sessions"), TestClient(main.app):
        pass



def test_production_api_fails_closed_without_lifespan(monkeypatch):
    monkeypatch.setattr(main, "production", True)
    client = TestClient(main.app)
    response = client.post("/api/vfs/create", json={"path": "/unexpected.txt"})
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_admission_limit_and_storage_rejection(monkeypatch):
    token = "test-token-" + "b" * 32
    monkeypatch.setattr(main, "production", True)
    monkeypatch.setenv("VFSIM_PROXY_TOKEN", token)
    monkeypatch.setattr(sessions, "MAX_SESSIONS_PER_CLIENT", 1)
    headers = {"X-VFSim-Proxy-Token": token, "X-VFSim-Client-IP": "2001:db8::1"}
    with TestClient(main.app, base_url="https://vfsim.test", headers=headers) as client:
        assert client.post("/api/session").status_code == 204
        assert client.post("/api/session").status_code == 204
        monkeypatch.setattr(sessions, "MAX_STATE_BYTES", 1)
        rejected = client.post("/api/vfs/create", json={"path": "/too-large.txt"})
        assert rejected.status_code == 409
        assert client.get("/api/transactions").json() == []
        client.cookies.clear()
        throttled = client.post("/api/session", headers={"X-VFSim-Client-IP": "2001:db8::2"})
        assert throttled.status_code == 429
        assert int(throttled.headers["retry-after"]) > 0
        assert "set-cookie" not in throttled.headers
        assert client.post("/api/session", headers={"X-VFSim-Client-IP": "192.0.2.1"}).status_code == 204

import sqlite3

from fastapi.testclient import TestClient
import pytest

import main
import sessions


def test_production_session_isolation_and_restart(monkeypatch, tmp_path):
    token = "test-token-" + "a" * 32
    monkeypatch.setattr(main, "production", True)
    monkeypatch.setenv("VFSIM_PROXY_TOKEN", token)
    monkeypatch.setenv("VFSIM_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    headers = {"X-VFSim-Proxy-Token": token}
    origin = "https://vfsim.test"

    with TestClient(main.app, base_url=origin) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["storage"] == "durable"
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

        client.cookies.clear()
        assert client.post("/api/session", headers=headers).status_code == 204
        assert client.cookies[sessions.SESSION_COOKIE] != original_cookie
        assert client.get("/api/transactions", headers=headers).json() == []
        assert [node["path"] for node in client.get("/api/graph", headers=headers).json()["nodes"]] == ["/"]

    # Reopening the ASGI service creates a new store connected to the same disk.
    with TestClient(main.app, base_url=origin, headers=headers) as restarted:
        restarted.cookies.set(sessions.SESSION_COOKIE, original_cookie, path="/")
        graph = restarted.get("/api/graph").json()
        assert any(node["path"] == "/later.txt" for node in graph["nodes"])
        assert any(edge["relation"] == "hardlink" for edge in graph["edges"])
        assert len(restarted.get("/api/transactions").json()) == len(operations)
        recovery = restarted.post("/api/crash/simulate", json={
            "transactionId": transaction_id,
            "scenarioId": "crash-last-sync",
        }).json()
        assert "/report.txt" in recovery["recoveredPaths"]
        assert "/later.txt" not in recovery["recoveredPaths"]
        assert restarted.get(f"/api/risk/{transaction_id}").json()["score"] is not None
        assert restarted.get("/api/research/gnn?scope=full").status_code == 200


def test_production_refuses_to_start_without_durable_configuration(monkeypatch):
    monkeypatch.setattr(main, "production", True)
    monkeypatch.delenv("VFSIM_PROXY_TOKEN", raising=False)
    monkeypatch.delenv("VFSIM_SESSION_DB", raising=False)
    with pytest.raises(RuntimeError, match="required for hosted sessions"), TestClient(main.app):
        pass



def test_production_api_fails_closed_without_lifespan(monkeypatch):
    monkeypatch.setattr(main, "production", True)
    client = TestClient(main.app)
    response = client.post("/api/vfs/create", json={"path": "/unexpected.txt"})
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_admission_limit_and_storage_rejection(monkeypatch, tmp_path):
    token = "test-token-" + "b" * 32
    database = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(database) as db:
        # Upgrade an existing database without dropping its current rows.
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, expires INTEGER NOT NULL, state TEXT NOT NULL)")
    monkeypatch.setattr(main, "production", True)
    monkeypatch.setenv("VFSIM_PROXY_TOKEN", token)
    monkeypatch.setenv("VFSIM_SESSION_DB", str(database))
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

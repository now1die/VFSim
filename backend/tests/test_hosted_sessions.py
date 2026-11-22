from fastapi.testclient import TestClient

import main


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
        assert all(flag in start.headers["set-cookie"] for flag in ("HttpOnly", "Secure", "SameSite=lax", "Path=/api"))
        original_cookie = client.cookies["vfsim_session"]
        assert client.post("/api/session", headers=headers).status_code == 204
        assert client.cookies["vfsim_session"] == original_cookie

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
        assert client.cookies["vfsim_session"] != original_cookie
        assert client.get("/api/transactions", headers=headers).json() == []
        assert [node["path"] for node in client.get("/api/graph", headers=headers).json()["nodes"]] == ["/"]

    # Reopening the ASGI service creates a new store connected to the same disk.
    with TestClient(main.app, base_url=origin, headers=headers) as restarted:
        restarted.cookies.set("vfsim_session", original_cookie, path="/api")
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
    import pytest

    monkeypatch.setattr(main, "production", True)
    monkeypatch.delenv("VFSIM_PROXY_TOKEN", raising=False)
    monkeypatch.delenv("VFSIM_SESSION_DB", raising=False)
    with pytest.raises(RuntimeError, match="required for hosted sessions"), TestClient(main.app):
        pass

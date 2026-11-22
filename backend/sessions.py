"""Request-scoped simulator state on one worker, stored in a private temporary directory."""
from __future__ import annotations

import asyncio
from contextlib import closing
from contextvars import ContextVar
from dataclasses import asdict
from hmac import compare_digest, digest
from ipaddress import ip_address, ip_network
import json
import logging
from pathlib import Path
import re
import secrets
import sqlite3
import time

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from vfs.filesystem import VirtualFileSystem
from vfs.models import FileHandle
from vfs.transaction import Transaction

SESSION_COOKIE = "__Host-vfsim_session"
MAX_AGE = 7 * 24 * 60 * 60
MAX_SESSIONS = 64
MAX_SESSIONS_PER_CLIENT = 12
MAX_OPERATIONS = 150
MAX_STATE_BYTES = 2 * 1024 * 1024
session_vfs: ContextVar[VirtualFileSystem | None] = ContextVar("session_vfs", default=None)
logger = logging.getLogger(__name__)


def encode(vfs: VirtualFileSystem) -> str:
    return json.dumps({
        "version": 1,
        "history": vfs._history,
        "transactions": [asdict(tx) for tx in vfs.get_transactions()],
        "durable_index": vfs._durable_index,
        "file_handles": [asdict(handle) for handle in vfs.file_handles.values()],
        "next_fd": vfs.next_fd,
    }, separators=(",", ":"))


def decode(payload: str) -> VirtualFileSystem:
    saved = json.loads(payload)
    if saved["version"] != 1:
        raise ValueError("Unsupported simulator session version")
    vfs = VirtualFileSystem()
    vfs.load_snapshot(saved["history"][-1])
    vfs._history = saved["history"]
    vfs.transaction_log.transactions = [Transaction(**tx) for tx in saved["transactions"]]
    vfs._durable_index = saved["durable_index"]
    vfs.file_handles = {item["fd"]: FileHandle(**item) for item in saved["file_handles"]}
    vfs.next_fd = saved["next_fd"]
    return vfs


class SessionStore:
    def __init__(self, database: Path, proxy_token: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", proxy_token):
            raise ValueError("VFSIM_PROXY_TOKEN must contain at least 32 URL-safe characters")
        self.database = database
        self.proxy_token = proxy_token
        # Serialize hosted simulation requests. This bounds simultaneous history
        # deserialization and model inference on the 512 MiB worker.
        self.simulation_slot = asyncio.Lock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as db, db:
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise sqlite3.OperationalError("Session storage requires WAL support")
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    expires INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    client_key TEXT NOT NULL
                )
            """)

    def ready(self) -> bool:
        with closing(sqlite3.connect(f"{self.database.as_uri()}?mode=rw", uri=True, timeout=2)) as db:
            db.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
        return True

    def client_key(self, address: str) -> str:
        try:
            ip = ip_address(address)
            if ip.version == 6:
                ip = ip_network(f"{ip}/64", strict=False).network_address
            client = str(ip)
        except ValueError:
            client = "unknown"
        return digest(self.proxy_token.encode(), client.encode(), "sha256").hex()

    def start(self, session_id: str | None, client_key: str) -> tuple[str | None, int | None]:
        now = int(time.time())
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
            if session_id and db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
                return session_id, None
            count, first_expiry = db.execute(
                "SELECT COUNT(*), MIN(expires) FROM sessions WHERE client_key = ?",
                (client_key,),
            ).fetchone()
            if count >= MAX_SESSIONS_PER_CLIENT:
                return None, first_expiry - now
            if db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= MAX_SESSIONS:
                return None, None
            session_id = secrets.token_urlsafe(32)
            db.execute("INSERT INTO sessions (id, expires, state, client_key) VALUES (?, ?, ?, ?)", (
                session_id, now + MAX_AGE, encode(VirtualFileSystem()), client_key,
            ))
            return session_id, None

    def load(self, session_id: str) -> VirtualFileSystem | None:
        with closing(self.connect()) as db:
            row = db.execute("SELECT state FROM sessions WHERE id = ? AND expires > ?", (
                session_id, int(time.time()),
            )).fetchone()
        return decode(row[0]) if row else None

    def save(self, session_id: str, vfs: VirtualFileSystem) -> bool:
        payload = encode(vfs)
        if len(payload.encode("utf-8")) > MAX_STATE_BYTES:
            return False
        with closing(self.connect()) as db, db:
            if db.execute("UPDATE sessions SET state = ? WHERE id = ?", (payload, session_id)).rowcount != 1:
                raise sqlite3.OperationalError("Session disappeared during update")
        return True

    async def handle(self, request: Request, call_next):
        supplied = request.headers.get("x-vfsim-proxy-token", "")
        if not supplied.isascii() or not compare_digest(supplied, self.proxy_token):
            return _error(403, "This API is available through VFSim.")
        session_id = request.cookies.get(SESSION_COOKIE)
        if session_id and not re.fullmatch(r"[A-Za-z0-9_-]{43}", session_id):
            session_id = None

        try:
            if request.url.path == "/api/session" and request.method == "POST":
                client_key = self.client_key(request.headers.get("x-vfsim-client-ip", ""))
                new_id, retry_after = await run_in_threadpool(self.start, session_id, client_key)
                if retry_after is not None:
                    return JSONResponse(
                        {"detail": "Too many active simulator sessions from this network."},
                        status_code=429,
                        headers={"cache-control": "no-store", "retry-after": str(retry_after)},
                    )
                if new_id is None:
                    return _error(503, "The simulator is at capacity. Try again later.")
                response = Response(status_code=204, headers={"cache-control": "no-store"})
                if new_id != session_id:
                    response.set_cookie(
                        SESSION_COOKIE, new_id, max_age=MAX_AGE, path="/",
                        secure=True, httponly=True, samesite="lax",
                    )
                return response
            if request.url.path == "/api/health":
                return await call_next(request)
            if not session_id:
                return _error(401, "Start a simulator session first.")

            async with self.simulation_slot:
                vfs = await run_in_threadpool(self.load, session_id)
                if vfs is None:
                    return _error(401, "The temporary session expired or the backend restarted.")
                mutation = request.url.path.startswith("/api/vfs/") and request.method in {"POST", "DELETE"}
                if mutation and len(vfs.get_transactions()) >= MAX_OPERATIONS:
                    return _error(409, "This session reached its operation limit. Use a new browser session to continue.")
                context = session_vfs.set(vfs)
                try:
                    response = await call_next(request)
                    if mutation and response.status_code < 400 and not await run_in_threadpool(self.save, session_id, vfs):
                        return _error(409, "This session reached its storage limit.")
                    response.headers["cache-control"] = "no-store"
                    return response
                finally:
                    session_vfs.reset(context)
        except (sqlite3.Error, ValueError, KeyError):
            logger.exception("Could not read or save a VFSim session")
            return _error(503, "Simulator storage is temporarily unavailable.")


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers={"cache-control": "no-store"})

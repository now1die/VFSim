"""Optional single-port VFSim host for a *real* compiled TanStack frontend.

Usage (after npm run bundle:static): cd backend && python -m uvicorn static_host:app
No /ui preview, demo backend, or synthesized frontend is served here.
"""
import mimetypes
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from main import app

# main:app includes a JSON root used when serving API-only. Static hosting
# intentionally replaces that single route, leaving every /api path intact.
app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/"]

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_PAGES = ("graph", "transactions", "risk", "simulation", "model", "settings")

def _static_root():
    return Path(os.environ.get("VFSIM_FRONTEND_DIST", str(ROOT / "frontend-dist"))).resolve()

def _send_asset(rel):
    root = _static_root()
    if not (root / "index.html").is_file():
        raise HTTPException(503, "Run npm run bundle:static to compile the actual React application")
    p = (root / rel).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise HTTPException(404, "Frontend asset not found")
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")

@app.get("/", include_in_schema=False)
def frontend_home():
    return _send_asset("index.html")

@app.get("/assets/{asset_path:path}", include_in_schema=False)
def frontend_asset(asset_path: str):
    return _send_asset(f"assets/{asset_path}")

@app.get("/favicon.ico", include_in_schema=False)
def frontend_favicon():
    return _send_asset("favicon.ico")

@app.get("/vfsim-icon.svg", include_in_schema=False)
def frontend_icon():
    return _send_asset("vfsim-icon.svg")


@app.get("/pixel-header-clouds.svg", include_in_schema=False)
def frontend_clouds():
    return _send_asset("pixel-header-clouds.svg")


@app.get("/robots.txt", include_in_schema=False)
def frontend_robots():
    return _send_asset("robots.txt")

def _page(route: str):
    root = _static_root()
    if (root / route / "index.html").is_file():
        return _send_asset(f"{route}/index.html")
    return _send_asset("index.html")

# Real app routes only. API routes are defined in main.py and remain unchanged.
for _route in ALLOWED_PAGES:
    app.add_api_route(f"/{_route}", lambda r=_route: _page(r), methods=["GET"], include_in_schema=False)

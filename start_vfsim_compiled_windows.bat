@echo off
setlocal
cd /d "%~dp0"
if not exist "frontend-dist\index.html" (
  echo No compiled VFSim frontend. Run: npm ci ^&^& npm run bundle:static
  pause
  exit /b 1
)
if not exist "frontend-dist\vfsim-build.json" (
  echo Compiled frontend has no VFSim build manifest; refusing to serve stale assets.
  pause
  exit /b 1
)
cd backend
python -m uvicorn static_host:app --host 127.0.0.1 --port 8000

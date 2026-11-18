@echo off
setlocal
cd /d "%~dp0"
if not exist "node_modules" (
  echo Frontend dependencies are missing. Run npm install in this folder first.
  pause
  exit /b 1
)
start "VFSim FastAPI" /D "%~dp0backend" cmd /k python -m uvicorn main:app --host 127.0.0.1 --port 8000
start "VFSim React" /D "%~dp0" cmd /k npm run dev
start "" "http://127.0.0.1:3000"

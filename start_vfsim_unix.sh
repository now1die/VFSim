#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d node_modules ]]; then echo "Run npm install first (frontend dependencies missing)."; exit 1; fi
(cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000) &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
npm run dev

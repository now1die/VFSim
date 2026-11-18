#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -f frontend-dist/index.html && -f frontend-dist/vfsim-build.json ]] || { echo "Run npm ci && npm run bundle:static first"; exit 1; }
cd backend
python -m uvicorn static_host:app --host 127.0.0.1 --port 8000

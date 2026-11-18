"""Publish a real TanStack prerendered client build for optional FastAPI hosting.

Fails closed: never copy old v1.1 assets or manufacture an index.html.
"""
from pathlib import Path
import shutil
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dist" / "client"
DEST = ROOT / "frontend-dist"

def main():
    if not (SOURCE / "index.html").is_file():
        raise SystemExit("No real TanStack client build at dist/client/index.html; run npm run build first")
    if not (SOURCE / "assets").is_dir() or not any((SOURCE / "assets").iterdir()):
        raise SystemExit("Build is missing compiled assets; refusing to publish")
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(SOURCE, DEST)
    version = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]
    (DEST / "vfsim-build.json").write_text(json.dumps({
      "version": version,
      "source": "TanStack Start production build",
      "published_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"Published real TanStack build: {DEST}")
if __name__ == "__main__":
    main()

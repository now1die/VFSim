"""Validate this extracted VFSim installation from its project root."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
main_file = root / 'backend' / 'main.py'
frontend_file = root / 'src' / 'components' / 'layout' / 'AppShell.tsx'
lockfile = root / 'package-lock.json'
artifacts = root / 'backend' / 'artifacts'

for path in (main_file, frontend_file, lockfile, artifacts):
    if not path.exists():
        sys.exit(f'FAILED: Missing {path}. Extract the entire archive into an EMPTY folder.')

source = main_file.read_text(encoding='utf-8')
if 'from gnn.inference import RiskGNNInference' in source:
    sys.exit('FAILED: An obsolete backend/main.py was found. Do not merge into an old installation.')
if 'from vfs.filesystem import VirtualFileSystem' not in source:
    sys.exit('FAILED: backend/main.py does not match this release.')

sys.path.insert(0, str(root / 'backend'))
try:
    import main
    from fastapi.testclient import TestClient
except Exception as exc:
    sys.exit(f'FAILED: Cannot import backend: {exc!r}\nInstall dependencies: python -m pip install -r backend/requirements.txt')

actual_file = Path(main.__file__).resolve()
if actual_file != main_file.resolve():
    sys.exit(f'FAILED: Imported WRONG backend file: {actual_file}')

try:
    result = TestClient(main.app).get('/api/health')
    result.raise_for_status()
except Exception as exc:
    sys.exit(f'FAILED: Backend /api/health did not respond: {exc!r}')

checkpoints = sorted(artifacts.rglob('*.pt'))
if len(checkpoints) < 8:
    sys.exit(f'FAILED: Only {len(checkpoints)} trained checkpoint files found; archive may be incomplete.')

print('PASS: VFSim Backend imported from:', actual_file)
print('PASS: App title:', main.app.title)
print('PASS: /api/health status:', result.status_code)
print('PASS: Trained checkpoint files:', len(checkpoints))
print('PASS: Correct project structure; ready to start Uvicorn.')

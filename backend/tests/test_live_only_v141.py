"""Regression guards for a real-operation-only React frontend."""
from pathlib import Path
from fastapi.testclient import TestClient
import main

ROOT = Path(__file__).resolve().parents[2]

def test_no_sample_or_mock_apis():
    response = TestClient(main.app).post('/api/research/sample')
    assert response.status_code == 404
    assert 'loadResearchSample' not in (ROOT/'src/services/api.ts').read_text(encoding='utf-8')
    assert 'isDemo' not in (ROOT/'src/components/vfsim/BackendStatus.tsx').read_text(encoding='utf-8')
    assert not (ROOT/'src/lib/mock-data.ts').exists()

def test_real_operations_are_connected_to_frontend():
    src = (ROOT/'src/components/vfsim/LiveOperations.tsx').read_text(encoding='utf-8')
    for name in ('createFile','writeFile','linkFile','renameFile','unlinkFile','syncFile'):
        assert 'api.'+name in src
    assert 'LiveOperations' in (ROOT/'src/routes/index.tsx').read_text(encoding='utf-8')

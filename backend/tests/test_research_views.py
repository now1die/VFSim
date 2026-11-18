"""Evidence-facing tests: no fabricated model attention, scores, or crash labels."""
from collections import defaultdict
from fastapi.testclient import TestClient
from vfs.filesystem import VirtualFileSystem
import main


def setup_model_vfs():
    v = VirtualFileSystem()
    v.create('/home', 'directory')
    v.create('/home/notes.txt')
    v.write('/home/notes.txt', 1024)
    v.fsync('/')
    link_tx = v.link('/home/notes.txt', '/home/notes-copy.txt')
    v.rename('/home/notes.txt', '/home/draft.txt')
    v.open('/home/draft.txt', 'r')
    return v, link_tx


def test_scoring_subgraph_matches_production_inference(monkeypatch):
    v, tx = setup_model_vfs()
    monkeypatch.setattr(main, 'vfs', v)
    client = TestClient(main.app)
    payload = client.get('/api/research/gnn', params={'transaction_id': tx.txn_id}).json()
    score = client.get('/api/risk/' + tx.txn_id).json()['score']
    assert payload['graphScope'] == 'scoring'
    assert abs(payload['score'] - score) < 0.00011
    assert payload['touched']
    assert all(node_id in {n['id'] for n in payload['nodes']} for node_id in payload['touched'])
    assert payload['attentionDescription'].startswith('Actual single-head')
    assert payload['trainingWarning']
    for node in payload['nodes']:
        assert len(node['features']) >= 1
        assert len(node['embeddings']['encoded']) == 32
        assert len(node['embeddings']['layer1']) == 32
        assert len(node['embeddings']['layer2']) == 32
    for edge in payload['edges']:
        assert edge['source'] in {node['id'] for node in payload['nodes']}
        assert edge['target'] in {node['id'] for node in payload['nodes']}
        for layer in ['layer1', 'layer2']:
            assert edge['attention'][layer] is not None
            assert 0 <= edge['attention'][layer] <= 1.00001


def test_all_five_model_node_types_visible_in_full_graph(monkeypatch):
    v, _ = setup_model_vfs()
    monkeypatch.setattr(main, 'vfs', v)
    payload = TestClient(main.app).get('/api/research/gnn?scope=full').json()
    assert payload['graphScope'] == 'full'
    assert payload['score'] is None  # Do not misrepresent full-state trace as risk prediction
    assert payload['touched'] == []
    assert all(payload['counts'][name] > 0 for name in ('inode', 'dirent', 'fd', 'link', 'rename'))
    assert sum(payload['counts'].values()) == len(payload['nodes'])
    assert any(edge['relation'] == 'opened_by' for edge in payload['edges'])
    assert any(edge['relation'] == 'rename_target' for edge in payload['edges'])
    assert any(edge['relation'] == 'has_link' for edge in payload['edges'])
    assert len(payload['edges']) > 10
    assert all(len(node['embeddings']['layer2']) == 32 for node in payload['nodes'])


def test_attention_denominators_track_real_message_aggregation(monkeypatch):
    v, _ = setup_model_vfs()
    monkeypatch.setattr(main, 'vfs', v)
    payload = TestClient(main.app).get('/api/research/gnn?scope=full').json()
    for layer in ('layer1', 'layer2'):
        by_target = defaultdict(float)
        for edge in payload['edges']:
            by_target[edge['target']] += edge['attention'][layer]
        # The original model uses max(sum incoming raw weights, 1) as its denominator.
        # Thus sums can be below one, but never above one (up to serialization rounding).
        assert all(0 < value <= 1.00002 for value in by_target.values())


def test_transaction_analysis_is_consistent_and_nondestructive(monkeypatch):
    v, tx = setup_model_vfs()
    monkeypatch.setattr(main, 'vfs', v)
    before = v.snapshot()
    result = TestClient(main.app).get('/api/research/transactions/' + tx.txn_id)
    assert result.status_code == 200
    payload = result.json()
    assert payload['operation'] == 'link'
    assert payload['namespaceDelta']['createdPaths'] == ['/home/notes-copy.txt']
    assert len(payload['crashPreviews']) == 3
    assert all(item['consistencyPassed'] for item in payload['crashPreviews'])
    assert 'not physical' in payload['disclaimer'].lower()
    assert v.snapshot() == before


def test_research_endpoint_errors_and_no_fake_preview(monkeypatch):
    monkeypatch.setattr(main, 'vfs', VirtualFileSystem())
    client = TestClient(main.app)
    assert client.get('/api/research/gnn?scope=incorrect').status_code == 400
    assert client.get('/api/research/gnn?transaction_id=missing').status_code == 404
    empty = client.get('/api/research/gnn?scope=full').json()
    assert empty['score'] is None
    assert empty['counts']['inode'] == 1
    # The only UI is the actual TanStack Start app, started independently on port 3000.
    for endpoint in ('/ui', '/preview', '/research', '/research/research.js'):
        assert client.get(endpoint).status_code == 404

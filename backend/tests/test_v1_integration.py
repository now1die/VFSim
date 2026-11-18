"""VFSim v1.0 integration contract for the merged React frontend."""
from fastapi.testclient import TestClient
import main
from vfs.filesystem import VirtualFileSystem



def test_live_vfs_operations_preserve_hardlinks_and_feed_real_graph(monkeypatch):
    monkeypatch.setattr(main, 'vfs', VirtualFileSystem())
    client = TestClient(main.app)
    assert client.get('/api/stats').json()['nodes'] == 1
    assert client.post('/api/research/sample').status_code == 404
    assert client.post('/api/vfs/create', json={'path':'/workspace','inode_type':'directory'}).status_code == 200
    assert client.post('/api/vfs/create', json={'path':'/workspace/report.txt','inode_type':'file'}).status_code == 200
    assert client.post('/api/vfs/write', json={'path':'/workspace/report.txt','size':1400}).status_code == 200
    assert client.post('/api/vfs/link', json={'existing_path':'/workspace/report.txt','new_path':'/workspace/alias.txt'}).status_code == 200
    assert client.post('/api/vfs/write', json={'path':'/workspace/alias.txt','size':160}).status_code == 200
    assert main.vfs.get_inode('/workspace/alias.txt') is main.vfs.get_inode('/workspace/report.txt')
    assert client.post('/api/vfs/fsync',json={'path':'/workspace/report.txt'}).status_code == 200
    assert main.vfs.validate()
    assert len(main.vfs.get_transactions()) == 6
    assert client.get('/api/stats').json()['nodes'] > 1
    graph = client.get('/api/research/gnn?scope=full')
    assert graph.status_code == 200
    assert graph.json()['counts']['link'] >= 1
    assert len(graph.json()['edges']) > 0
    assert client.get('/api/model').json()['status'] == 'checkpoint_loaded'


def test_synthetic_benchmark_access_remains_unchanged():
    response=TestClient(main.app).get('/api/research/independent-v09')
    assert response.status_code==200
    info=response.json()
    assert info['external_cohort']['summary']['count']==1440
    assert len(info['registered_config']['training_seeds'])==5
    assert 'mae_to_oracle' in info['summary']['models']['GNN v0.8 operand-aware']

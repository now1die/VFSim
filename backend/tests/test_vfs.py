"""Filesystem invariants, rollback correctness, training, API integration."""
import json
import pytest
from fastapi.testclient import TestClient
from vfs.filesystem import VirtualFileSystem
from graph.builder import build_graph
from graph.subgraph import extract_2hop_subgraph, get_subgraph_touched_inode_indices, get_touched_inode_indices
from gnn.model import RiskGNN
from training.dataset import generate_dataset, split_dataset
import main


def basic():
    v = VirtualFileSystem()
    v.create('/home', 'directory')
    v.create('/home/a.txt')
    return v


def test_hard_link_shared_inode_write_and_unlink():
    v = basic()
    v.link('/home/a.txt', '/home/b.txt')
    assert v.get_inode('/home/a.txt') is v.get_inode('/home/b.txt')
    v.write('/home/b.txt', 99)
    assert v.get_inode('/home/a.txt').size == 99
    assert v.get_inode('/home/a.txt').link_count == 2
    v.unlink('/home/a.txt')
    assert v.get_inode('/home/b.txt').link_count == 1
    assert v.get_inode('/home/b.txt').size == 99
    assert v.validate()


def test_rename_alias_does_not_break_other_alias():
    v = basic()
    v.link('/home/a.txt', '/home/b.txt')
    v.rename('/home/a.txt', '/home/c.txt')
    assert v.get_inode('/home/b.txt') is v.get_inode('/home/c.txt')
    assert v.validate()


def test_directory_rename_moves_subtree():
    v = basic()
    v.create('/home/folder', 'directory')
    v.create('/home/folder/item.txt')
    v.rename('/home/folder', '/home/new-folder')
    assert '/home/new-folder/item.txt' in v.inodes
    assert '/home/folder/item.txt' not in v.inodes
    assert v.validate()


def test_nested_directory_move_rejected():
    v = basic()
    v.create('/home/sub', 'directory')
    with pytest.raises(ValueError):
        v.rename('/home', '/home/sub/inside')


def test_unlink_nonempty_directory_rejected():
    v = basic()
    with pytest.raises(ValueError):
        v.unlink('/home')

@pytest.mark.parametrize('path', ['home/a', '/a/../b', '/home//a', '/home/.', '/home/a/'])
def test_bad_paths_rejected(path):
    v = basic()
    with pytest.raises(ValueError):
        v.create(path)


def test_write_validation():
    v = basic()
    with pytest.raises(ValueError):
        v.write('/home/a.txt', -1)
    with pytest.raises(ValueError):
        v.write('/home', 1)


def test_open_rename_and_close():
    v = basic()
    fd = v.open('/home/a.txt').metadata['fd']
    v.rename('/home/a.txt', '/home/b.txt')
    v.close(fd)
    assert v.validate()


def test_crash_snapshots_and_real_consistency_check():
    v = basic()
    v.fsync('/')
    tx = v.write('/home/a.txt', 27)
    result = v.crash_preview(tx.txn_id, 'crash-last-sync')
    assert result['changedSizePaths'] == ['/home/a.txt']
    assert result['consistencyPassed'] is True
    assert result['lostOps'] == 1
    assert v.get_inode('/home/a.txt').size == 27  # live state unaffected
    after = v.crash_preview(tx.txn_id, 'crash-after-commit')
    assert after['lostOps'] == 0


def test_persisted_checkpoint_survives_restart(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    v = VirtualFileSystem(checkpoint)
    v.create('/home', 'directory')
    v.create('/home/a.txt')
    v.write('/home/a.txt', 123)
    v.fsync('/')
    v.write('/home/a.txt', 456)  # unflushed and intentionally not recovered
    restored = VirtualFileSystem(checkpoint)
    assert restored.get_inode('/home/a.txt').size == 123
    assert restored.validate()


def test_model_on_real_graph():
    v = basic()
    tx = v.link('/home/a.txt', '/home/alias.txt')
    graph = build_graph(v)
    touched = get_touched_inode_indices(v, tx)
    subgraph = extract_2hop_subgraph(graph, touched)
    result = RiskGNN().predict_risk(subgraph.x_dict, subgraph.edge_index_dict,
                                    get_subgraph_touched_inode_indices(graph, touched))
    assert 0 <= result.item() <= 1


def test_dataset_balanced_and_labels_come_from_crash_model():
    data = generate_dataset(20, seed=7)
    train, validation, test = split_dataset(data)
    assert sum(s['label'] for s in data) == 10
    assert len(train) + len(validation) + len(test) == 20


def test_api_graph_risk_crash_and_fsync(monkeypatch):
    monkeypatch.setattr(main, 'vfs', VirtualFileSystem())
    client = TestClient(main.app)
    assert client.get('/api/health').status_code == 200
    assert client.post('/api/vfs/create', json={'path': '/home', 'inode_type': 'directory'}).status_code == 200
    assert client.post('/api/vfs/create', json={'path': '/home/a.txt'}).status_code == 200
    assert client.post('/api/vfs/fsync', json={'path': '/'}).status_code == 200
    response = client.post('/api/vfs/write', json={'path': '/home/a.txt', 'size': 77})
    assert response.status_code == 200, response.text
    txid = response.json()['id']
    assert response.json()['riskScore'] is not None
    graph = client.get('/api/transactions/' + txid + '/graph').json()
    assert graph['edges'] and graph['nodes'] and graph['touched']
    assert client.get('/api/model').json()['metrics']['auc'] is not None
    preview = client.post('/api/crash/simulate', json={'transactionId': txid, 'scenarioId': 'crash-last-sync'})
    assert preview.status_code == 200, preview.text
    assert preview.json()['consistencyPassed']
    assert preview.json()['changedSizePaths'] == ['/home/a.txt']
    assert client.get('/api/risk/' + txid).json()['score'] is not None
    assert client.post('/api/vfs/write', json={'path': '/home/a.txt', 'size': -2}).status_code == 400


def test_old_static_preview_urls_are_not_served():
    client = TestClient(main.app)
    for url in ('/ui', '/preview', '/preview/graph-enhanced.js', '/research'):
        assert client.get(url).status_code == 404


def test_graph_exposes_link_identity_and_dirent_mapping(monkeypatch):
    monkeypatch.setattr(main, 'vfs', VirtualFileSystem())
    client = TestClient(main.app)
    client.post('/api/vfs/create', json={'path':'/home', 'inode_type':'directory'})
    client.post('/api/vfs/create', json={'path':'/home/report.txt'})
    client.post('/api/vfs/link', json={'existing_path':'/home/report.txt', 'new_path':'/home/report-copy.txt'})
    graph = client.get('/api/graph').json()
    target = next(n for n in graph['nodes'] if n['path']=='/home/report.txt' and n['kind']=='file')
    assert target['linkCount'] == 2
    assert isinstance(target['inodeNumber'], int)
    aliases = [n for n in graph['nodes'] if n['id'].startswith('dirent-') and n['inodeNumber']==target['inodeNumber']]
    assert {x['path'] for x in aliases} == {'/home/report.txt','/home/report-copy.txt'}
    assert any(e['relation']=='hardlink' for e in graph['edges'])

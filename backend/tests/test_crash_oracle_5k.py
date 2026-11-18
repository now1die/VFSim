"""Regression coverage for the 5,000-case synthetic crash-invariant benchmark."""
from collections import Counter
import json
import torch
from fastapi.testclient import TestClient
from vfs.filesystem import VirtualFileSystem
from graph.builder import FEATURE_WIDTHS, RELATIONS, build_graph
from training.dataset import (generate_dataset, split_family_heldout, invariant_holds,
                              TRAIN_FAMILIES, OOD_FAMILIES)
from training.baselines import graph_vector, prioritize
from gnn.model import RiskGNN
import main


def test_oracle_can_detect_mismatch_after_intermediate_fsync():
    v = VirtualFileSystem()
    v.create('/a')
    v.create('/b')
    v.declare_pair('/a', '/b')
    v.fsync('/')
    v.write('/a', 256)
    v.fsync('/')
    tx = v.write('/b', 256)
    recovered = v._history[v.crash_preview(tx.txn_id, 'crash-last-sync')['survivedOps']]
    assert invariant_holds(v.snapshot(), ('/a', '/b'), 'size_equal')
    assert not invariant_holds(recovered, ('/a', '/b'), 'size_equal')
    assert v.validate()


def test_unsynced_pair_rollback_does_not_violate_invariant():
    v = VirtualFileSystem()
    v.create('/a')
    v.create('/b')
    v.declare_pair('/a', '/b')
    v.fsync('/')
    v.write('/a', 256)
    tx = v.write('/b', 256)
    recovered = v._history[v.crash_preview(tx.txn_id, 'crash-last-sync')['survivedOps']]
    assert invariant_holds(recovered, ('/a', '/b'), 'size_equal')
    assert all(i.dirty for i in v.get_all_inodes() if i.path in ('/a','/b'))


def test_family_samples_have_real_oracle_labels_and_hardlinks():
    samples = generate_dataset(200, seed=31)
    assert set(s['family'] for s in samples) == set(TRAIN_FAMILIES) | set(OOD_FAMILIES)
    labels = Counter((s['family'], s['label']) for s in samples)
    assert all(labels[(family, label)] > 0 for family in TRAIN_FAMILIES + OOD_FAMILIES for label in (0,1))
    assert sum(s['label'] for s in samples) == 100
    assert any(s['scenario'] == 'unlink' for s in samples)
    assert any(s['scenario'] == 'link' for s in samples)
    assert any(s['dirty_rule'] == 1 and s['label'] == 0 for s in samples)


def test_family_holdout_never_leaks_into_training():
    train, val, it, ood = split_family_heldout(generate_dataset(200, seed=11), seed=11)
    assert all(s['family'] in TRAIN_FAMILIES for group in (train,val,it) for s in group)
    assert all(s['family'] in OOD_FAMILIES for s in ood)
    assert len(train) + len(val) + len(it) + len(ood) == 200
    assert train and val and it and ood


def test_pair_graph_and_real_hardlink_semantics():
    v = VirtualFileSystem()
    v.create('/a'); v.create('/b')
    v.link('/a','/alias-a')
    v.declare_pair('/a','/b')
    v.fsync('/')
    v.write('/alias-a',12)
    assert v.get_inode('/a') is v.get_inode('/alias-a')
    assert v.get_inode('/alias-a').size == 12
    graph = build_graph(v)
    assert FEATURE_WIDTHS['inode'] == 9
    assert graph.edge_index_dict[('inode','application_pair','inode')].shape[1] == 2
    assert len(RELATIONS) == 9
    assert graph.x_dict['inode'][:,6].max() == 0  # no durable file data at last fsync


def test_historical_replay_maintains_fsync_features():
    v = VirtualFileSystem()
    v.create('/a'); v.create('/b');v.declare_pair('/a','/b');v.fsync('/')
    v.write('/a',40);v.fsync('/');v.write('/b',40)
    replay = v.replay_at(len(v.get_transactions()))
    assert replay._durable_index == v._durable_index
    assert torch.equal(build_graph(replay).x_dict['inode'],build_graph(v).x_dict['inode'])


def test_readout_trace_matches_actual_forward():
    sample = next(s for s in generate_dataset(40,seed=21) if s['family']=='size_pair' and s['label'])
    g = sample['graph']
    net = RiskGNN().eval()
    prediction = float(torch.sigmoid(net(g.x_dict,g.edge_index_dict,sample['touched_inodes'])).item())
    trace = net.inspect(g.x_dict,g.edge_index_dict,sample['touched_inodes'])
    assert abs(prediction - trace['score']) < 1e-6
    assert len(trace['pair_contrasts']) == 9
    assert sum(trace['pair_contrasts']) > 0


def test_prioritization_never_uses_labels_for_sorting():
    labels = [0,1,0,1,0]
    result = prioritize(labels,[.9,.8,.7,.6,.1],budgets=(.2,.4))
    assert result['tests_to_first_violation'] == 2
    assert result['budgets']['20pct']['violations_found'] == 0
    assert result['budgets']['40pct']['violations_found'] == 1


def test_benchmark_endpoint_and_pair_registration(monkeypatch):
    monkeypatch.setattr(main,'vfs',VirtualFileSystem())
    client = TestClient(main.app)
    result = client.get('/api/research/benchmark')
    assert result.status_code == 200
    payload = result.json()
    assert payload['gnn']['samples'] == 5000
    assert payload['gnn']['split']['train'] == 3000
    assert payload['gnn']['split']['heldout_family_test'] == 1000
    assert 'XGBoost' in payload['baselines']
    assert client.post('/api/vfs/create',json={'path':'/a'}).status_code == 200
    assert client.post('/api/vfs/create',json={'path':'/b'}).status_code == 200
    assert client.post('/api/vfs/pair',json={'first':'/a','second':'/b'}).status_code == 200
    full = client.get('/api/research/gnn?scope=full').json()
    assert full['relations']['inode__application_pair__inode'] == 2
    assert full['pairContrastReadout'] is not None


def test_priority_endpoint_ranks_real_model_scores(monkeypatch):
    v=VirtualFileSystem();v.create('/a');v.create('/b');v.declare_pair('/a','/b');v.fsync('/')
    v.write('/a',32);v.fsync('/');v.write('/b',32)
    monkeypatch.setattr(main,'vfs',v)
    api=TestClient(main.app)
    result=api.get('/api/research/prioritize?limit=5')
    assert result.status_code==200
    rows=result.json()['candidates']
    assert rows and all(rows[i]['score']>=rows[i+1]['score'] for i in range(len(rows)-1))
    assert all(row['transactionId'] in [tx.txn_id for tx in v.get_transactions()] for row in rows)
    assert api.get('/api/research/prioritize?limit=101').status_code==400


def test_model_crash_preview_reports_application_violation_separately():
    v=VirtualFileSystem();v.create('/a');v.create('/b')
    v.declare_pair('/a','/b','size_equal');v.fsync('/')
    v.write('/a',64);v.fsync('/');tx=v.write('/b',64)
    result=v.crash_preview(tx.txn_id,'crash-last-sync')
    assert result['consistencyPassed'] is True
    assert result['applicationInvariantsPassed'] is False
    assert result['violatedPairs'][0]['paths']==['/a','/b']
    assert v.crash_preview(tx.txn_id,'crash-after-commit')['applicationInvariantsPassed'] is True

"""Regression checks for outcome-based labels, held-out families and benchmark integrity."""
from fastapi.testclient import TestClient
from vfs.filesystem import VirtualFileSystem
from training.dataset import generate_dataset, split_family_heldout, invariant_holds
from training.baselines import prioritize, graph_vector
import main


def test_generator_has_both_outcomes_for_each_family():
    samples = generate_dataset(120, seed=9)
    families = {s['family'] for s in samples}
    assert families == {'size_pair', 'create_pair', 'link_pair', 'mixed', 'rename_pair', 'unlink_pair'}
    for family in families:
        assert {s['label'] for s in samples if s['family'] == family} == {0, 1}
        assert all(len(graph_vector(s)) > 0 for s in samples if s['family'] == family)


def test_family_heldout_is_not_a_random_leakage_split():
    samples = generate_dataset(120, seed=10)
    train, validation, id_test, ood_test = split_family_heldout(samples, seed=10)
    assert len(train) + len(validation) + len(id_test) + len(ood_test) == len(samples)
    seen = {'size_pair', 'create_pair', 'link_pair', 'mixed'}
    unseen = {'rename_pair', 'unlink_pair'}
    assert set(s['family'] for s in train + validation + id_test) == seen
    assert set(s['family'] for s in ood_test) == unseen


def test_safe_risky_label_depends_on_recovery_invariant():
    v = VirtualFileSystem()
    v.create('/a');v.create('/b');v.write('/a', 100);v.write('/b', 100)
    v.fsync('/')
    v.write('/a', 25)
    v.fsync('/')
    tx = v.write('/b', 25)
    preview = v.crash_preview(tx.txn_id, 'crash-last-sync')
    assert invariant_holds(v.snapshot(),('/a','/b'),'size_equal')
    assert not invariant_holds(v._history[preview['survivedOps']],('/a','/b'),'size_equal')


def test_ranking_reports_fixed_budget():
    report = prioritize([0,1,0,1,0,1,0,1,0,1], [0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6,0.5,0.55])
    assert report['tests_to_first_violation'] == 1
    assert report['budgets']['20pct']['violations_found'] == 2
    assert report['budgets']['20pct']['tests'] == 2


def test_live_pair_and_benchmark_endpoint(monkeypatch):
    v=VirtualFileSystem();v.create('/a');v.create('/b')
    monkeypatch.setattr(main, 'vfs', v)
    client=TestClient(main.app)
    assert client.post('/api/vfs/pair',json={'first':'/a','second':'/b'}).status_code == 200
    assert v.risk_pairs and v.risk_pairs[0]['inode_a'] != v.risk_pairs[0]['inode_b']
    response=client.get('/api/research/benchmark')
    assert response.status_code == 200
    body=response.json()
    assert body['gnn']['samples'] == 5000
    assert body['gnn']['split']['heldout_family_test'] == 1000
    assert 'Random Forest' in body['baselines']
    assert 'XGBoost' in body['baselines']
    assert 'not ext4' in body['disclaimer']

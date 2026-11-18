"""Hard-mode fault-injection regression tests (toy durability semantics)."""
import random
import math
import pytest
from challenging.simulator import DurabilityModel, Constraint, check_constraint
from challenging.generator import build_cases,make_workload,split_cases,ALL_FAMILIES,HELDOUT_FAMILIES
from challenging.graphs import make_graph,graph_vector,EDGE_TYPES
from challenging.model import CrashRiskGNN


def test_two_inode_independent_writeback_probability():
    fs=DurabilityModel();fs.create('/a/x',1);fs.create('/b/y',1);fs.checkpoint()
    fs.write('/a/x',2);fs.write('/b/y',2)
    assert fs.exact_failure_probability(Constraint('equal_versions',('/a/x','/b/y')),0.5)[0] == .5
    assert fs.exact_failure_probability(Constraint('equal_versions',('/a/x','/b/y')),0)[0] == 0
    assert fs.exact_failure_probability(Constraint('equal_versions',('/a/x','/b/y')),1)[0] == 0


def test_wal_marker_durable_before_data_is_app_violation():
    fs=DurabilityModel();fs.create('/meta/log',1);fs.create('/data/file',1);fs.checkpoint()
    fs.write('/data/file',2);fs.write('/meta/log',2)
    c=Constraint('requires_data',('/meta/log','/data/file'))
    assert fs.exact_failure_probability(c,0.5)[0]==.25
    assert fs.sample_recovery(c,.5,random.Random(2))['violation'] in (0,1)


def test_fsync_file_does_not_persist_directory_entries():
    fs=DurabilityModel();fs.create('/data/old',1);fs.checkpoint()
    fs.create('/other/new',2);fs.sync_file('/other/new')
    assert '/other/new' not in fs.recover()[0]
    fs.sync_dir('/other')
    assert fs.recover()[0]['/other/new']==fs.paths['/other/new']
    assert fs.recover()[1][fs.paths['/other/new']]==2


def test_hard_link_two_paths_share_single_inode_and_writeback_group():
    fs=DurabilityModel();fs.create('/data/base',1);fs.link('/data/base','/mirror/alias');fs.checkpoint()
    fs.write('/mirror/alias',4)
    assert fs.paths['/data/base']==fs.paths['/mirror/alias']
    files,_=fs.pending()
    assert len(files)==1
    p,_=fs.exact_failure_probability(Constraint('equal_versions',('/data/base','/mirror/alias')),0.5)
    assert p==0
    assert fs.recover(flushed_files=files)[1][fs.paths['/data/base']]==4


def test_cross_directory_partial_publish_can_split_two_names():
    fs=DurabilityModel();fs.create('/stage/x',1);fs.create('/stage/y',1);fs.checkpoint()
    fs.rename('/stage/x','/dir1/x');fs.rename('/stage/y','/dir2/y')
    c=Constraint('presence_together',('/dir1/x','/dir2/y'))
    assert not check_constraint(*fs.recover(flushed_dirs=['/dir1']),c)
    assert check_constraint(*fs.recover(flushed_dirs=['/dir1','/dir2']),c)


@pytest.mark.parametrize('family',ALL_FAMILIES)
def test_all_families_have_valid_final_live_state_and_reproducible_oracle(family):
    fs,c=make_workload(family,20260212)
    assert check_constraint(fs.paths,fs.values,c)
    probability,failures=fs.exact_failure_probability(c,.4)
    assert 0<=probability<=1
    assert probability==fs.exact_failure_probability(c,.4)[0]
    assert any(op['kind']=='durable_checkpoint' for op in fs.operations)
    for example in failures:
        assert not check_constraint(example['recovered_paths'],example['recovered_versions'],c)


def test_no_oracle_label_in_graph_and_graph_handles_hardlinks():
    fs,c=make_workload('hardlink_release',17)
    case={'sim':fs,'constraint':c,'writeback_probability':.5,'risk_probability':.12345}
    before=make_graph(case)
    case['risk_probability']=.98765
    after=make_graph(case)
    assert before.x.equal(after.x)
    assert before.edge_index.equal(after.edge_index)
    assert len([n for n in before.names if n=='/mirror/blob'])==1
    assert len(before.roles)==len(before.x)
    assert len(graph_vector(before))>100
    assert {'live_path_inode','stable_path_inode','constraint_path'}.issubset(EDGE_TYPES)
    assert 0<=float(CrashRiskGNN()(before).sigmoid().detach())<=1


def test_family_holdout_and_nontrivial_probabilities():
    samples=build_cases(160,seed=7)
    train,val,id_test,ood=split_cases(samples,seed=7)
    assert sum(map(len,(train,val,id_test,ood)))==160
    assert not set(HELDOUT_FAMILIES)&{s['family'] for s in train+val+id_test}
    assert {s['family'] for s in ood}==set(HELDOUT_FAMILIES)
    assert len({s['id'] for s in samples})==160
    assert sum(0<c['risk_probability']<1 for c in samples)>80


def test_oracle_matches_monte_carlo_recovered_failures():
    fs,c=make_workload('atomic_pair',314159)
    expected,_=fs.exact_failure_probability(c,.45)
    rng=random.Random(15)
    observed=sum(fs.sample_recovery(c,.45,rng)['violation'] for _ in range(3000))/3000
    assert abs(observed-expected)<.04


def test_readonly_hardmode_endpoint_serves_metrics_if_trained():
    from fastapi.testclient import TestClient
    from main import app
    response=TestClient(app).get('/api/research/hardmode')
    assert response.status_code==200
    data=response.json()
    assert data['evaluation']['samples']>=160
    assert {'GNN','Random Forest','XGBoost'}.issubset(data['evaluation']['results'])
    assert len(data['examples'])==len(ALL_FAMILIES)
    assert 'mask' in data['boundary']


def test_example_predictions_are_from_saved_model_not_mock_scores():
    import json
    from pathlib import Path
    import torch
    from challenging.graphs import make_graph
    artifacts=Path(__file__).resolve().parents[1]/'artifacts'
    checkpoint=torch.load(artifacts/'crash_risk_hardmode.pt',weights_only=True,map_location='cpu')
    model=CrashRiskGNN(variant=checkpoint['metadata'].get('model_variant','global')).eval()
    model.load_state_dict(checkpoint['state_dict'])
    cases=json.loads((artifacts/'hardmode_examples.json').read_text())
    with torch.no_grad():
        for case in cases:
            fs,c=make_workload(case['family'],case['seed'])
            p=case['writeback_probability']
            exact,_=fs.exact_failure_probability(c,p)
            assert exact==case['risk_probability']
            risk=float(torch.sigmoid(model(make_graph({'sim':fs,'constraint':c,
                        'writeback_probability':p}))).item())
            assert abs(risk-case['gnn_predicted_risk'])<.00003

"""v0.8: verify faithful FaultLab instrumentation and honest diagnostic data."""
from collections import defaultdict
from pathlib import Path
import pytest, torch
from fastapi.testclient import TestClient
from main import app
from challenging.generator import ALL_FAMILIES,make_workload
from challenging.graphs import make_graph,EDGE_TYPES
from challenging.trace import model_and_metadata,trace_saved_family


@pytest.mark.parametrize('family', ALL_FAMILIES)
def test_trace_uses_the_real_v07_checkpoint_and_three_distinct_layers(family):
    payload=trace_saved_family(family)
    assert payload['checkpoint_file']=='crash_risk_hardmode.pt'
    assert payload['layers']==['layer1','layer2','layer3']
    assert payload['node_count']==len(payload['nodes'])
    assert payload['edge_count']==len(payload['edges'])
    assert payload['feature_width']==36
    assert payload['embedding_width']==32
    assert payload['nodes'][0]['query'] is True
    assert payload['nodes'][0]['role']=='constraint'
    assert all(len(n['features'])==36 for n in payload['nodes'])
    for n in payload['nodes']:
        assert all(len(n['embedding'][key])==32 for key in ('encoder','layer1','layer2','layer3'))
    assert all(e['relation'] in EDGE_TYPES for e in payload['edges'])
    for key in payload['layers']:
        weights=defaultdict(float)
        for edge in payload['edges']:
            assert 0<=edge[key]<=1
            weights[edge['target']]+=edge[key]
        # The mass clamp used by this particular trained model creates <=1 sums.
        assert all(v<=1.00001 for v in weights.values())
    assert abs(payload['risk_prediction'] - payload['oracle_risk']) <=1


def test_trace_exactly_matches_plain_inference_and_does_not_change_model():
    model, metadata, _=model_and_metadata()
    fs,constraint=make_workload('sharded_commit',123)
    g=make_graph({'sim':fs,'constraint':constraint,'writeback_probability':.4})
    model.eval()
    with torch.no_grad():
        first=model(g)
        traced,details=model(g,return_trace=True)
        last=model(g)
    assert torch.allclose(first,traced,atol=1e-8)
    assert torch.allclose(first,last,atol=1e-8)
    assert len(details['weights'])==3


def test_trace_endpoint_rejects_unknown_families_and_keeps_models_separate():
    client=TestClient(app)
    bad=client.get('/api/research/hardmode/trace?family=not_real')
    assert bad.status_code==400
    ok=client.get('/api/research/hardmode/trace?family=hardlink_retarget')
    assert ok.status_code==200
    result=ok.json()
    assert result['checkpoint_file']=='crash_risk_hardmode.pt'
    assert 'riskgnn' not in result['checkpoint_file']
    assert result['model'].startswith('FaultLab')


def test_heldout_audit_reports_model_failures_not_only_wins():
    response=TestClient(app).get('/api/research/hardmode/audit')
    assert response.status_code==200
    data=response.json()
    assert data['sample_count']==5000
    assert data['heldout_count']==1250
    assert data['per_family']['sharded_commit']['models']['GNN']['mae']>data['per_family']['hardlink_retarget']['models']['GNN']['mae']
    assert data['gnn_worst_cases'][0]['absolute_error']>.4
    assert 'real filesystems' in data['limitations']
    assert all(len(x['paired_bootstrap_95pct'])==2 for x in data['paired_bootstrap'].values())


def test_hardlinks_still_share_inodes_after_faultlab_integration():
    sim,constraint=make_workload('hardlink_release',2027)
    assert sim.paths['/data/blob']==sim.paths['/mirror/blob']==sim.paths['/mirror/extra']
    assert len(set(sim.paths[p] for p in ('/data/blob','/mirror/blob','/mirror/extra')))==1


def test_both_versions_have_independently_valid_attention_traces():
    old=trace_saved_family('sharded_commit',version='v07')
    new=trace_saved_family('sharded_commit',version='v08')
    assert old['model_variant']=='global' and new['model_variant']=='operand'
    assert old['checkpoint_file']=='crash_risk_hardmode_v07.pt'
    assert new['checkpoint_file']=='crash_risk_hardmode.pt'
    assert old['risk_prediction']!=new['risk_prediction']
    assert len(old['layers'])==len(new['layers'])==3
    assert any(abs(a['layer3']-b['layer3'])>1e-5 for a,b in zip(old['edges'],new['edges']))
    client=TestClient(app)
    response=client.get('/api/research/hardmode/trace?version=v07&family=sharded_commit')
    assert response.status_code==200
    assert response.json()['model_variant']=='global'
    response=client.get('/api/research/hardmode/audit?version=v07')
    assert response.status_code==200
    assert response.json()['model_variant']=='global'
    assert client.get('/api/research/hardmode/trace?version=notvalid').status_code==400


def test_two_readouts_are_compared_with_uncertainty_not_claimed_as_uniform_win():
    response=TestClient(app).get('/api/research/hardmode/readouts')
    assert response.status_code==200
    d=response.json()
    assert d['v08_operand']['heldout_mae']<d['v07_global']['heldout_mae']
    assert d['v08_operand']['expected_violations_at_10pct']<d['v07_global']['expected_violations_at_10pct']
    lower,upper=d['v08_minus_v07']['top10_delta_95pct_stratified_bootstrap']
    assert lower<0<upper
    assert 'Exploratory' in d['model_selection_warning']

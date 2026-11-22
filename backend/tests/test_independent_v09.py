"""External template holdout, provenance, and reproducibility regression tests."""
import math
import random
from pathlib import Path

import pytest
import torch

from challenging.generator import ALL_FAMILIES,TRAIN_FAMILIES
from challenging.independent import EXTERNAL_FAMILIES,POLICIES,external_cases,iter_external_cases,make_external_workload
from challenging.graphs import make_graph
from challenging.simulator import check_constraint
from challenging.multiseed import fixed_manifest,paired_seed_summary


@pytest.mark.parametrize('family',EXTERNAL_FAMILIES)
def test_new_family_has_valid_live_state_and_exact_label(family):
    sim, spec = make_external_workload(family,20261116)
    assert check_constraint(sim.paths,sim.values,spec)
    assert any(op['kind']=='durable_checkpoint' for op in sim.operations)
    assert len(sim.pending()[0])+len(sim.pending()[1]) <=12
    for p in POLICIES:
        probability,failures=sim.exact_failure_probability(spec,p)
        assert 0 <= probability <= 1
        for failure in failures:
            assert not check_constraint(failure['recovered_paths'],failure['recovered_versions'],spec)


def test_external_families_not_in_old_training_or_old_holdout():
    assert set(EXTERNAL_FAMILIES).isdisjoint(ALL_FAMILIES)
    assert set(EXTERNAL_FAMILIES).isdisjoint(TRAIN_FAMILIES)


def test_deterministic_frozen_eval_cohort_and_checksum():
    a=external_cases(per_family=3,seed=20261116)
    b=external_cases(per_family=3,seed=20261116)
    path=Path(__file__).resolve().parents[1]/'challenging'/'independent.py'
    assert fixed_manifest(a,path)==fixed_manifest(b,path)
    assert len(a)==18
    assert len({c['id'] for c in a})==18
    assert len({c['seed'] for c in a})==18


def test_no_recovered_outcome_or_label_leaks_to_external_graph():
    cases=external_cases(per_family=1)
    for case in cases:
        first=make_graph(case)
        mutated={**case,'risk_probability':-999.0,'sampled_outcome':1,'persisted_mask':['all']}
        second=make_graph(mutated)
        assert torch.equal(first.x,second.x)
        assert torch.equal(first.edge_type,second.edge_type)
        assert torch.equal(first.edge_index,second.edge_index)
        assert len(first.names)==len(first.roles)==first.x.shape[0]


def test_simulated_crash_sampling_agrees_with_oracle():
    sim,spec=make_external_workload('mirrored_alias_rotation',182)
    p=.55
    expected,_=sim.exact_failure_probability(spec,p)
    rng=random.Random(2026)
    observed=sum(sim.sample_recovery(spec,p,rng)['violation'] for _ in range(4000))/4000
    assert abs(expected-observed)<.035


def test_external_cohort_has_intermediate_risks_and_real_hardlinks():
    cases=external_cases(per_family=24)
    risk=[c['risk_probability'] for c in cases]
    assert sum(0<r<1 for r in risk)>int(.55*len(risk))
    hardlink=[c for c in cases if c['family']=='mirrored_alias_rotation']
    assert all(any(len([p for p,ino in c['sim'].paths.items() if ino==id_])>1
                   for id_ in set(c['sim'].paths.values())) for c in hardlink)


def test_paired_seed_comparison_signs():
    def score(v,top):
        return {'mae_to_oracle':v,'spearman_risk_rank':.8,
                'prioritization':{'10pct':{'expected_violations':top}}}
    runs=[{'models':{'GNN v0.8 operand-aware':score(.1,20), 'Random Forest':score(.2,18)}},
          {'models':{'GNN v0.8 operand-aware':score(.12,19),'Random Forest':score(.25,17)}}]
    result=paired_seed_summary(runs)
    assert result['paired_vs_gnn']['Random Forest']['mae_advantage_of_gnn']['mean']>.1
    assert result['paired_vs_gnn']['Random Forest']['ranking_advantage_of_gnn']['mean']==2


def test_completed_external_evidence_endpoint_and_seed_provenance():
    """Allow a source checkout before running the optional long experiment."""
    from fastapi.testclient import TestClient
    from main import app
    import json
    path=Path(__file__).resolve().parents[1]/'artifacts'/'independent_v09'/'external_multiseed_results.json'
    reply=TestClient(app).get('/api/research/independent-v09')
    if not path.is_file():
        assert reply.status_code==503
        return
    assert reply.status_code==200
    result=reply.json()
    seeds=result['registered_config']['training_seeds']
    assert len(seeds)==len(set(seeds))>=2
    assert result['external_cohort']['summary']['count']>0
    assert set(result['external_cohort']['workload_families'])==set(EXTERNAL_FAMILIES)
    assert len(result['runs'])==len(seeds)
    assert all(run['training_seed'] in seeds for run in result['runs'])
    assert all('GNN v0.8 operand-aware' in run['models'] for run in result['runs'])


def test_saved_independent_model_predictions_recompute_when_available():
    """No faked model outputs in the experiment artifact."""
    import json
    folder=Path(__file__).resolve().parents[1]/'artifacts'/'independent_v09'
    checkpoint=folder/'seed_11'/'crash_risk_hardmode.pt'
    predictions=folder/'seed_11'/'external_predictions.json'
    if not checkpoint.is_file() or not predictions.is_file():
        pytest.skip('Full repeated-seed training artifacts unavailable')
    from challenging.model import CrashRiskGNN
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CrashRiskGNN(variant='operand').eval()
    model.load_state_dict(raw['state_dict'])
    cohort=external_cases(per_family=240)
    saved=json.loads(predictions.read_text())
    assert [c['id'] for c in cohort]==saved['case_ids']
    for ix in (0,80,480,1439):
        g=make_graph(cohort[ix])
        with torch.no_grad(): actual=float(torch.sigmoid(model(g)).item())
        assert abs(actual-saved['predictions']['GNN v0.8 operand-aware'][ix])<1e-6


@pytest.mark.parametrize('seed',[11,29,47,83,131])
def test_seed_specific_trace_uses_exact_selected_checkpoint(seed):
    from challenging.trace import trace_saved_family,model_and_metadata
    from challenging.independent import EXTERNAL_FAMILIES
    from challenging.graphs import make_graph
    data=trace_saved_family('package_upgrade',version=f'v09_s{seed}')
    model,meta,path=model_and_metadata(f'v09_s{seed}')
    assert meta['seed']==seed
    assert path.parent.name==f'seed_{seed}'
    assert data['checkpoint_file'].startswith('independent_v09/seed_'+str(seed)+'/')
    assert len(data['layers'])==3
    assert data['family'] in EXTERNAL_FAMILIES
    assert data['feature_width']==36
    assert len(data['nodes'])==data['node_count']
    assert len(data['edges'])==data['edge_count']
    assert all('layer3' in edge for edge in data['edges'])
    assert abs(data['risk_prediction']-data['oracle_risk']) <= 1


def test_v09_trace_rejects_unknown_seed_or_old_external_family():
    from challenging.trace import trace_saved_family
    with pytest.raises(ValueError):
        trace_saved_family('package_upgrade','v09_s9999')
    with pytest.raises(ValueError):
        trace_saved_family('package_upgrade','v08')


def test_research_views_integrated_into_actual_tanstack_frontend():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    explorer = (root / 'src/components/vfsim/ResearchGraph.tsx').read_text(encoding='utf8')
    model_page = (root / 'src/routes/model.tsx').read_text(encoding='utf8')
    risk_page = (root / 'src/routes/risk.tsx').read_text(encoding='utf8')
    for feature in ('filesystemTrace', 'faultlabTrace', 'Incoming effective attention',
                    'Minimum attention', 'Raw input features', 'Full embedding vector'):
        assert feature in explorer
    assert 'MultiseedPanel' in model_page
    assert 'MultiseedPanel' in risk_page
    from fastapi.testclient import TestClient
    from main import app
    assert TestClient(app).get('/api/research/independent-v09').status_code == 200



def test_streaming_one_external_family_preserves_candidate_order_and_oracles():
    full = external_cases(per_family=3)
    fields = ("id", "seed", "writeback_probability", "risk_probability")
    for family in EXTERNAL_FAMILIES:
        streamed = iter_external_cases(per_family=3, selected_family=family)
        assert [tuple(case[key] for key in fields) for case in streamed] == [
            tuple(case[key] for key in fields) for case in full if case["family"] == family
        ]

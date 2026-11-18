"""Compare two already-trained GNN readouts on the SAME held-out cases.

Exploratory post-hoc architecture comparison: both checkpoints were assessed
on seed-42 test cases, so model-choice claims require independent confirmation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from challenging.generator import build_cases, split_cases, HELDOUT_FAMILIES
from challenging.graphs import make_graph
from challenging.model import CrashRiskGNN


def rank_expected(truth,pred,k):
    return float(truth[np.argsort(-pred,kind='stable')[:k]].sum())


def compare(seed=42,samples=5000,repeats=600):
    torch.set_num_threads(2)
    art=Path(__file__).resolve().parents[1]/'artifacts'
    cases=split_cases(build_cases(samples,seed=seed),seed=seed)[3]
    graphs=[make_graph(c) for c in cases]
    truth=np.array([c['risk_probability'] for c in cases])
    predictions={}
    for name,filename in [('v07_global','crash_risk_hardmode_v07.pt'),('v08_operand','crash_risk_hardmode.pt')]:
        ckpt=torch.load(art/filename,weights_only=True,map_location='cpu')
        model=CrashRiskGNN(variant=ckpt.get('metadata',{}).get('model_variant','global')).eval()
        model.load_state_dict(ckpt['state_dict'])
        with torch.no_grad():predictions[name]=np.array([torch.sigmoid(model(g)).item() for g in graphs])
    old,new=predictions.values();k=round(len(truth)*.1)
    diff=np.abs(truth-old)-np.abs(truth-new)
    rng=np.random.default_rng(2027)
    family_idx={name:np.flatnonzero([c['family']==name for c in cases]) for name in HELDOUT_FAMILIES}
    bootstrap_mae=[];bootstrap_ranking=[]
    for _ in range(repeats):
        idx=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in family_idx.values()])
        yy,aa,bb=truth[idx],old[idx],new[idx]
        bootstrap_mae.append(float((np.abs(yy-aa)-np.abs(yy-bb)).mean()))
        bootstrap_ranking.append(rank_expected(yy,bb,k)-rank_expected(yy,aa,k))
    report={
        'model_selection_warning':'Exploratory v0.8 comparison was examined on the same held-out seed-42 cases as v0.7; do not claim an unbiased architecture-search win.',
        'samples':samples,'heldout_count':len(truth),'seeds':{'dataset':seed,'bootstrap':2027},
        'v07_global':{'heldout_mae':round(float(np.abs(truth-old).mean()),5),
                      'expected_violations_at_10pct':round(rank_expected(truth,old,k),2)},
        'v08_operand':{'heldout_mae':round(float(np.abs(truth-new).mean()),5),
                      'expected_violations_at_10pct':round(rank_expected(truth,new,k),2)},
        'v08_minus_v07':{
            'mae_reduction':round(float(diff.mean()),5),
            'mae_reduction_95pct_stratified_bootstrap':[round(float(q),5) for q in np.quantile(bootstrap_mae,[.025,.975])],
            'expected_violations_top10_delta':round(rank_expected(truth,new,k)-rank_expected(truth,old,k),3),
            'top10_delta_95pct_stratified_bootstrap':[round(float(q),3) for q in np.quantile(bootstrap_ranking,[.025,.975])]},
        'per_family':{name:{'count':len(indices),
          'v07_mae':round(float(np.abs(truth[indices]-old[indices]).mean()),5),
          'v08_mae':round(float(np.abs(truth[indices]-new[indices]).mean()),5)} for name,indices in family_idx.items()},
        'limitations':'All tests use an application-level synthetic oracle and one dataset seed. Repeated model training seeds, unseen generators and real filesystem studies are not covered.'}
    (art/'readout_comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':compare()

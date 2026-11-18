"""Reproduce and audit the v0.7 benchmark, without selecting on held-out families.

Produces per-family errors, sampled ranking outcomes and paired workload
bootstrap intervals.  Intervals characterize this one synthetic test cohort;
they do NOT cover alternative generators, seeds, platforms, or real ext4.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from challenging.generator import build_cases,split_cases,HELDOUT_FAMILIES
from challenging.graphs import make_graph,graph_vector
from challenging.model import CrashRiskGNN


def rank_expected(y,p,budget=.1):
    n=max(1,round(len(y)*budget));return float(np.asarray(y)[np.argsort(-np.asarray(p),kind='stable')[:n]].sum())


def bootstrap_deltas(y,base,other,seed=4417,repetitions=500):
    """Confidence interval for (baseline MAE - GNN MAE), fixed candidate cohort."""
    rng=np.random.default_rng(seed);N=len(y);d=np.abs(np.asarray(y)-np.asarray(other))-np.abs(np.asarray(y)-np.asarray(base))
    vals=[]
    for _ in range(repetitions):
        idx=rng.integers(0,N,size=N);vals.append(float(np.mean(d[idx])))
    return {'improvement_mae':round(float(d.mean()),5),
            'paired_bootstrap_95pct':[round(float(z),5) for z in np.quantile(vals,[.025,.975])],
            'improvement_sign': 'positive means GNN lower MAE'}


def audit(samples=5000,seed=42,checkpoint=None,output=None):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    cases=build_cases(samples,seed=seed)
    train,val,id_cases,ood=split_cases(cases,seed=seed)
    ckpt=Path(checkpoint or Path(__file__).resolve().parents[1]/'artifacts/crash_risk_hardmode.pt')
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    if state['metadata'].get('samples')!=samples or state['metadata'].get('seed')!=seed:
        raise ValueError('Dataset seed/count must match checkpoint training provenance')
    model=CrashRiskGNN(variant=state.get('metadata', {}).get('model_variant', 'global'));model.load_state_dict(state['state_dict']);model.eval()
    graphs={id(c):make_graph(c) for c in train+ood}
    print('Evaluating trained GNN on held-out families...',flush=True)
    with torch.no_grad():
        gnn=np.array([float(torch.sigmoid(model(graphs[id(c)])).item()) for c in ood])
    truth=np.array([c['risk_probability'] for c in ood])
    print('Preparing baseline graph vectors...',flush=True)
    train_m=np.stack([graph_vector(graphs[id(c)]) for c in train]);test_m=np.stack([graph_vector(graphs[id(c)]) for c in ood]);
    sel=VarianceThreshold(threshold=1e-8);X_train=sel.fit_transform(train_m);X_test=sel.transform(test_m)
    ytrain=np.array([c['risk_probability'] for c in train])
    estimators={
        'Random Forest':RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=3,random_state=seed,n_jobs=2),
        'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=.07,random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        estimators['XGBoost']=XGBRegressor(n_estimators=130,max_depth=5,learning_rate=.07,subsample=.9,colsample_bytree=.85,tree_method='hist',n_jobs=2,random_state=seed)
    except ImportError: pass
    preds={'GNN':gnn}
    for name,estimator in estimators.items():
        print('Fitting',name,flush=True);estimator.fit(X_train,ytrain)
        preds[name]=np.clip(estimator.predict(X_test),0,1)
    roles={name:{'mae':round(float(np.abs(p-truth).mean()),5),
                 'expected_violations_top10pct':round(rank_expected(truth,p),2)} for name,p in preds.items()}
    by_family={}
    for family in HELDOUT_FAMILIES:
        indexes=np.array([i for i,c in enumerate(ood) if c['family']==family],dtype=int)
        by_family[family]={'count':len(indexes), 'mean_risk':round(float(truth[indexes].mean()),5),
           'models':{name:{'mae':round(float(np.abs(truth[indexes]-p[indexes]).mean()),5),
                    'expected_violations_top10pct':round(rank_expected(truth[indexes],p[indexes]),2)} for name,p in preds.items()}}
    worst=np.argsort(-np.abs(gnn-truth),kind='stable')[:12]
    worst_cases=[{'id':ood[i]['id'],'family':ood[i]['family'],'seed':ood[i]['seed'],
                  'writeback_probability':ood[i]['writeback_probability'],
                  'oracle_risk':round(float(truth[i]),5),
                  'gnn_prediction':round(float(gnn[i]),5),
                  'absolute_error':round(float(abs(gnn[i]-truth[i])),5)} for i in worst]
    intervals={name:bootstrap_deltas(truth,gnn,p) for name,p in preds.items() if name!='GNN'}
    # Per-bin ECE for understanding calibration; it is a model diagnostic, not a post-hoc calibration fit.
    bins=[]
    for left in np.linspace(0,1,6)[:-1]:
        right=left+.2;in_bin=(gnn>=left)&(gnn<right if right<1 else gnn<=1)
        if np.any(in_bin):bins.append({'predicted_range':[round(float(left),2),round(float(right),2)],
            'count':int(in_bin.sum()),'mean_prediction':round(float(gnn[in_bin].mean()),4),
            'mean_oracle_probability':round(float(truth[in_bin].mean()),4),
            'absolute_calibration_gap':round(float(abs(gnn[in_bin].mean()-truth[in_bin].mean())),4)})
    report={'provenance':'Same checkpoint and dataset as v0.7, no refitting of the GNN on OOD',
        'sample_count':samples,'dataset_seed':seed,'model_variant':model.variant,'train_count':len(train), 'heldout_count':len(ood),
        'heldout_families':list(HELDOUT_FAMILIES), 'model_comparison':roles,
        'per_family':by_family,'gnn_worst_cases':worst_cases,'calibration_bins':bins,
        'paired_bootstrap':intervals,
        'limitations':('Bootstraps resample examples from the same generator and single checkpoint. '
           'Not independent training-seed confidence bounds and not evidence on real filesystems. '
           'The held-out test set was inspected for diagnostics, not used to retrain the reported checkpoint.')}
    outfile=Path(output or Path(__file__).resolve().parents[1]/'artifacts/hardmode_audit.json')
    outfile.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Saved',outfile,flush=True)
    print('Model comparison:',json.dumps(roles),flush=True)
    print('Paired intervals:',json.dumps(intervals),flush=True)
    print('Worst example:',worst_cases[0],flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=5000);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output',default=None)
    a=ap.parse_args();audit(a.samples,a.seed,output=a.output)

"""Pre-registered multiple-seed experiment against a fixed *new* synthetic cohort.

- The six external workload families are NOT included in model training,
  validation, early-stopping, or any baseline fitting.
- A fixed external cohort is shared across all seeds/models for paired tests.
- GNN checkpoints and per-seed artifacts live in a separate directory; v0.8
  production checkpoints and visualizations are not modified.
- MAE and top-budget scores are about the toy writeback model, NOT ext4.

Run from backend/: python -m challenging.multiseed --seeds 11 29 47 83 131
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import math

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from challenging.generator import build_cases,split_cases
from challenging.independent import EXTERNAL_FAMILIES, EXTERNAL_SEED, external_cases,cohort_summary
from challenging.graphs import graph_vector,make_graph
from challenging.model import CrashRiskGNN
from challenging.train import metrics,train

DEFAULT_SEEDS=(11,29,47,83,131)


def mean_and_sample_sd(numbers):
    a=np.asarray(numbers,dtype=np.float64)
    return {'n':int(len(a)),'mean':round(float(a.mean()),6),
            'std_across_seeds':round(float(a.std(ddof=1)),6) if len(a)>1 else None,
            'min':round(float(a.min()),6),'max':round(float(a.max()),6),
            'values':[round(float(v),6) for v in a]}


def score_by_family(cases,predictions):
    groups={}
    y=np.array([c['risk_probability'] for c in cases])
    p=np.clip(np.asarray(predictions),0,1)
    for family in EXTERNAL_FAMILIES:
        ids=np.array([i for i,c in enumerate(cases) if c['family']==family])
        rank=np.argsort(-p[ids],kind='stable')
        n=max(1,round(len(ids)*.10))
        groups[family]={'count':len(ids),'mae':round(float(abs(y[ids]-p[ids]).mean()),6),
                        'mean_oracle_risk':round(float(y[ids].mean()),6),
                        'expected_failures_top10pct':round(float(y[ids][rank][:n].sum()),6)}
    return groups


def select_training_data(dataset_seed, n):
    cases=build_cases(n,seed=dataset_seed)
    tr,_,_,_=split_cases(cases,seed=dataset_seed)
    return tr


def fit_baselines(train_cases, X_external, seed):
    train_vectors=np.stack([graph_vector(make_graph(c)) for c in train_cases])
    y_train=np.array([c['risk_probability'] for c in train_cases],dtype=np.float64)
    selector=VarianceThreshold(threshold=1e-8)
    X_train=selector.fit_transform(train_vectors)
    X_ext=selector.transform(X_external)
    models={
        'Random Forest':RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=3,random_state=seed,n_jobs=2),
        'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=.07,random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost']=XGBRegressor(n_estimators=130,max_depth=5,learning_rate=.07,subsample=.9,
                                     colsample_bytree=.85,tree_method='hist',n_jobs=2,random_state=seed)
    except ImportError:
        pass
    predictions={}
    for name, model in models.items():
        model.fit(X_train,y_train)
        predictions[name]=np.clip(model.predict(X_ext),0,1)
    scaler=StandardScaler()
    ridge=Ridge(alpha=100.0)
    ridge.fit(scaler.fit_transform(X_train),y_train)
    predictions['Ridge (graph statistics)']=np.clip(ridge.predict(scaler.transform(X_ext)),0,1)
    return predictions


def fixed_manifest(cases, source_path):
    rows=[(c['id'],c['family'],c['seed'],c['writeback_probability'],c['risk_probability']) for c in cases]
    payload=json.dumps(rows,separators=(',',':'),ensure_ascii=True)
    return {'cohort_sha256':hashlib.sha256(payload.encode()).hexdigest(),
            'generator_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'cohort_seed':EXTERNAL_SEED,'workload_families':list(EXTERNAL_FAMILIES),
            'cases_per_family':len(cases)//len(EXTERNAL_FAMILIES),
            'summary':cohort_summary(cases)}


def paired_seed_summary(runs):
    names=list(runs[0]['models'])
    primary='mae_to_oracle'
    ranking=lambda d:d['prioritization']['10pct']['expected_violations']
    output={'models':{},'paired_vs_gnn':{}}
    for name in names:
        output['models'][name]={
            'mae_to_oracle':mean_and_sample_sd([r['models'][name][primary] for r in runs]),
            'rank_spearman':mean_and_sample_sd([r['models'][name]['spearman_risk_rank'] for r in runs]),
            'expected_failures_top10pct':mean_and_sample_sd([ranking(r['models'][name]) for r in runs]),
        }
        if name!='GNN v0.8 operand-aware':
            # Positive MAE delta => GNN lower error; positive ranking delta => GNN
            # discovers more expected failures in the same 10% testing budget.
            output['paired_vs_gnn'][name]={
                'mae_advantage_of_gnn':mean_and_sample_sd([r['models'][name][primary]-r['models']['GNN v0.8 operand-aware'][primary] for r in runs]),
                'ranking_advantage_of_gnn':mean_and_sample_sd([ranking(r['models']['GNN v0.8 operand-aware'])-ranking(r['models'][name]) for r in runs]),
            }
    return output


def experiment(*,seeds=DEFAULT_SEEDS,samples=5000,epochs=5,per_family=240,outdir=None):
    """Train on in-family cases only; evaluate ONCE on a frozen new-family cohort.

    Do not change the architecture, seed list, validation strategy, or tested
    cohort after reviewing the external results and then claim a pristine holdout.
    """
    seeds=tuple(int(s) for s in seeds)
    if len(set(seeds))!=len(seeds):raise ValueError('Repeated seeds are not independent runs')
    if len(seeds)<2:raise ValueError('Use at least two training seeds')
    if samples<160:raise ValueError('Need enough data for train/validation split')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    outdir=Path(outdir) if outdir else Path(__file__).resolve().parents[1]/'artifacts'/'independent_v09'
    outdir.mkdir(parents=True,exist_ok=True)
    frozen=external_cases(per_family=per_family)
    manifest=fixed_manifest(frozen,Path(__file__).with_name('independent.py'))
    (outdir/'external_cohort_manifest.json').write_text(json.dumps(manifest,indent=2))
    graphs=[make_graph(c) for c in frozen]
    X_ext=np.stack([graph_vector(g) for g in graphs])
    runs=[]
    for seed in seeds:
        print(f'== Training seed {seed} ==',flush=True)
        directory=outdir/f'seed_{seed}'
        directory.mkdir(parents=True,exist_ok=True)
        train(samples=samples,epochs=epochs,seed=seed,outdir=directory,baselines=False,variant='operand')
        raw=torch.load(directory/'crash_risk_hardmode.pt',map_location='cpu',weights_only=True)
        model=CrashRiskGNN(variant='operand');model.load_state_dict(raw['state_dict']);model.eval()
        with torch.no_grad():
            scores=np.array([float(torch.sigmoid(model(g)).item()) for g in graphs])
        baseline_train=select_training_data(seed,samples)
        alternatives=fit_baselines(baseline_train,X_ext,seed)
        # The heuristic has NO access to the oracle label or actual persistence
        # mask; it is a deliberately weak but completely specified comparator.
        alternatives['Pending-writeback heuristic']=np.array([
            1-(1-c['writeback_probability'])**sum(map(len,c['sim'].pending())) for c in frozen])
        predictions={'GNN v0.8 operand-aware':scores,**alternatives}
        rows={name:metrics(frozen,pred) for name,pred in predictions.items()}
        fam={name:score_by_family(frozen,pred) for name,pred in predictions.items()}
        record={'training_seed':seed,'training_samples':samples,'epochs':epochs,
                'selected_epoch':min(raw['metadata']['history'],key=lambda row:row['validation_mae'])['epoch'],
                'best_validation_mae':min(x['validation_mae'] for x in raw['metadata']['history']),
                'models':rows,'by_family':fam}
        runs.append(record)
        (directory/'external_predictions.json').write_text(json.dumps({
            'model_names':list(predictions),'case_ids':[c['id'] for c in frozen],
            'predictions':{k:[round(float(v),7) for v in pred] for k,pred in predictions.items()}},indent=2))
        print(f"Finished seed={seed}: GNN MAE={rows['GNN v0.8 operand-aware']['mae_to_oracle']}, "
              f"HistGB MAE={rows['HistGradientBoosting']['mae_to_oracle']}, "
              f"GNN top10={rows['GNN v0.8 operand-aware']['prioritization']['10pct']['expected_violations']}",flush=True)
        (outdir/'progress.json').write_text(json.dumps({'completed_seeds':[r['training_seed'] for r in runs],
             'target_seeds':list(seeds),'incomplete':True},indent=2))
    result={'study':'VFSim v0.9 external synthetic workload families + independent training seeds',
            'status':'completed','trained_model':'FaultLab operand-aware v0.8 architecture (unchanged)',
            'registered_config':{'training_seeds':list(seeds),'samples_per_seed':samples,
                'max_epochs':epochs,'checkpoint_selection':'lowest in-family validation MAE',
                'external_per_family':per_family,'external_dataset_seed':EXTERNAL_SEED},
            'external_cohort':manifest,'runs':runs,'summary':paired_seed_summary(runs),
            'limitations':[
                'Every example uses the same in-memory, independent per-object writeback simulator; no ext4 execution.',
                'Families are independently designed templates, not independent production workload traces.',
                'External cohort is shared across seeds for paired comparisons; seeds vary training generation and initialization.',
                'Five seeds are not sufficient alone to infer universal performance on real operating systems.',
                'Original v0.8 architecture was developed after looking at older held-out families.',
                'Attention coefficients are graph message weights, not causal attributions.',
            ],'created_at':datetime.now(timezone.utc).isoformat()}
    (outdir/'external_multiseed_results.json').write_text(json.dumps(result,indent=2))
    (outdir/'progress.json').write_text(json.dumps({'completed_seeds':list(seeds),'incomplete':False},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=list(DEFAULT_SEEDS))
    parser.add_argument('--samples',type=int,default=5000)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--per-family',type=int,default=240)
    parser.add_argument('--outdir')
    args=parser.parse_args()
    experiment(seeds=args.seeds,samples=args.samples,epochs=args.epochs,per_family=args.per_family,outdir=args.outdir)

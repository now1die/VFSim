"""Reproducible hard-mode pre-crash ranking benchmark.

Example (from backend/): python -m challenging.train --samples 5000 --epochs 5
No real Linux testing. All probabilities are exact under the documented toy
per-object writeback model; scoring does not see the sampled writeback mask.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score
from challenging.generator import build_cases,split_cases,summarize,ALL_FAMILIES,HELDOUT_FAMILIES
from challenging.graphs import make_graph,graph_vector
from challenging.model import CrashRiskGNN


def metrics(cases,predictions):
    y=np.array([s['risk_probability'] for s in cases],dtype=np.float64)
    p=np.clip(np.asarray(predictions,dtype=np.float64),0,1)
    # One *real simulated recovery* per candidate, not an analytically
    # fabricated label. Seeds are stable per workload across model comparisons.
    observed=np.asarray([case['sim'].sample_recovery(case['constraint'],case['writeback_probability'],
        random.Random(0x1D1E + case['seed']))['violation'] for case in cases],dtype=int)
    rank=spearmanr(y,p).statistic if len(np.unique(y))>1 and len(np.unique(p))>1 else 0.0
    result={'count':len(y),'mean_oracle_risk':round(float(y.mean()),5),
            'mae_to_oracle':round(float(np.abs(y-p).mean()),5),
            'rmse_to_oracle':round(float(np.mean((y-p)**2)**.5),5),
            'brier_to_sampled_outcome':round(float(np.mean((observed-p)**2)),5),
            'spearman_risk_rank':round(float(rank),5),
            'realized_violations':int(observed.sum())}
    if len(set(observed))>1:
        result['sampled_roc_auc']=round(float(roc_auc_score(observed,p)),5)
        result['sampled_average_precision']=round(float(average_precision_score(observed,p)),5)
    order=np.argsort(-p,kind='stable')
    top={}
    for percent in (5,10,20,50):
        k=max(1,round(len(y)*percent/100))
        top[str(percent)+'pct']={'tests':k,
              'expected_violations':round(float(y[order][:k].sum()),2),
              'observed_violations':int(observed[order][:k].sum()),
              'oracle_optimal_expected':round(float(np.sort(y)[::-1][:k].sum()),2)}
    result['prioritization']=top
    return result


def evaluate(model,cases,graphs):
    model.eval()
    with torch.no_grad():
        return np.array([float(torch.sigmoid(model(graphs[id(case)]))) for case in cases])


def train(samples=5000,epochs=5,seed=42,outdir=None,baselines=True,variant="global"):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    cases=build_cases(samples,seed=seed)
    train_cases,val_cases,id_cases,ood_cases=split_cases(cases,seed=seed)
    print('Oracle summary',summarize(cases),'splits',list(map(len,(train_cases,val_cases,id_cases,ood_cases))),flush=True)
    graphs={id(case):make_graph(case) for case in cases}
    model=CrashRiskGNN(variant=variant)
    optim=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.0005)
    loss_fn=torch.nn.MSELoss()
    best=float('inf');state=None;history=[]
    for epoch in range(1,epochs+1):
        model.train();random.shuffle(train_cases)
        losses=[]
        for case in train_cases:
            optim.zero_grad(set_to_none=True)
            prediction=torch.sigmoid(model(graphs[id(case)]))
            label=torch.tensor([case['risk_probability']],dtype=torch.float32)
            loss=loss_fn(prediction,label)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optim.step();losses.append(float(loss.item()))
        vp=evaluate(model,val_cases,graphs)
        val_m=metrics(val_cases,vp)
        history.append({'epoch':epoch,'train_mse':round(float(np.mean(losses)),5),
                        'validation_mae':val_m['mae_to_oracle'],
                        'validation_spearman':val_m['spearman_risk_rank']})
        print(f'Epoch {epoch} train MSE {np.mean(losses):.4f} validation MAE {val_m["mae_to_oracle"]:.4f} ranking rho {val_m["spearman_risk_rank"]:.3f}',flush=True)
        if val_m['mae_to_oracle']<best:
            best=val_m['mae_to_oracle'];state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    scores={'GNN':{'in_distribution':metrics(id_cases,evaluate(model,id_cases,graphs)),
                   'heldout_families':metrics(ood_cases,evaluate(model,ood_cases,graphs))}}
    if baselines:
        groups=(train_cases,val_cases,id_cases,ood_cases)
        vector={id(c):graph_vector(graphs[id(c)]) for c in cases}
        matrices=[np.stack([vector[id(c)] for c in group]) for group in groups]
        targets=[np.array([c['risk_probability'] for c in group]) for group in groups]
        # Identify constant and almost-constant statistics using training data only.
        selector=VarianceThreshold(threshold=1e-8)
        X_train=selector.fit_transform(matrices[0]);X_id=selector.transform(matrices[2]);X_ood=selector.transform(matrices[3])
        print('Baseline input dimensions:',X_train.shape[1],flush=True)
        estimators={
            'Random Forest':RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=3,random_state=seed,n_jobs=2),
            'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=.07,random_state=seed),
        }
        try:
            from xgboost import XGBRegressor
            estimators['XGBoost']=XGBRegressor(n_estimators=130,max_depth=5,learning_rate=.07,
                        subsample=.9,colsample_bytree=.85,tree_method='hist',n_jobs=2,random_state=seed)
        except ImportError: pass
        for name,estimator in estimators.items():
            estimator.fit(X_train,targets[0]);scores[name]={
                'in_distribution':metrics(id_cases,estimator.predict(X_id)),
                'heldout_families':metrics(ood_cases,estimator.predict(X_ood))}
            print(name,'OOD MAE',scores[name]['heldout_families']['mae_to_oracle'],flush=True)
        norm=StandardScaler()
        rr=Ridge(alpha=100.0)
        rr.fit(norm.fit_transform(X_train),targets[0])
        scores['Ridge (graph statistics)']={
            'in_distribution':metrics(id_cases,rr.predict(norm.transform(X_id))),
            'heldout_families':metrics(ood_cases,rr.predict(norm.transform(X_ood)))}
        # A cheap, intentionally simple baseline that needs no labels or training.
        for name,groupset in (('in_distribution',id_cases),('heldout_families',ood_cases)):
            heur=[]
            for case in groupset:
                fs,ds=case['sim'].pending()
                heur.append(1-(1-case['writeback_probability'])**len(fs+ds))
            scores.setdefault('Pending-writeback heuristic',{})[name]=metrics(groupset,heur)
    outdir=Path(outdir) if outdir else Path(__file__).resolve().parents[1]/'artifacts'
    outdir.mkdir(parents=True,exist_ok=True)
    ckpt=outdir/'crash_risk_hardmode.pt'
    metadata={'name':'VFSim Hard-Mode FaultLab','samples':samples,'seed':seed,'epochs':epochs,'model_variant':variant,
              'split':dict(zip(('train','validation','in_distribution_test','heldout_family_test'),map(len,(train_cases,val_cases,id_cases,ood_cases)))),
              'families':list(ALL_FAMILIES),'heldout_families':list(HELDOUT_FAMILIES),
              'oracle_summary':summarize(cases),'method':'Exhaustive application-invariant recovery oracle over independent per-object writeback masks',
              'prediction_boundary':'Workload, stable/current objects and writeback probability known; realized persistence mask and recovered state hidden',
              'limitations':'Synthetic independent object/dir writeback; no ext4 VM or Linux block tracing; no measured real filesystem vulnerabilities',
              'history':history,'results':scores,'created_at':datetime.now(timezone.utc).isoformat()}
    torch.save({'state_dict':model.state_dict(),'metadata':metadata},ckpt)
    (outdir/'hardmode_metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    exemplars=[]
    for family in ALL_FAMILIES:
        family_cases=[c for c in cases if c['family']==family]
        eligible=[c for c in family_cases if .07<c['risk_probability']<.95]
        case=eligible[0] if eligible else family_cases[0]
        sim=case['sim']; constraint=case['constraint']
        expected,failure_states=sim.exact_failure_probability(constraint,case['writeback_probability'])
        with torch.no_grad():
            example_prediction=float(torch.sigmoid(model(graphs[id(case)])).item())
        exemplars.append({'id':case['id'],'seed':case['seed'], 'gnn_predicted_risk':round(example_prediction,5),
                          'family':family,'risk_probability':expected,
                          'writeback_probability':case['writeback_probability'],
                          'constraint':constraint.kind,'operand_paths':list(constraint.paths),
                          'pending':dict(zip(('files','directories'),map(list,sim.pending()))),
                          'operations':sim.operations,'failure_example':failure_states[0] if failure_states else None})
    (outdir/'hardmode_examples.json').write_text(json.dumps(exemplars,indent=2),encoding='utf8')
    print('Saved',ckpt,flush=True)
    for name,dat in scores.items():print(name,'ID',dat['in_distribution']['mae_to_oracle'],'OOD',dat['heldout_families']['mae_to_oracle'], 'OOD expected @10%',dat['heldout_families']['prioritization']['10pct']['expected_violations'],flush=True)
    return metadata

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--samples',type=int,default=5000);a.add_argument('--epochs',type=int,default=5);a.add_argument('--seed',type=int,default=42);a.add_argument('--outdir');a.add_argument('--no-baselines',action='store_true');a.add_argument('--variant',choices=('global','operand'),default='global')
    args=a.parse_args();train(args.samples,args.epochs,args.seed,args.outdir,not args.no_baselines,variant=args.variant)

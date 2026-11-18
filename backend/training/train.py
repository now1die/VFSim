"""Train VFSim RiskGNN on 5,000 oracle-labeled *simulated* crash workloads.

Run from backend: python -m training.train --samples 5000 --epochs 5
Workload-family holdout is strict. This is NOT a Linux crash-consistency claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import numpy as np
import torch
from training.dataset import (TOTAL_SAMPLES, dataset_summary, generate_dataset,
                              split_family_heldout, TRAIN_FAMILIES, OOD_FAMILIES)
from training.baselines import summarize_scores, prioritize, compare_baselines
from gnn.model import RiskGNN


def evaluate(model, samples, loss_fn, *, with_predictions=False):
    if not samples:
        return {} if not with_predictions else ({}, [])
    model.eval()
    probabilities, labels, losses = [], [], []
    with torch.no_grad():
        for sample in samples:
            g = sample['graph']
            prediction = model(g.x_dict, g.edge_index_dict, sample['touched_inodes'])
            y = float(sample['label'])
            losses.append(float(loss_fn(prediction, torch.tensor([y]))))
            probabilities.append(float(torch.sigmoid(prediction).item()))
            labels.append(int(y))
    metrics = summarize_scores(labels, probabilities)
    metrics['loss'] = round(float(np.mean(losses)), 5)
    if with_predictions:
        return metrics, probabilities
    return metrics


def train_model(total_samples=TOTAL_SAMPLES, epochs=5, seed=42, output=None, baseline=True):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    samples = generate_dataset(total_samples=total_samples, seed=seed)
    train, validation, id_test, ood_test = split_family_heldout(samples, seed=seed)
    if not min(map(len, (train, validation, id_test, ood_test))):
        raise ValueError('Insufficient samples to form four nonempty splits')
    model = RiskGNN()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_loss, best_state = float('inf'), None
    print('Dataset:', dataset_summary(samples), flush=True)
    print('Splits:', *map(len, (train, validation, id_test, ood_test)), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        for sample in train:
            optimizer.zero_grad()
            g = sample['graph']
            logit = model(g.x_dict, g.edge_index_dict, sample['touched_inodes'])
            loss = criterion(logit, torch.tensor([float(sample['label'])]))
            loss.backward()
            optimizer.step()
        training_metrics = evaluate(model, train, criterion)
        val_metrics = evaluate(model, validation, criterion)
        history.append({'epoch': epoch, 'train_loss': training_metrics['loss'],
                        'train_accuracy': training_metrics['accuracy'],
                        'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'],
                        'val_f1': val_metrics['f1']})
        print(f'Epoch {epoch:02d} train_F1={training_metrics["f1"]:.3f} '
              f'val_F1={val_metrics["f1"]:.3f} val_loss={val_metrics["loss"]:.3f}', flush=True)
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    validation_metrics = evaluate(model, validation, criterion)
    id_metrics, id_predictions = evaluate(model, id_test, criterion, with_predictions=True)
    ood_metrics, ood_predictions = evaluate(model, ood_test, criterion, with_predictions=True)
    ranking = {
        'in_distribution': prioritize([s['label'] for s in id_test], id_predictions),
        'heldout_families': prioritize([s['label'] for s in ood_test], ood_predictions),
    }
    meta = {
        'name': 'VFSim RiskGNN', 'architecture': '2-layer relation-aware heterogeneous attention',
        'dataset': 'synthetic VFS crash-recovery application-invariant oracle',
        'limitations': 'Simplified global-fsync snapshot semantics; not Linux block-level failures',
        'samples': total_samples, 'summary': dataset_summary(samples),
        'train_families': list(TRAIN_FAMILIES), 'heldout_families': list(OOD_FAMILIES),
        'split': {'train':len(train),'validation':len(validation),
                  'in_distribution_test':len(id_test),'heldout_family_test':len(ood_test)},
        'seed':seed, 'epochs':epochs, 'validation':validation_metrics,
        'in_distribution_test':id_metrics, 'heldout_family_test':ood_metrics,
        'test':ood_metrics, 'prioritization':ranking,
        'history':history,'trained_at':datetime.now(timezone.utc).isoformat(),
    }
    output = Path(output) if output else Path(__file__).resolve().parents[1]/'artifacts'/'riskgnn.pt'
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'metadata':meta}, output)
    (output.parent/'training_metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if baseline:
        results = compare_baselines(train, validation, id_test, ood_test, seed=seed)
        (output.parent/'baseline_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print('Baseline F1:', {k:{part:v[part]['f1'] for part in ('in_distribution','heldout_families')}
                               for k,v in results.items()},flush=True)
    print('Saved',output,'\nID test:',id_metrics,'\nOOD:',ood_metrics,flush=True)
    return model,meta

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=TOTAL_SAMPLES)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output')
    parser.add_argument('--no-baselines',action='store_true')
    args=parser.parse_args()
    train_model(args.samples,args.epochs,args.seed,args.output,not args.no_baselines)

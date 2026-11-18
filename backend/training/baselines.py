"""Fair, reproducible baselines and crash-test *prioritization* metrics.

All features come from the same input graph available to the GNN at prediction
time. The oracle label is never passed to models. This is a simulator benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from graph.builder import NODE_TYPES, RELATIONS
from training.dataset import generate_dataset, split_family_heldout


def graph_vector(sample):
    g = sample['graph']
    vector = []
    for node_type in NODE_TYPES:
        values = g.x_dict[node_type].numpy()
        width = values.shape[1]
        vector.append(float(values.shape[0]))
        if values.shape[0]:
            vector.extend(values.mean(0).tolist())
            vector.extend(values.max(0).tolist())
            vector.extend(values.min(0).tolist())
        else:
            vector.extend([0.0] * (width * 3))
    vector.extend(float(g.edge_index_dict[rel].size(1)) for rel in RELATIONS)
    touched = sample['touched_inodes']
    values = g.x_dict['inode'].numpy()
    vector.extend(values[touched].mean(0).tolist() if touched else [0.] * values.shape[1])
    return np.asarray(vector, dtype=np.float32)


def summarize_scores(labels, probabilities, *, threshold=0.5):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    result = dict(samples=int(len(y)), positive=int(y.sum()),
                  accuracy=round(float(accuracy_score(y, pred)), 4),
                  precision=round(float(precision_score(y, pred, zero_division=0)), 4),
                  recall=round(float(recall_score(y, pred, zero_division=0)), 4),
                  f1=round(float(f1_score(y, pred, zero_division=0)), 4))
    if len(set(y)) > 1:
        result['roc_auc'] = round(float(roc_auc_score(y, p)), 4)
        result['average_precision'] = round(float(average_precision_score(y, p)), 4)
    return result


def prioritize(labels, scores, *, budgets=(0.05, 0.1, 0.2, 0.5)):
    """Rank candidate simulated crashes; return recovered violations in first K tests.

    A stable sort breaks tied scores by original sample order; results from tied
    baselines should NOT be interpreted as meaningful within-tie ordering.
    """
    y = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    ranked = y[order]
    first = np.flatnonzero(ranked)
    total_positive = max(int(y.sum()), 1)
    output = {'candidate_count': len(y), 'positive_candidates': int(y.sum()),
              'tests_to_first_violation': int(first[0]+1) if len(first) else None,
              'budgets': {}}
    for frac in budgets:
        k = max(1, int(round(len(y) * frac)))
        found = int(ranked[:k].sum())
        output['budgets'][f'{int(frac*100)}pct'] = {
            'tests': k, 'violations_found': found,
            'precision_at_budget': round(found/k, 4),
            'recall_at_budget': round(found/total_positive, 4),
        }
    return output


def compare_baselines(train, validation, id_test, ood_test, *, seed=42):
    features = [np.stack([graph_vector(s) for s in split])
                for split in (train, validation, id_test, ood_test)]
    labels = [np.asarray([s['label'] for s in split], dtype=int)
              for split in (train, validation, id_test, ood_test)]
    candidates = {
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=8,
                                                min_samples_leaf=2, random_state=seed,
                                                n_jobs=2),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                                               learning_rate=0.075, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        candidates['XGBoost'] = XGBClassifier(n_estimators=120, max_depth=4,
                                              learning_rate=0.07, subsample=0.9,
                                              colsample_bytree=0.9, n_jobs=2,
                                              objective='binary:logistic', random_state=seed)
    except ImportError:
        pass
    outputs = {}
    for model_name, clf in candidates.items():
        clf.fit(features[0], labels[0])
        outputs[model_name] = {}
        for part, X, y in zip(('validation', 'in_distribution', 'heldout_families'),
                              features[1:], labels[1:]):
            probabilities = clf.predict_proba(X)[:,1] if hasattr(clf,'predict_proba') else clf.predict(X)
            outputs[model_name][part] = {
                **summarize_scores(y, probabilities),
                'prioritization': prioritize(y, probabilities),
            }
    # Always include the deliberately weak, previously successful dirty-bit baseline.
    outputs['Any dirty inode'] = {}
    for part, samples, y in zip(('validation', 'in_distribution', 'heldout_families'),
                                (validation, id_test, ood_test), labels[1:]):
        p = [s['dirty_rule'] for s in samples]
        outputs['Any dirty inode'][part] = {**summarize_scores(y, p),
                                            'prioritization': prioritize(y, p)}
    return outputs


def benchmark(total_samples=5000, seed=42):
    train, val, id_test, ood = split_family_heldout(generate_dataset(total_samples, seed), seed)
    results = compare_baselines(train, val, id_test, ood, seed=seed)
    target = Path(__file__).resolve().parents[1] / 'artifacts' / 'baseline_metrics.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    results = benchmark(args.samples, args.seed)
    print(json.dumps({k: {group: v[group]['f1'] for group in ('in_distribution','heldout_families')}
                      for k, v in results.items()}, indent=2))

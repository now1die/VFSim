"""Trace exactly the saved FaultLab model, not the legacy filesystem GNN."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import torch
from challenging.generator import make_workload, ALL_FAMILIES
from challenging.independent import EXTERNAL_FAMILIES, iter_external_cases
from challenging.graphs import make_graph, EDGE_TYPES, CONSTRAINTS, OP_TYPES
from challenging.model import CrashRiskGNN

ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts'


V09_SEEDS=(11,29,47,83,131)


def _checkpoint_files(version):
    if version in ('v07','v08'):
        name='crash_risk_hardmode_v07.pt' if version=='v07' else 'crash_risk_hardmode.pt'
        example='hardmode_examples_v07.json' if version=='v07' else 'hardmode_examples.json'
        return ARTIFACTS/name, ARTIFACTS/example
    if version.startswith('v09_s'):
        suffix=version.split('v09_s',1)[1]
        if not suffix.isdigit() or int(suffix) not in V09_SEEDS:
            raise ValueError('Unknown v0.9 seed')
        folder=ARTIFACTS/'independent_v09'/('seed_'+suffix)
        return folder/'crash_risk_hardmode.pt',folder/'hardmode_examples.json'
    raise ValueError('Use v07, v08, or a registered v09_sSEED checkpoint')


@lru_cache(maxsize=8)
def model_and_metadata(version='v08'):
    path,_=_checkpoint_files(version)
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=CrashRiskGNN(variant=state.get('metadata',{}).get('model_variant','global'))
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model,state.get('metadata',{}),path


@lru_cache(maxsize=len(EXTERNAL_FAMILIES))
def _external_example(family):
    cases = iter_external_cases(selected_family=family)
    first = next(cases)
    if .07 < first['risk_probability'] < .95:
        return first
    return next((case for case in cases if .07 < case['risk_probability'] < .95), first)


def trace_saved_family(family,version='v08'):
    checkpoint,examples_path=_checkpoint_files(version)
    if version.startswith('v09_s') and family in EXTERNAL_FAMILIES:
        # Fixed example selection depends only on oracle risk band, NEVER on
        # the GNN's prediction or attention, so no cherry-picked model example.
        selected = _external_example(family)
        sim,constraint=selected['sim'],selected['constraint']
        p=float(selected['writeback_probability'])
        record={'id':selected['id'],'risk_probability':selected['risk_probability']}
    else:
        if family not in ALL_FAMILIES:
            raise ValueError(f'Unknown scenario: {family}')
        examples=json.loads(examples_path.read_text(encoding='utf8'))
        record=next((row for row in examples if row['family']==family),None)
        if record is None:raise ValueError(f'No saved example for {family}')
        sim,constraint=make_workload(family,int(record['seed']))
        p=float(record['writeback_probability'])
    graph = make_graph({'sim': sim, 'constraint': constraint, 'writeback_probability': p})
    model, metadata, path = model_and_metadata(version)
    with torch.no_grad():
        logits, tensors = model(graph, return_trace=True)
        prediction = float(torch.sigmoid(logits).item())
        normal = float(torch.sigmoid(model(graph)).item())
    if abs(prediction - normal) > 1e-7:
        raise AssertionError('Tracing must not alter model inference')
    # Preserve all 36 features, including one-hot roles and constraint semantics.
    nodes = []
    for i, (role, name) in enumerate(zip(graph.roles, graph.names)):
        raw = [float(v) for v in graph.x[i].tolist()]
        nodes.append({'id': i, 'role': role, 'label': name,
                      'features': raw,
                      'embedding': {stage: vectors[i] for stage, vectors in tensors['embeddings'].items()},
                      'query': i == tensors['query_index']})
    edges = []
    for i, (src, dst) in enumerate(graph.edge_index.T.tolist()):
        edges.append({'id': i, 'source': src, 'target': dst,
                      'relation': EDGE_TYPES[int(graph.edge_type[i])],
                      **{stage: values[i] for stage, values in tensors['weights'].items()}})
    exact, failure_states = sim.exact_failure_probability(constraint, p)
    # Oracle is returned ONLY as ground truth for comparison; it is not in model input.
    if abs(exact - record['risk_probability']) > 1e-8:
        raise AssertionError('Saved oracle differs from reconstructed workload')
    return {
        'model': 'FaultLab / ' + (version if version.startswith('v09_s') else path.name), 'architecture': '3 relational-attention layers',
        'checkpoint_samples': metadata.get('samples'), 'checkpoint_seed': metadata.get('seed'),
        'checkpoint_file': str(path.relative_to(ARTIFACTS)), 'model_variant': model.variant, 'family': family, 'case_id': record['id'],
        'risk_prediction': prediction, 'oracle_risk': exact, 'absolute_error': abs(prediction - exact),
        'writeback_probability': p, 'constraint': constraint.kind,
        'constraint_paths': list(constraint.paths),
        'node_count': len(nodes), 'edge_count': len(edges), 'nodes': nodes, 'edges': edges,
        'layers': ['layer1', 'layer2', 'layer3'], 'feature_width': graph.x.shape[1],
        'embedding_width': len(tensors['embeddings']['encoder'][0]),
        'readout': tensors['readout'],
        'attention_description': ('Effective directed-edge coefficients from the actual three GNN '
           'message aggregation layers; coefficients sum to at most one per receiving node due to '
           'the model\'s mass clamp. Attention is not a causal attribution.'),
        'failure_example': failure_states[0] if failure_states else None,
        'warning': 'Synthetic application invariant and independent writeback model; not ext4 evidence.'
    }

"""Torch-native 2-layer relation-aware attention GNN (no torch-geometric needed)."""
from __future__ import annotations
import math
import torch
from torch import nn
from graph.builder import FEATURE_WIDTHS, NODE_TYPES, RELATIONS


def relation_name(relation):
    return '__'.join(relation)


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_projections = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in NODE_TYPES})
        self.keys = nn.ModuleDict({relation_name(r): nn.Linear(hidden_dim, hidden_dim, bias=False) for r in RELATIONS})
        self.values = nn.ModuleDict({relation_name(r): nn.Linear(hidden_dim, hidden_dim, bias=False) for r in RELATIONS})
        self.queries = nn.ModuleDict({relation_name(r): nn.Linear(hidden_dim, hidden_dim, bias=False) for r in RELATIONS})
        self.hidden_dim = hidden_dim

    def forward(self, features, edges, trace=False):
        """Aggregate incoming relation messages; optionally expose actual attention.

        A destination's incoming edge weights are normalized *across relation
        types*, matching the denominator used to calculate its message. They
        are model-internal attention coefficients, not explanations of risk.
        """
        accum = {t: torch.zeros_like(features[t]) for t in NODE_TYPES}
        weights = {t: torch.zeros((len(features[t]), 1), dtype=features[t].dtype, device=features[t].device)
                   for t in NODE_TYPES}
        candidates = []
        for relation, edge_index in edges.items():
            if edge_index.size(1) == 0:
                continue
            src_type, _, dst_type = relation
            key = relation_name(relation)
            source, target = edge_index[0], edge_index[1]
            q = self.queries[key](features[dst_type][target])
            k = self.keys[key](features[src_type][source])
            scores = (q * k).sum(-1, keepdim=True) / math.sqrt(self.hidden_dim)
            raw_weights = scores.clamp(-8, 8).exp()
            values = self.values[key](features[src_type][source]) * raw_weights
            accum[dst_type] = accum[dst_type].index_add(0, target, values)
            weights[dst_type] = weights[dst_type].index_add(0, target, raw_weights)
            if trace:
                candidates.append((relation, edge_index, raw_weights))
        out = {t: torch.relu(self.self_projections[t](features[t]) + accum[t] / weights[t].clamp(min=1))
               for t in NODE_TYPES}
        if not trace:
            return out
        attention = []
        for relation, edge_index, raw_weights in candidates:
            dst_type = relation[2]
            for position, (src_index, dst_index) in enumerate(edge_index.T.tolist()):
                denominator = weights[dst_type][dst_index].clamp(min=1)
                attention.append({
                    "relation": relation,
                    "source_index": int(src_index),
                    "target_index": int(dst_index),
                    "coefficient": float((raw_weights[position] / denominator).item()),
                    "unnormalized": float(raw_weights[position].item()),
                })
        return out, attention


class RiskGNN(nn.Module):
    def __init__(self, hidden_dim=32, heads=4, dropout=0.2):
        super().__init__()
        # heads kept for backwards-compatible model constructor; attention is relation-aware.
        self.encoders = nn.ModuleDict({t: nn.Sequential(nn.Linear(FEATURE_WIDTHS[t], hidden_dim), nn.ReLU())
                                       for t in NODE_TYPES})
        self.conv1 = GraphAttentionLayer(hidden_dim)
        self.conv2 = GraphAttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim + FEATURE_WIDTHS["inode"], 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    @staticmethod
    def pair_contrasts(x_dict, edge_index_dict):
        """Graph readout: differences across declared application-pair edges.

        This *explicit structural feature* is derived from the same graph used
        for message passing. It does NOT read a label or recovered oracle state.
        Compare with simple graph-feature baselines; this is not evidence that
        attention itself is beneficial.
        """
        features = x_dict['inode']
        relations = edge_index_dict.get(('inode', 'application_pair', 'inode'))
        if relations is None or relations.numel() == 0:
            return features.new_zeros(FEATURE_WIDTHS['inode'])
        a, b = relations[0], relations[1]
        return (features[a] - features[b]).abs().max(dim=0).values

    def forward(self, x_dict, edge_index_dict, touched_node_ids):
        if not touched_node_ids:
            raise ValueError('No surviving touched inode is present in the graph')
        x = {t: self.encoders[t](x_dict[t]) for t in NODE_TYPES}
        x = {t: self.dropout(v) for t, v in self.conv1(x, edge_index_dict).items()}
        x = self.conv2(x, edge_index_dict)
        pooled = x['inode'][touched_node_ids].mean(dim=0)
        context = self.pair_contrasts(x_dict, edge_index_dict)
        return self.mlp(torch.cat([pooled, context], dim=0)).view(1)

    def predict_risk(self, x_dict, edge_index_dict, touched_node_ids):
        with torch.no_grad():
            return torch.sigmoid(self(x_dict, edge_index_dict, touched_node_ids))

    @torch.no_grad()
    def inspect(self, x_dict, edge_index_dict, touched_node_ids):
        """Compute attention and embeddings on the exact inference graph.

        This is a diagnostic trace, NOT a faithful causal explanation. The
        graph/scores are computed under eval() so dropout is disabled.
        """
        self.eval()
        encoded = {t: self.encoders[t](x_dict[t]) for t in NODE_TYPES}
        first, attention_1 = self.conv1(encoded, edge_index_dict, trace=True)
        second, attention_2 = self.conv2(first, edge_index_dict, trace=True)
        score = None
        logit = None
        if touched_node_ids:
            pooled = second['inode'][touched_node_ids].mean(dim=0)
            context = self.pair_contrasts(x_dict, edge_index_dict)
            logit = float(self.mlp(torch.cat([pooled, context], dim=0)).view(1).item())
            score = float(torch.sigmoid(torch.tensor(logit)).item())
        return {
            'pair_contrasts': self.pair_contrasts(x_dict, edge_index_dict).tolist(),
            'embeddings': {
                'encoded': {t: v.tolist() for t, v in encoded.items()},
                'layer1': {t: v.tolist() for t, v in first.items()},
                'layer2': {t: v.tolist() for t, v in second.items()},
            },
            'attention': {'layer1': attention_1, 'layer2': attention_2},
            'logit': logit,
            'score': score,
        }

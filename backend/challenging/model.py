"""Relation-aware pre-crash risk GNN with optional *faithful* inference traces.

The trace runs the same operations as forward() and exposes effective edge
aggregation weights (not a causal explanation). Legacy checkpoints are compatible.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from challenging.graphs import F, EDGE_TYPES


class RelationalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.self_proj = nn.Linear(dim, dim)
        self.relation_value = nn.Embedding(len(EDGE_TYPES), dim)
        self.relation_score = nn.Embedding(len(EDGE_TYPES), 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edges, edge_types, return_attention=False):
        src, dst = edges
        q = self.query(x[dst])
        k = self.key(x[src])
        score = ((q * k).sum(-1) / math.sqrt(x.shape[1]) +
                 self.relation_score(edge_types).flatten()).clamp(-8, 8)
        weights = score.exp()
        mass = x.new_zeros((len(x), 1)).index_add(0, dst, weights.unsqueeze(-1))
        effective_weights = weights / mass[dst].flatten().clamp(min=1)
        message = (self.value(x[src]) + self.relation_value(edge_types)) * weights.unsqueeze(-1)
        aggregate = torch.zeros_like(x).index_add(0, dst, message)
        output = self.norm(torch.relu(self.self_proj(x) + aggregate / mass.clamp(min=1)))
        if return_attention:
            # These are the EXACT effective coefficients used in message aggregation.
            # They sum to <= 1 for a destination because mass is clamped at 1.
            return output, effective_weights
        return output


class CrashRiskGNN(nn.Module):
    def __init__(self, dim=32, dropout=.1, variant="global"):
        super().__init__()
        if variant not in ("global", "operand"):
            raise ValueError("Unknown readout variant")
        self.variant = variant
        self.encode = nn.Sequential(nn.Linear(F, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList([RelationalAttention(dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        output_width = dim * 4 + 2 if variant == "operand" else dim * 2 + 1
        self.classifier = nn.Sequential(nn.Linear(output_width, 96 if variant == "operand" else 64),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(96 if variant == "operand" else 64, 1))

    def forward(self, g, return_trace=False):
        x = self.encode(g.x)
        if return_trace:
            embeddings = {'encoder': x.detach().cpu().tolist()}
            weights = {}
        for i, conv in enumerate(self.layers, start=1):
            if return_trace:
                updated, att = conv(x, g.edge_index, g.edge_type, return_attention=True)
                weights[f'layer{i}'] = att.detach().cpu().tolist()
            else:
                updated = conv(x, g.edge_index, g.edge_type)
            x = x + self.dropout(updated) * .5
            if return_trace:
                embeddings[f'layer{i}'] = x.detach().cpu().tolist()
        query = x[0]
        pooled = x.mean(0)
        policy = g.x[0, 26].reshape(1)
        if self.variant == "operand":
            # Structural inductive bias: the number and state of application
            # operands matter when a commit spans several dependent objects.
            mask = g.x[:, 30] > .5
            operand_nodes = x[mask] if bool(mask.any()) else x[0:1]
            operands_mean = operand_nodes.mean(0)
            operands_max = operand_nodes.max(0).values
            count = x.new_tensor([min(1., len(operand_nodes)/6)])
            readout_input = torch.cat((query, pooled, operands_mean, operands_max, policy, count), dim=0)
        else:
            readout_input = torch.cat((query, pooled, policy), dim=0)
        logit = self.classifier(readout_input).flatten()
        if return_trace:
            return logit, {'embeddings': embeddings, 'weights': weights,
                           'query_index': 0, 'readout': {'query': query.detach().cpu().tolist(),
                           'pooled': pooled.detach().cpu().tolist(),
                           'policy': float(policy.item()), 'variant': self.variant}}
        return logit

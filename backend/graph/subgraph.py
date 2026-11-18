"""Undirected neighborhood extraction for the heterogeneous relation graph."""
from __future__ import annotations
import torch
from .builder import HeteroGraph, NODE_TYPES


def get_touched_inode_indices(vfs, transaction):
    mapping = {i.inode_id: n for n, i in enumerate(vfs.get_all_inodes())}
    return list(dict.fromkeys(mapping[i] for i in transaction.inodes_touched if i in mapping))


def get_neighbors(graph, node_type, node_index):
    result = set()
    for (src_type, _name, dst_type), edges in graph.edge_index_dict.items():
        for a, b in edges.T.tolist():
            if src_type == node_type and a == node_index:
                result.add((dst_type, b))
            if dst_type == node_type and b == node_index:
                result.add((src_type, a))
    return sorted(result)


def get_1hop_inode_neighbors(graph, touched_indices):
    return _nodes_within(graph, touched_indices, 1)


def _nodes_within(graph, touched_indices, hops):
    visited = {('inode', i) for i in touched_indices}
    frontier = set(visited)
    for _ in range(hops):
        next_frontier = set()
        for kind, idx in frontier:
            next_frontier.update(get_neighbors(graph, kind, idx))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    return {t: sorted(i for typ, i in visited if typ == t) for t in NODE_TYPES}


def get_2hop_subgraph(graph, touched_indices):
    return _nodes_within(graph, touched_indices, 2)


def extract_2hop_subgraph(graph, touched_indices):
    keep = get_2hop_subgraph(graph, touched_indices)
    mapping = {t: {old: new for new, old in enumerate(keep[t])} for t in NODE_TYPES}
    x_dict = {t: graph.x_dict[t][keep[t]] for t in NODE_TYPES}
    edge_index_dict = {}
    for (src, name, dst), edge_index in graph.edge_index_dict.items():
        pairs = [(mapping[src][a], mapping[dst][b]) for a, b in edge_index.T.tolist()
                 if a in mapping[src] and b in mapping[dst]]
        edge_index_dict[(src, name, dst)] = (
            torch.tensor(pairs, dtype=torch.long).T.contiguous() if pairs
            else torch.empty((2, 0), dtype=torch.long)
        )
    return HeteroGraph(x_dict, edge_index_dict)


def get_subgraph_touched_inode_indices(graph, touched_indices):
    ids = get_2hop_subgraph(graph, touched_indices)['inode']
    mapping = {old: new for new, old in enumerate(ids)}
    return [mapping[i] for i in touched_indices if i in mapping]

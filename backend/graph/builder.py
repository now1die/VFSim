"""PyTorch-only heterogeneous filesystem graph, without optional PyG wheels."""
from __future__ import annotations

from types import SimpleNamespace
import torch

NODE_TYPES = ('inode', 'dirent', 'fd', 'link', 'rename')
FEATURE_WIDTHS = {'inode': 9, 'dirent': 1, 'fd': 3, 'link': 1, 'rename': 2}
RELATIONS = (
    ('inode', 'contains', 'dirent'), ('dirent', 'points_to', 'inode'),
    ('inode', 'application_pair', 'inode'),
    ('inode', 'has_link', 'link'), ('link', 'points_to', 'inode'),
    ('fd', 'open_by', 'inode'), ('inode', 'opened_by', 'fd'),
    ('inode', 'rename_source', 'rename'), ('rename', 'rename_target', 'inode'),
)


class HeteroGraph:
    def __init__(self, x_dict=None, edge_index_dict=None):
        self.x_dict = x_dict or {t: torch.empty((0, FEATURE_WIDTHS[t]), dtype=torch.float32) for t in NODE_TYPES}
        self.edge_index_dict = edge_index_dict or {r: torch.empty((2, 0), dtype=torch.long) for r in RELATIONS}

    @property
    def node_types(self):
        return list(self.x_dict)

    @property
    def edge_types(self):
        return list(self.edge_index_dict)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return SimpleNamespace(edge_index=self.edge_index_dict[key])
        return SimpleNamespace(x=self.x_dict[key])


def build_graph(vfs):
    inodes = vfs.get_all_inodes()
    dirents = vfs.get_all_dirents()
    index = {i.inode_id: n for n, i in enumerate(inodes)}
    fd = [f for f in vfs.file_handles.values() if f.inode_id in index]
    renames = [r for r in vfs.get_rename_history() if r['inode_id'] in index]
    aliases = {}
    for d in dirents:
        aliases[d.inode_id] = aliases.get(d.inode_id, 0) + 1
    links = [i for i in inodes for _ in range(max(aliases.get(i.inode_id, 0) - 1, 0))]
    durable_index = max(0, min(vfs._durable_index, len(vfs._history) - 1))
    durable_snapshot = vfs._history[durable_index]
    durable_nodes = durable_snapshot['inodes']
    durable_paths = durable_snapshot['paths']
    max_size = max([1] + [i.size for i in inodes] +
                   [node['size'] for node in durable_nodes.values()])
    def durably_named(inode):
        return float(durable_paths.get(inode.path) == inode.inode_id)
    def durable_size(inode):
        return durable_nodes.get(str(inode.inode_id), {}).get('size', 0) / max_size
    def durable_links(inode):
        return float(durable_nodes.get(str(inode.inode_id), {}).get('link_count', 0))

    features = {
        'inode': [[float(i.inode_type == 'file'), float(i.inode_type == 'directory'),
                   i.size / max_size, float(i.link_count), float(i.dirty),
                   float(i.path.count('/') - (0 if i.path == '/' else 1)),
                   durable_size(i), durably_named(i), durable_links(i)] for i in inodes],
        'dirent': [[float(len(d.name))] for d in dirents],
        'fd': [[float('r' in f.mode), float('w' in f.mode), float('a' in f.mode)] for f in fd],
        'link': [[1.0] for _ in links],
        'rename': [[float(len(r['old_path'])), float(len(r['new_path']))] for r in renames],
    }
    x_dict = {k: torch.tensor(v, dtype=torch.float32).reshape(-1, FEATURE_WIDTHS[k])
              for k, v in features.items()}
    edges = {r: [] for r in RELATIONS}
    for constraint in getattr(vfs, 'risk_pairs', []):
        a, b = index.get(constraint['inode_a']), index.get(constraint['inode_b'])
        if a is not None and b is not None:
            edges[('inode', 'application_pair', 'inode')].extend([(a, b), (b, a)])
    for j, d in enumerate(dirents):
        if d.parent_inode_id in index and d.inode_id in index:
            edges[RELATIONS[0]].append((index[d.parent_inode_id], j))
            edges[RELATIONS[1]].append((j, index[d.inode_id]))
    for j, i in enumerate(links):
        edges[RELATIONS[3]].append((index[i.inode_id], j))
        edges[RELATIONS[4]].append((j, index[i.inode_id]))
    for j, handle in enumerate(fd):
        edges[RELATIONS[5]].append((j, index[handle.inode_id]))
        edges[RELATIONS[6]].append((index[handle.inode_id], j))
    for j, rename in enumerate(renames):
        edges[RELATIONS[7]].append((index[rename['inode_id']], j))
        edges[RELATIONS[8]].append((j, index[rename['inode_id']]))
    edge_index_dict = {
        r: torch.tensor(pairs, dtype=torch.long).T.contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        for r, pairs in edges.items()
    }
    return HeteroGraph(x_dict, edge_index_dict)

"""Live, reproducible diagnostics for the research console.

All heterogeneous graph entities and attention coefficients are traced from
exactly the graph passed to inference, never arbitrary UI placeholders.
"""
from collections import Counter
from math import sqrt
from vfs.filesystem import VirtualFileSystem
from graph.builder import NODE_TYPES, RELATIONS, FEATURE_WIDTHS, build_graph
from graph.subgraph import get_touched_inode_indices, get_2hop_subgraph, extract_2hop_subgraph, get_subgraph_touched_inode_indices

FEATURE_LABELS = {
    'inode': ['is_file', 'is_directory', 'relative_size', 'link_count', 'dirty', 'path_depth',
              'last_fsync_size', 'named_at_fsync', 'last_fsync_link_count'],
    'dirent': ['name_length'],
    'fd': ['read_mode', 'write_mode', 'append_mode'],
    'link': ['is_hard_link_alias'],
    'rename': ['old_path_length', 'new_path_length'],
}


def _labels(vfs):
    inodes = vfs.get_all_inodes()
    dirents = vfs.get_all_dirents()
    file_handles = [f for f in vfs.file_handles.values() if f.inode_id in {n.inode_id for n in inodes}]
    renames = [r for r in vfs.get_rename_history() if r['inode_id'] in {n.inode_id for n in inodes}]
    names = [path for path in vfs.dirents.keys()]
    links = []
    for inode in inodes:
        count = sum(1 for d in dirents if d.inode_id == inode.inode_id)
        for idx in range(max(0, count - 1)):
            links.append({
                'label': f'Alias link {idx + 1}', 'subtitle': f'inode #{inode.inode_id}',
                'inodeNumber': inode.inode_id, 'path': inode.path,
            })
    return {
        'inode': [{
            'label': ('/' if inode.path == '/' else inode.path.split('/')[-1]),
            'subtitle': inode.path, 'path': inode.path, 'inodeNumber': inode.inode_id,
            'dirty': inode.dirty, 'sizeBytes': inode.size,
            'inodeType': inode.inode_type, 'linkCount': inode.link_count,
        } for inode in inodes],
        'dirent': [{
            'label': d.name, 'subtitle': names[idx] if idx < len(names) else d.name,
            'path': names[idx] if idx < len(names) else None,
            'inodeNumber': d.inode_id, 'parentInodeNumber': d.parent_inode_id,
        } for idx, d in enumerate(dirents)],
        'fd': [{'label': f'fd {handle.fd}', 'subtitle': f'{handle.mode} · {handle.path}',
                'path': handle.path, 'descriptor': handle.fd, 'mode': handle.mode}
               for handle in file_handles],
        'link': links,
        'rename': [{'label': 'rename', 'subtitle': f"{r['old_path']} → {r['new_path']}",
                    'oldPath': r['old_path'], 'newPath': r['new_path'],
                    'inodeNumber': r['inode_id']} for r in renames],
    }


def inspection_payload(vfs, trained, transaction_id=None, scope="scoring"):
    if scope not in ('scoring', 'full'):
        raise ValueError('scope must be scoring or full')
    txs = vfs.get_transactions()
    if scope == 'full':
        transaction_id = None
    if scope == 'scoring' and transaction_id is None and txs:
        transaction_id = txs[-1].txn_id
    tx = None
    graph_vfs = vfs
    if transaction_id:
        offset = next((i+1 for i, item in enumerate(txs) if item.txn_id == transaction_id), None)
        if offset is None:
            raise ValueError('Transaction not found')
        tx = txs[offset-1]
        graph_vfs = vfs.replay_at(offset)
    full_graph = build_graph(graph_vfs)
    touched_indices = get_touched_inode_indices(graph_vfs, tx) if tx else []
    if touched_indices and scope == 'scoring':
        selected_indices = get_2hop_subgraph(full_graph, touched_indices)
        graph = extract_2hop_subgraph(full_graph, touched_indices)
        local_touched = get_subgraph_touched_inode_indices(full_graph, touched_indices)
    else:
        selected_indices = {k: list(range(full_graph.x_dict[k].size(0))) for k in NODE_TYPES}
        graph = full_graph
        local_touched = []
    labels = _labels(graph_vfs)
    trace = trained[0].inspect(graph.x_dict, graph.edge_index_dict, local_touched if scope == 'scoring' else []) if trained else None
    nodes = []
    for node_type in NODE_TYPES:
        for local_index, global_index in enumerate(selected_indices[node_type]):
            meta = labels[node_type][global_index]
            node = {
                'id': f'{node_type}:{local_index}',
                'nodeType': node_type,
                'localIndex': local_index,
                'fullIndex': global_index,
                'touched': node_type == 'inode' and local_index in local_touched,
                'features': {name: round(float(value), 6) for name, value in
                             zip(FEATURE_LABELS[node_type], graph.x_dict[node_type][local_index].tolist())},
                **meta,
            }
            if trace:
                node['embeddings'] = {stage: [round(float(v), 6) for v in
                    trace['embeddings'][stage][node_type][local_index]]
                    for stage in ('encoded', 'layer1', 'layer2')}
                node['embeddingNorms'] = {stage: round(sqrt(sum(v*v for v in node['embeddings'][stage])), 5)
                                          for stage in node['embeddings']}
            nodes.append(node)
    lookup = {(n['nodeType'], n['localIndex']): n['id'] for n in nodes}
    weights = {'layer1': {}, 'layer2': {}}
    if trace:
        for stage in ('layer1', 'layer2'):
            for record in trace['attention'][stage]:
                relation = tuple(record['relation'])
                key = (relation, record['source_index'], record['target_index'])
                weights[stage][key] = round(record['coefficient'], 8)
    edges = []
    for relation, indices in graph.edge_index_dict.items():
        for src_index, dst_index in indices.T.tolist():
            source = lookup[(relation[0], src_index)]
            target = lookup[(relation[2], dst_index)]
            key = (relation, src_index, dst_index)
            edges.append({
                'id': f'{relation[0]}:{src_index}-{relation[1]}-{relation[2]}:{dst_index}',
                'source': source, 'target': target,
                'relation': relation[1],
                'relationKey': '__'.join(relation),
                'attention': {stage: weights[stage].get(key) for stage in ('layer1', 'layer2')},
            })
    counts = Counter(n['nodeType'] for n in nodes)
    return {
        'transactionId': transaction_id,
        'operation': tx.op_type if tx else None,
        'graphScope': scope,
        'scope': 'Exact 2-hop transaction scoring graph' if scope == 'scoring' and touched_indices else 'Current full heterogeneous graph (no prediction)',
        'modelLoaded': bool(trained),
        'score': round(trace['score'], 6) if trace and trace['score'] is not None else None,
        'logit': round(trace['logit'], 6) if trace and trace['logit'] is not None else None,
        'nodes': nodes,
        'edges': edges,
        'touched': [f'inode:{i}' for i in local_touched],
        'counts': {k: counts[k] for k in NODE_TYPES},
        'relations': {'__'.join(r): int(graph.edge_index_dict[r].size(1)) for r in RELATIONS},
        'featureSchema': FEATURE_LABELS,
        'pairContrastReadout': [round(float(v), 6) for v in trace['pair_contrasts']] if trace else None,
        'attentionDescription': ('Actual single-head relation-aware model attention; normalized per destination '
                                 'across all incoming edge types for each layer (minimum denominator: 1). It is NOT a causal attribution '
                                 'or a calibrated probability of a physical filesystem failure.'),
        'trainingWarning': ('Checkpoint uses observed simulated recovery outcomes, with structural pair contrasts. These are not calibrated physical crash probabilities. Checkpoints were trained on simulator application-invariant outcomes. '
                            'Held-out workloads and baseline results appear in the benchmark panel. '
                            'No real ext4 crash outcomes are used.'),
    }


def transaction_analysis(vfs, trained, transaction_id):
    transactions = vfs.get_transactions()
    offset = next((i+1 for i, t in enumerate(transactions) if t.txn_id == transaction_id), None)
    if offset is None:
        raise ValueError('Transaction not found')
    transaction = transactions[offset-1]
    before = vfs._history[offset-1]
    after = vfs._history[offset]
    before_paths, after_paths = set(before['paths']), set(after['paths'])
    replay = VirtualFileSystem()
    replay.load_snapshot(after)
    graph = inspection_payload(vfs, trained, transaction_id)
    previews = [vfs.crash_preview(transaction_id, scenario) for scenario in
                ('crash-last-sync', 'crash-before-commit', 'crash-after-commit')]
    return {
        'transactionId': transaction_id,
        'operation': transaction.op_type,
        'metadata': transaction.metadata,
        'timestamp': transaction.timestamp,
        'sequence': offset,
        'totalOperations': len(transactions),
        'touchedInodeNumbers': transaction.inodes_touched,
        'dirtyInodeNumbers': [i.inode_id for i in replay.get_all_inodes() if i.dirty],
        'score': graph['score'],
        'modelLoaded': bool(trained),
        'namespaceDelta': {
            'createdPaths': sorted(after_paths-before_paths),
            'deletedPaths': sorted(before_paths-after_paths),
            'modifiedSizes': sorted(p for p in before_paths & after_paths
                if before['inodes'][str(before['paths'][p])]['size'] !=
                   after['inodes'][str(after['paths'][p])]['size']),
        },
        'crashPreviews': previews,
        'disclaimer': 'Crash previews reconstruct simplified VFS snapshots, not physical Linux block-level crashes.',
    }

"""Application-invariant crash oracle over the *simplified* VFSim durability model.

Crashes recover the most recent global fsync snapshot, not actual filesystem
blocks. A label is 1 only if the recovered snapshot violates a declared pair
invariant while the final live state satisfies it. Safe and unsafe cases have
similar operation sequences; a bare 'dirty inode' check cannot label them.

This is NOT ext4/Btrfs crash-fault ground truth. The synthetic workload families
and held-out-family split are intended to expose generalization limitations.
"""
from __future__ import annotations
from collections import Counter
import random
from vfs.filesystem import VirtualFileSystem
from graph.builder import build_graph
from graph.subgraph import (get_touched_inode_indices, extract_2hop_subgraph,
                            get_subgraph_touched_inode_indices)

TOTAL_SAMPLES = 5000
RANDOM_SEED = 42
TRAIN_FAMILIES = ('size_pair', 'create_pair', 'link_pair', 'mixed')
OOD_FAMILIES = ('rename_pair', 'unlink_pair')
FAMILY_WEIGHTS = {'size_pair': 2, 'create_pair': 2, 'link_pair': 2,
                  'mixed': 2, 'rename_pair': 1, 'unlink_pair': 1}


def invariant_holds(snapshot, paths, kind):
    """Oracle: evaluate *recovered* application invariants, not an op label."""
    a, b = paths
    mapping = snapshot['paths']
    present_a, present_b = a in mapping, b in mapping
    if kind == 'presence_equal':
        return present_a == present_b
    if kind == 'size_equal':
        if not (present_a and present_b):
            return False
        a_node = snapshot['inodes'][str(mapping[a])]
        b_node = snapshot['inodes'][str(mapping[b])]
        return a_node['size'] == b_node['size']
    raise ValueError(f'Unsupported application invariant: {kind}')


def make_sample(vfs, tx, *, family, paths, invariant, seed_plan):
    final = vfs.snapshot()
    if not invariant_holds(final, paths, invariant):
        raise AssertionError('Invalid workload: live application invariant is already broken')
    preview = vfs.crash_preview(tx.txn_id, 'crash-last-sync')
    # Only evaluate the recovered durable state. Do not use the dirty bit or
    # an operation count to manufacture a training label.
    recovered = vfs._history[preview['survivedOps']]
    label = int(not invariant_holds(recovered, paths, invariant))
    if label != int(seed_plan):
        raise AssertionError(f'Oracle disagrees with scenario construction: {family} {seed_plan} {label}')
    full = build_graph(vfs)
    touched = get_touched_inode_indices(vfs, tx)
    if not touched:
        raise ValueError('No touched inode survives the selected operation')
    return {
        'graph': extract_2hop_subgraph(full, touched),
        'touched_inodes': get_subgraph_touched_inode_indices(full, touched),
        'label': label,
        'family': family,
        'scenario': tx.op_type,
        'oracle': invariant,
        'oracle_paths': paths,
        'crash_boundary': preview['survivedOps'],
        'operation_count': len(vfs.get_transactions()),
        'dirty_rule': int(any(node.dirty for node in vfs.get_all_inodes())),
    }


def _workload(rng, family, unsafe):
    v = VirtualFileSystem()
    v.create('/work', 'directory')
    v.create('/work/backups', 'directory')
    # Background context varies independent of the outcome label.
    for i in range(rng.randint(0, 3)):
        p = f'/work/context{i}.dat'
        v.create(p)
        if rng.random() < 0.65:
            v.write(p, rng.randint(1, 4000))
    if family == 'mixed':
        chosen = rng.choice(('size_pair', 'create_pair', 'link_pair'))
        # Unrelated churn is permitted but must be durable before the paired operation.
        v.create('/work/backups/junk')
        v.unlink('/work/backups/junk')
    else:
        chosen = family
    a, b = '/work/left.dat', '/work/backups/right.dat'
    invariant = 'size_equal' if chosen == 'size_pair' else 'presence_equal'
    if chosen in ('size_pair', 'link_pair', 'rename_pair', 'unlink_pair'):
        v.create(a)
        v.create(b)
        if chosen == 'size_pair':
            initial_size = rng.randint(10, 4000)
            v.write(a, initial_size)
            v.write(b, initial_size)
        if chosen in ('link_pair', 'unlink_pair'):
            initial_size = rng.randint(10, 3000)
            v.write(a, initial_size)
            v.write(b, initial_size)
        v.declare_pair(a, b)
    if chosen == 'unlink_pair':
        v.link(a, '/work/left-alias.dat')
        v.link(b, '/work/backups/right-alias.dat')
    v.fsync('/')
    delta = rng.randint(1, 6000)
    if chosen == 'size_pair':
        v.write(a, delta)
        if unsafe:
            v.fsync('/')
        tx = v.write(b, delta)
        targets = (a, b)
    elif chosen == 'create_pair':
        v.create(a)
        if unsafe:
            v.fsync('/')
        tx = v.create(b)
        v.declare_pair(a, b)
        targets = (a, b)
    elif chosen == 'link_pair':
        targets = ('/work/left-alias.dat', '/work/backups/right-alias.dat')
        v.link(a, targets[0])
        if unsafe:
            v.fsync('/')
        tx = v.link(b, targets[1])
    elif chosen == 'rename_pair':
        targets = ('/work/left-renamed.dat', '/work/backups/right-renamed.dat')
        v.rename(a, targets[0])
        if unsafe:
            v.fsync('/')
        tx = v.rename(b, targets[1])
    elif chosen == 'unlink_pair':
        targets = ('/work/left-alias.dat', '/work/backups/right-alias.dat')
        v.unlink(targets[0])
        if unsafe:
            v.fsync('/')
        tx = v.unlink(targets[1])
    else:
        raise ValueError(f'Unknown family: {chosen}')
    return make_sample(v, tx, family=family, paths=targets,
                       invariant=invariant, seed_plan=unsafe)


def generate_dataset(total_samples=TOTAL_SAMPLES, seed=RANDOM_SEED):
    if total_samples < 20 or total_samples % 2:
        raise ValueError('total_samples must be even and at least 20')
    rng = random.Random(seed)
    # Always ensure each family has both outcome classes; round-robin weights
    # make the mixture reproducible for both small regression tests and 5k runs.
    plan = []
    families = [family for family, weight in FAMILY_WEIGHTS.items() for _ in range(weight)]
    for i in range(total_samples // 2):
        family = families[i % len(families)]
        plan.extend(((family, False), (family, True)))
    rng.shuffle(plan)
    return [_workload(rng, family, unsafe) for family, unsafe in plan]


def split_dataset(samples, seed=RANDOM_SEED):
    """Legacy three-way stratified split, retained for existing exercises/tests."""
    rng = random.Random(seed)
    groups = {(s['family'], s['label']) for s in samples}
    outputs = [[], [], []]
    for family, label in sorted(groups):
        subset = [s for s in samples if (s['family'], s['label']) == (family, label)]
        rng.shuffle(subset)
        a, b = round(len(subset) * .7), round(len(subset) * .85)
        for out, part in zip(outputs, (subset[:a], subset[a:b], subset[b:])):
            out.extend(part)
    for out in outputs:
        rng.shuffle(out)
    return tuple(outputs)


def split_family_heldout(samples, seed=RANDOM_SEED):
    """Train/validation/ID test on four families; reserve two entire families for OOD."""
    rng = random.Random(seed)
    train, validation, id_test, ood = [], [], [], []
    for family in TRAIN_FAMILIES:
        for label in (0, 1):
            subset = [s for s in samples if s['family'] == family and s['label'] == label]
            rng.shuffle(subset)
            n = len(subset)
            a, b = round(.75 * n), round(.875 * n)
            train.extend(subset[:a])
            validation.extend(subset[a:b])
            id_test.extend(subset[b:])
    ood = [s for s in samples if s['family'] in OOD_FAMILIES]
    if len(train) + len(validation) + len(id_test) + len(ood) != len(samples):
        raise AssertionError('Family split lost or duplicated examples')
    for group in (train, validation, id_test, ood):
        rng.shuffle(group)
    return train, validation, id_test, ood


def dataset_summary(samples):
    return {'size': len(samples), 'families': dict(Counter(s['family'] for s in samples)),
            'outcomes': dict(Counter(str(s['label']) for s in samples)),
            'operations': dict(Counter(s['scenario'] for s in samples))}

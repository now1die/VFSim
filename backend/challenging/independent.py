"""Frozen, separately implemented application workload families for external synthetic evaluation.

These scenarios deliberately DO NOT feed the training generator. They remain on
VFSim's toy independent-writeback oracle, NOT ext4 or production-app traces.
Each family contains valid live states and tests an application invariant after
partial simulated persistence. The random-seed/policy manifest is deterministic.
"""
from __future__ import annotations

from collections import Counter
import random
from challenging.simulator import Constraint, DurabilityModel, check_constraint

EXTERNAL_FAMILIES = (
    'editor_autosave',
    'package_upgrade',
    'mirrored_alias_rotation',
    'manifest_publication',
    'directory_migration',
    'snapshot_batch',
)
POLICIES = (0.12, 0.25, 0.40, 0.55, 0.70, 0.85)
EXTERNAL_SEED = 20261116


def make_external_workload(family: str, seed: int):
    """Create a workload with no post-crash outcome or risk label in its features.

    Family templates are distinct from the training generator. They are not
    independently sourced real application traces. All updates, syncs, and
    partial persistence use the same documented toy DurabilityModel.
    """
    if family not in EXTERNAL_FAMILIES:
        raise ValueError(f'Unknown external workload family: {family}')
    r = random.Random(seed)
    fs = DurabilityModel()
    old = r.randint(1, 4)
    new = old + r.randint(1, 4)

    # Different storage paths and background churn from the training templates.
    if family == 'editor_autosave':
        fs.create('/documents/working', old)
        fs.create('/history/revision', old)
        fs.create('/documents/autosave', old)
        constraint = Constraint('equal_versions', ('/documents/working', '/history/revision'))
    elif family == 'package_upgrade':
        for path in ('/pkg/bin/cli', '/pkg/lib/core', '/pkg/lib/helper', '/pkg/manifest/release'):
            fs.create(path, old)
        constraint = Constraint('equal_versions', ('/pkg/bin/cli', '/pkg/lib/core', '/pkg/lib/helper', '/pkg/manifest/release'))
    elif family == 'mirrored_alias_rotation':
        fs.create('/objects/revision-a', old)
        fs.link('/objects/revision-a', '/active/current')
        fs.link('/objects/revision-a', '/standby/current')
        fs.create('/objects/revision-b', new)
        constraint = Constraint('alias_identity', ('/active/current', '/standby/current'))
    elif family == 'manifest_publication':
        fs.create('/catalog/index', old)
        fs.create('/payloads/blob', old)
        fs.create('/payloads/old-blob', old)
        constraint = Constraint('requires_data', ('/catalog/index', '/payloads/blob'))
    elif family == 'directory_migration':
        for path in ('/legacy/components/a', '/legacy/components/b', '/legacy/components/index'):
            fs.create(path, old)
        constraint = Constraint('presence_together', ('/new/modules/a', '/new/index/b'))
    else:  # snapshot_batch
        count = r.randint(3, 6)
        paths = tuple(f'/snapshots/group{i % 3}/part{i}' for i in range(count))
        for path in paths:
            fs.create(path, old)
        constraint = Constraint('equal_versions', paths)

    # Set a shared stable state. All operations after this checkpoint are
    # the candidate crash window and may be partially persisted.
    fs.checkpoint()

    if family == 'editor_autosave':
        fs.write('/documents/autosave', new)
        if r.random() < .5: fs.sync_file('/documents/autosave')
        fs.rename('/documents/autosave', '/documents/working')
        if r.random() < .43: fs.sync_dir('/documents')
        fs.write('/history/revision', new)
        if r.random() < .38: fs.sync_file('/history/revision')
        fs.link('/documents/working', '/documents/autosave-backup')
        if r.random() < .15: fs.sync_dir('/documents')
    elif family == 'package_upgrade':
        for path in ('/pkg/bin/cli', '/pkg/lib/core', '/pkg/lib/helper'):
            temporary = path + '.new'
            fs.create(temporary, new)
            if r.random() < .45: fs.sync_file(temporary)
            fs.rename(temporary, path)
            if r.random() < .25: fs.sync_dir(path.rsplit('/', 1)[0])
        fs.write('/pkg/manifest/release', new)
        if r.random() < .50: fs.sync_file('/pkg/manifest/release')
        if r.random() < .25: fs.sync_dir('/pkg/manifest')
    elif family == 'mirrored_alias_rotation':
        # Both references change to the new inode in the live state, but each
        # directory may reach storage independently, splitting the aliases.
        for path in ('/active/current', '/standby/current'):
            fs.unlink(path)
            fs.link('/objects/revision-b', path)
            if r.random() < .37: fs.sync_dir(path.rsplit('/', 1)[0])
        if r.random() < .25: fs.write('/objects/revision-b', new + 1)
        if r.random() < .4: fs.sync_file('/objects/revision-b')
        fs.link('/objects/revision-a', '/objects/archived')
    elif family == 'manifest_publication':
        fs.create('/payloads/.uploading', 0)
        fs.write('/payloads/.uploading', new)
        if r.random() < .45: fs.sync_file('/payloads/.uploading')
        fs.rename('/payloads/.uploading', '/payloads/blob')
        if r.random() < .40: fs.sync_dir('/payloads')
        fs.write('/catalog/index', new)
        if r.random() < .38: fs.sync_file('/catalog/index')
        fs.unlink('/payloads/old-blob')
        if r.random() < .2: fs.sync_dir('/payloads')
    elif family == 'directory_migration':
        fs.rename('/legacy/components/a', '/new/modules/a')
        if r.random() < .36: fs.sync_dir('/new/modules')
        fs.rename('/legacy/components/b', '/new/index/b')
        if r.random() < .36: fs.sync_dir('/new/index')
        fs.rename('/legacy/components/index', '/new/modules/index')
        if r.random() < .24: fs.sync_dir('/legacy/components')
        if r.random() < .22: fs.sync_dir('/new/modules')
    else:
        for path in constraint.paths:
            fs.write(path, new)
            if r.random() < .30: fs.sync_file(path)
        # Mixed hard-link metadata not part of the invariant.
        fs.link(constraint.paths[0], '/snapshots/catalog/recovery')
        if r.random() < .35: fs.sync_dir('/snapshots/catalog')

    # Distractor operations are applied after constructing the application
    # dependency. They do not define the risk label.
    for k in range(r.randrange(3)):
        path = f'/diagnostics/{family}/log{k}'
        fs.create(path, r.randint(0, 5))
        if r.random() < .7: fs.write(path, r.randint(0, 9))
        if r.random() < .33: fs.sync_file(path)
        if r.random() < .20: fs.sync_dir(f'/diagnostics/{family}')

    if not check_constraint(fs.paths, fs.values, constraint):
        raise AssertionError(f'Invalid live state: {family} seed={seed}')
    return fs, constraint


def external_cases(per_family: int = 240, seed: int = EXTERNAL_SEED):
    """Return a fixed, explicitly labelled *external synthetic* test cohort.

    This cohort must not be consulted for epoch/architecture/parameter selection.
    Use fixed n, seed and independent scenario seeds across every model run.
    """
    return list(iter_external_cases(per_family, seed))


def iter_external_cases(per_family: int = 240, seed: int = EXTERNAL_SEED, *, selected_family=None):
    """Stream exact cohort cases, advancing the shared RNG for skipped families."""
    if per_family < 1:
        raise ValueError('At least one case per family')
    r = random.Random(seed)
    for family in EXTERNAL_FAMILIES:
        for j in range(per_family):
            run_seed = r.randrange(2**32)
            policy = POLICIES[r.randrange(len(POLICIES))]
            if selected_family is not None and family != selected_family:
                continue
            fs, constraint = make_external_workload(family, run_seed)
            risk, _ = fs.exact_failure_probability(constraint, policy)
            yield {'id': f'external/{family}/{j:04d}', 'family': family,
                   'seed': run_seed, 'sim': fs, 'constraint': constraint,
                   'writeback_probability': policy, 'risk_probability': risk}


def cohort_summary(cases):
    vals = [c['risk_probability'] for c in cases]
    return {'count':len(cases), 'families': dict(Counter(c['family'] for c in cases)),
            'zero_risk':sum(v == 0 for v in vals),
            'certain_failure':sum(v == 1 for v in vals),
            'intermediate_risk':sum(0 < v < 1 for v in vals),
            'mean_risk': round(sum(vals) / len(vals), 5) if vals else None}

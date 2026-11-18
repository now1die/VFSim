"""Workload families with real oracle-derived *probabilities* under toy writeback.

No scenario receives a label from its operation name. Ground truth is an
exhaustive, independent writeback-state enumeration; the candidate features
contain the workload and policy but NOT the realized persistence mask or label.
"""
from __future__ import annotations
import random
from collections import Counter
from challenging.simulator import DurabilityModel, Constraint, check_constraint

TRAIN_FAMILIES=('atomic_pair','write_ahead_log','staged_replace','hardlink_release','directory_publish','garbage_collect')
HELDOUT_FAMILIES=('hardlink_retarget','sharded_commit')
ALL_FAMILIES=TRAIN_FAMILIES+HELDOUT_FAMILIES


def make_workload(family: str, seed: int):
    if family not in ALL_FAMILIES: raise ValueError(f'Unknown scenario {family}')
    rng=random.Random(seed)
    s=DurabilityModel()
    base=rng.randint(1,4)
    version=base+rng.randint(1,4)
    # Distractors are synchronized *before* the transaction, then churn.
    for i in range(rng.randint(0,3)):
        s.create(f'/background/noise{i}',rng.randint(0,8))
    if family=='atomic_pair':
        paths=('/data/record','/log/record')
        for p in paths:s.create(p,base)
        c=Constraint('equal_versions',paths)
    elif family in ('write_ahead_log','staged_replace'):
        paths=('/meta/commit','/data/payload')
        s.create(paths[0],base)
        s.create(paths[1],base)
        c=Constraint('requires_data',paths)
    elif family=='hardlink_release':
        s.create('/data/blob',base)
        s.link('/data/blob','/mirror/blob')
        s.create('/meta/commit',base)
        c=Constraint('requires_data',('/meta/commit','/mirror/blob'))
    elif family=='directory_publish':
        s.create('/staging/one',base)
        s.create('/staging/two',base)
        c=Constraint('presence_together',('/release/one','/another/two'))
    elif family=='garbage_collect':
        s.create('/meta/reference',base)
        s.create('/store/blob',base)
        c=Constraint('required_if_present',('/meta/reference','/store/blob'))
    elif family=='hardlink_retarget':
        s.create('/data/old',base)
        s.create('/data/new',version)
        s.link('/data/old','/mirror/current')
        s.create('/meta/commit',base)
        c=Constraint('requires_data',('/meta/commit','/mirror/current'))
    elif family=='sharded_commit':
        n=rng.randint(3,5)
        paths=tuple(f'/shard{k}/data' for k in range(n))
        for p in paths:s.create(p,base)
        c=Constraint('equal_versions',paths)
    s.checkpoint()
    if family=='atomic_pair':
        for p in ('/data/record','/log/record'):
            s.write(p,version)
            if rng.random()<.32:s.sync_file(p)
    elif family=='write_ahead_log':
        s.write('/data/payload',version)
        if rng.random()<.43:s.sync_file('/data/payload')
        s.write('/meta/commit',version)
        if rng.random()<.33:s.sync_file('/meta/commit')
    elif family=='staged_replace':
        s.create('/stage/temp',0)
        s.write('/stage/temp',version)
        if rng.random()<.46:s.sync_file('/stage/temp')
        s.rename('/stage/temp','/data/payload')
        if rng.random()<.34:s.sync_dir('/data')
        if rng.random()<.15:s.sync_dir('/stage')
        s.write('/meta/commit',version)
        if rng.random()<.40:s.sync_file('/meta/commit')
    elif family=='hardlink_release':
        s.write('/mirror/blob',version)
        if rng.random()<.40:s.sync_file('/data/blob')
        s.write('/meta/commit',version)
        if rng.random()<.38:s.sync_file('/meta/commit')
        # Hard-link operation during the transaction. New aliases can be
        # persisted independently of data while sharing the same inode.
        s.link('/mirror/blob','/mirror/extra')
        if rng.random()<.3:s.sync_dir('/mirror')
    elif family=='directory_publish':
        s.rename('/staging/one','/release/one')
        if rng.random()<.30:s.sync_dir('/release')
        s.rename('/staging/two','/another/two')
        if rng.random()<.30:s.sync_dir('/another')
        if rng.random()<.17:s.sync_dir('/staging')
    elif family=='garbage_collect':
        s.unlink('/store/blob')
        if rng.random()<.40:s.sync_dir('/store')
        s.unlink('/meta/reference')
        if rng.random()<.40:s.sync_dir('/meta')
    elif family=='hardlink_retarget':
        s.unlink('/mirror/current')
        s.link('/data/new','/mirror/current')
        if rng.random()<.40:s.sync_dir('/mirror')
        s.write('/meta/commit',version)
        if rng.random()<.37:s.sync_file('/meta/commit')
    elif family=='sharded_commit':
        for p in c.paths:
            s.write(p,version)
            if rng.random()<.24:s.sync_file(p)
    # Benign operations are independent of the application invariant.
    noise=[p for p in s.paths if p.startswith('/background/')]
    for p in noise:
        if rng.random()<.7:
            s.write(p,rng.randint(1,10))
            if rng.random()<.3:s.sync_file(p)
    if not check_constraint(s.paths,s.values,c):
        raise AssertionError(f'Live application invariant invalid in family {family}')
    return s,c


def build_cases(n=5000,seed=2026):
    """Produce independent workbooks; *split by family/workload* before training.

    One randomized writeback model per workload, with risk probability derived
    from exact enumeration of all pending-group persistence combinations.
    """
    if n < 80: raise ValueError('Need >=80 workloads for family splits')
    rng=random.Random(seed)
    cases=[]
    for i in range(n):
        family=ALL_FAMILIES[i%len(ALL_FAMILIES)]
        scenario_seed=rng.randrange(2**32)
        sim,constraint=make_workload(family,scenario_seed)
        probability=rng.choice((0.12,0.25,0.4,0.55,0.7,0.85))
        expected,_=sim.exact_failure_probability(constraint,probability)
        cases.append({'id':f'faultlab-{i:05d}','seed':scenario_seed,
                      'family':family,'writeback_probability':probability,
                      'risk_probability':expected,'sim':sim,'constraint':constraint})
    rng.shuffle(cases)
    return cases


def split_cases(cases,seed=2026):
    """OOD families never appear during model training or selection.

    All cases are unique workbooks; extending to multiple crash masks per case
    would require grouping by case ID to prevent train/test leakage.
    """
    rng=random.Random(seed)
    train,val,id_test,ood=[],[],[],[]
    for family in TRAIN_FAMILIES:
        items=[x for x in cases if x['family']==family]
        rng.shuffle(items)
        a,b=round(.72*len(items)),round(.86*len(items))
        train.extend(items[:a]);val.extend(items[a:b]);id_test.extend(items[b:])
    ood=[x for x in cases if x['family'] in HELDOUT_FAMILIES]
    for item in (train,val,id_test,ood):rng.shuffle(item)
    if sum(map(len,(train,val,id_test,ood)))!=len(cases):raise AssertionError('Split mismatch')
    return train,val,id_test,ood


def summarize(cases):
    p=[c['risk_probability'] for c in cases]
    return {'count':len(cases), 'families':dict(Counter(c['family'] for c in cases)),
            'zero_risk':sum(x==0 for x in p), 'certain_failure':sum(x==1 for x in p),
            'intermediate_risk':sum(0<x<1 for x in p),
            'average_failure_probability':round(sum(p)/max(1,len(p)),5)}

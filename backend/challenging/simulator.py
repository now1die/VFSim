"""Small *explicit* per-object durability simulator for research experiments.

This is an application-consistency model, NOT POSIX/ext4 crash semantics.
- File-data writeback and directory-entry writeback are independent groups.
- fsync(file) persists one inode's value; fsync(directory) persists that
  directory's namespace entries. Kernel metadata barriers are not modeled.
- Each pending group independently reaches storage with probability p.
- The experiment's graph excludes the realized writeback mask.
- Enumerating masks yields an exact expected application-invariant violation
  probability under these assumptions (useful as regression ground truth).
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import PurePosixPath
from typing import Iterable
import copy


def parent(path: str) -> str:
    return str(PurePosixPath(path).parent)


@dataclass(frozen=True)
class Constraint:
    kind: str
    paths: tuple[str, ...]


def check_constraint(paths: dict[str,str], values: dict[str,int], spec: Constraint) -> bool:
    def value(path):
        inode = paths.get(path)
        return None if inode is None else values.get(inode, 0)
    vals = [value(p) for p in spec.paths]
    if spec.kind == 'equal_versions':
        return all(x is not None for x in vals) and len(set(vals)) == 1
    if spec.kind == 'requires_data':
        # A new marker may not reference data that is older or missing.
        return vals[0] is None or vals[0] == 0 or (vals[1] is not None and vals[1] >= vals[0])
    if spec.kind == 'presence_together':
        return len(set(p in paths for p in spec.paths)) == 1
    if spec.kind == 'alias_identity':
        a,b = spec.paths
        return (a not in paths and b not in paths) or (
            a in paths and b in paths and paths[a] == paths[b])
    if spec.kind == 'required_if_present':
        return spec.paths[0] not in paths or spec.paths[1] in paths
    raise ValueError(f'Unknown invariant {spec.kind}')


class DurabilityModel:
    def __init__(self):
        self.paths: dict[str,str] = {}
        self.values: dict[str,int] = {}
        self.stable_paths: dict[str,str] = {}
        self.stable_values: dict[str,int] = {}
        self.operations: list[dict] = []
        self.next_inode = 0
        self.initialized = False

    def _node(self) -> str:
        self.next_inode += 1
        return f'i{self.next_inode}'

    def _record(self, op, touched=(), **extra):
        self.operations.append({'kind': op, 'paths': list(touched), 'step': len(self.operations), **extra})

    def create(self,path,version=0):
        if path in self.paths:
            raise ValueError(f'Exists: {path}')
        inode = self._node()
        self.paths[path],self.values[inode] = inode,version
        self._record('create',(path,),inode=inode,version=version)
        return inode

    def write(self,path,version):
        inode=self.paths[path]
        self.values[inode]=int(version)
        self._record('write',(path,),inode=inode,version=version)

    def link(self,source,target):
        if target in self.paths: raise ValueError(f'Exists: {target}')
        inode=self.paths[source]
        self.paths[target]=inode
        self._record('link',(source,target),inode=inode)

    def rename(self,source,target):
        if source not in self.paths: raise ValueError(f'Missing: {source}')
        if target in self.paths: del self.paths[target]
        inode=self.paths.pop(source)
        self.paths[target]=inode
        self._record('rename',(source,target),inode=inode)

    def unlink(self,path):
        inode=self.paths.pop(path)
        self._record('unlink',(path,),inode=inode)

    def sync_file(self,path):
        inode=self.paths[path]
        self.stable_values[inode]=self.values[inode]
        self._record('fsync_file',(path,),inode=inode)

    def sync_dir(self,directory):
        self.stable_paths={p:i for p,i in self.stable_paths.items() if parent(p)!=directory}
        self.stable_paths.update({p:i for p,i in self.paths.items() if parent(p)==directory})
        self._record('fsync_dir',(directory,))

    def checkpoint(self):
        self.stable_paths=dict(self.paths)
        self.stable_values=dict(self.values)
        self.initialized=True
        self._record('durable_checkpoint',())

    def pending(self):
        all_inodes=set(self.values)|set(self.stable_values)
        files=tuple(sorted(i for i in all_inodes if self.values.get(i,0)!=self.stable_values.get(i,0)))
        dirs=set(map(parent,self.paths))|set(map(parent,self.stable_paths))
        def entries(mapping,d): return {p:i for p,i in mapping.items() if parent(p)==d}
        changed_dirs=tuple(sorted(d for d in dirs if entries(self.paths,d)!=entries(self.stable_paths,d)))
        return files,changed_dirs

    def recover(self,flushed_files: Iterable[str]=(),flushed_dirs: Iterable[str]=()):
        paths=dict(self.stable_paths)
        values=dict(self.stable_values)
        for inode in flushed_files:
            values[inode]=self.values[inode]
        for d in flushed_dirs:
            paths={p:i for p,i in paths.items() if parent(p)!=d}
            paths.update({p:i for p,i in self.paths.items() if parent(p)==d})
        # Inode with a durable name but no persisted contents recovers at
        # version zero. A real filesystem may have different allocation rules.
        for inode in paths.values():
            values.setdefault(inode,0)
        return paths,values

    def exact_failure_probability(self, constraint: Constraint, writeback_probability: float):
        if not 0 <= writeback_probability <=1: raise ValueError('Invalid persistence probability')
        files,dirs=self.pending()
        groups=[('file',x) for x in files]+[('dir',d) for d in dirs]
        if len(groups)>12: raise ValueError('Too many pending groups for exhaustive oracle')
        p=float(writeback_probability)
        risk=0.0
        failures=[]
        for switches in product((0,1),repeat=len(groups)):
            probability=1.0
            selected_files=[]
            selected_dirs=[]
            for (kind,key),on in zip(groups,switches):
                probability*=p if on else 1-p
                if on:
                    (selected_files if kind=='file' else selected_dirs).append(key)
            if probability==0: continue
            recovered=self.recover(selected_files,selected_dirs)
            violation=not check_constraint(*recovered,constraint)
            if violation:
                risk+=probability
                if len(failures)<3:
                    failures.append({'persisted_files':selected_files,'persisted_directories':selected_dirs,
                                     'recovered_paths':recovered[0], 'recovered_versions':recovered[1]})
        return round(min(1.0,max(0.0,risk)),10),failures

    def sample_recovery(self, constraint: Constraint, writeback_probability: float, rng):
        """Run one concrete randomized crash-image reconstruction.

        This is the actual simulated crash experiment behind a sampled label;
        unlike an analytic Bernoulli draw, it also returns its persisted units.
        """
        files, dirs = self.pending()
        picked_files = [i for i in files if rng.random() < writeback_probability]
        picked_dirs = [d for d in dirs if rng.random() < writeback_probability]
        recovered = self.recover(picked_files, picked_dirs)
        return {'violation': int(not check_constraint(*recovered,constraint)),
                'persisted_files': picked_files,
                'persisted_directories': picked_dirs,
                'paths': recovered[0], 'values': recovered[1]}

    def get_summary(self,constraint,writeback_probability):
        files,dirs=self.pending()
        risk,failures=self.exact_failure_probability(constraint,writeback_probability)
        return {'constraint':constraint.kind, 'paths':list(constraint.paths),
                'writeback_probability':writeback_probability,
                'risk_probability':risk,'dirty_file_groups':len(files),'dirty_directory_groups':len(dirs),
                'live_constraint_holds':check_constraint(self.paths,self.values,constraint),
                'stable_constraint_holds':check_constraint(self.stable_paths,self.stable_values,constraint),
                'example_failures':failures}

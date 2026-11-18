"""Graph input used at *pre-crash ranking time*. No oracle outcomes included."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from challenging.simulator import parent

OP_TYPES=('create','write','link','rename','unlink','fsync_file','fsync_dir','durable_checkpoint')
CONSTRAINTS=('equal_versions','requires_data','presence_together','alias_identity','required_if_present')
EDGE_TYPES=('live_path_inode','live_inode_path','stable_path_inode','stable_inode_path',
            'path_directory','directory_path','constraint_path','path_constraint',
            'operation_path','path_operation','next_op','previous_op')
F=36

@dataclass
class InputGraph:
    x:torch.Tensor
    edge_index:torch.Tensor
    edge_type:torch.Tensor
    names:list[str]
    roles:list[str]


def make_graph(case):
    sim=case['sim'];constraint=case['constraint'];p=case['writeback_probability']
    nodes=[]; names=[]; roles=[];edges=[];edge_types=[]
    def add(role,label,values):
        x=[0.0]*F
        x[{'constraint':0,'inode':1,'path':2,'directory':3,'operation':4}[role]]=1.0
        x[26]=p
        for ix,val in values.items():x[ix]=float(val)
        nodes.append(x);names.append(label);roles.append(role)
        return len(nodes)-1
    def link(a,b,relation):
        edges.append((a,b));edge_types.append(EDGE_TYPES.index(relation))
    constraint_values={20+CONSTRAINTS.index(constraint.kind):1}
    root=add('constraint',f'{constraint.kind}: {len(constraint.paths)} objects',constraint_values)
    # One inode node per identity, even when several paths are hard links.
    inode_ids=sorted(set(sim.values)|set(sim.stable_values)|set(sim.paths.values())|set(sim.stable_paths.values()))
    imap={}
    for inode in inode_ids:
        live=sim.values.get(inode,0);durable=sim.stable_values.get(inode,0)
        imap[inode]=add('inode',inode,{14:min(live,12)/12,15:min(durable,12)/12,
                                    16:(live-durable)/12,17:int(live!=durable),
                                    18:int(inode not in sim.stable_values)})
    paths=sorted(set(sim.paths)|set(sim.stable_paths)|set(p for op in sim.operations for p in op['paths'] if p.startswith('/')))
    pmap={}
    for name in paths:
        li=sim.paths.get(name); si=sim.stable_paths.get(name)
        role_index=constraint.paths.index(name)+1 if name in constraint.paths else 0
        pmap[name]=add('path',name,{19:float(li is not None),27:float(si is not None),
                                   28:float(li != si),29:float(role_index)/6,
                                   30:float(name in constraint.paths)})
        if li is not None:
            link(pmap[name],imap[li],'live_path_inode');link(imap[li],pmap[name],'live_inode_path')
        if si is not None:
            link(pmap[name],imap[si],'stable_path_inode');link(imap[si],pmap[name],'stable_inode_path')
    # Give each constraint operand an unambiguous relation to the query node.
    for name in constraint.paths:
        link(root,pmap[name],'constraint_path');link(pmap[name],root,'path_constraint')
    dirs=sorted(set(map(parent,paths))|set(p for op in sim.operations for p in op['paths'] if p.startswith('/') and p not in paths))
    dmap={}
    pending_dirs=set(sim.pending()[1])
    for name in dirs:
        dmap[name]=add('directory',name,{31:int(name in pending_dirs)})
    for path in paths:
        if parent(path) in dmap:
            link(pmap[path],dmap[parent(path)],'path_directory')
            link(dmap[parent(path)],pmap[path],'directory_path')
    previous=None
    for op in sim.operations:
        typ=op['kind']; step=op['step']
        v={'create':0,'write':1,'link':2,'rename':3,'unlink':4,'fsync_file':5,
           'fsync_dir':6,'durable_checkpoint':7}[typ]
        idx=add('operation',f'{step:02d} · {typ}',{5+v:1,32:step/max(1,len(sim.operations)),
                                               33:op.get('version',0)/12})
        if previous is not None:
            link(previous,idx,'next_op');link(idx,previous,'previous_op')
        previous=idx
        for touched in op['paths']:
            target=pmap.get(touched,dmap.get(touched))
            if target is not None:
                link(idx,target,'operation_path');link(target,idx,'path_operation')
    edge_index=torch.tensor(edges,dtype=torch.long).T.contiguous() if edges else torch.empty((2,0),dtype=torch.long)
    return InputGraph(torch.tensor(nodes,dtype=torch.float32),edge_index,
                      torch.tensor(edge_types,dtype=torch.long),names,roles)


def graph_vector(graph:InputGraph):
    """Relation-aware graph statistics for strong non-GNN baselines.

    All statistics are derived from precisely the same graph tensor as the GNN.
    Includes node-type moments, relation counts and source/target feature
    contrasts, and operand features (without recovered values or oracle).
    """
    import numpy as np
    x=graph.x.numpy();e=graph.edge_index.numpy();et=graph.edge_type.numpy()
    result=[]
    for typ in range(5):
        subset=x[x[:,typ]>.5]
        result.append(float(len(subset)))
        for reducer in ('mean','max','min'):
            result.extend((getattr(subset,reducer)(axis=0) if len(subset) else np.zeros(F)).tolist())
    for rel in range(len(EDGE_TYPES)):
        ids=np.flatnonzero(et==rel)
        result.append(float(len(ids)))
        if len(ids):
            a=x[e[0,ids]];b=x[e[1,ids]]
            result.extend(a.mean(0));result.extend(b.mean(0))
            result.extend(np.abs(a-b).mean(0))
        else:result.extend(np.zeros(F*3))
    # Explicit aggregate around constraint paths (not ordered pair shortcuts).
    relevant=np.flatnonzero(x[:,30]>.5)
    for feature in (14,15,16,17,19,27,28,29,31):
        result.append(float(x[relevant,feature].mean()) if len(relevant) else 0.)
    return np.asarray(result,dtype=np.float32)

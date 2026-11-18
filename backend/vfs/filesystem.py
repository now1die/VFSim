"""A deterministic, path-safe in-memory filesystem with operation-level crash checkpoints.

This is an educational filesystem *model*. It does not emulate ext4's block ordering,
page cache, or power-loss semantics. ``fsync`` provides a simplified durability barrier.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import uuid
from typing import Dict, List, Optional

from .models import DirEntry, FileHandle, Inode
from .transaction import Transaction, TransactionLog


class VirtualFileSystem:
    def __init__(self, checkpoint_path: Optional[str | Path] = None):
        self.inodes: Dict[str, Inode] = {}
        self.dirents: Dict[str, DirEntry] = {}
        self.file_handles: Dict[int, FileHandle] = {}
        self.next_inode_id = 1
        self.next_fd = 3
        self.rename_history: List[dict] = []
        self.risk_pairs: List[dict] = []  # application-level paired-inode declarations
        self.transaction_log = TransactionLog()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._new_inode('/', 'directory')
        self._history: List[dict] = [self.snapshot()]
        self._durable_index = 0
        if self.checkpoint_path and self.checkpoint_path.is_file():
            saved = json.loads(self.checkpoint_path.read_text(encoding='utf-8'))
            self.load_snapshot(saved)
            self._history = [self.snapshot()]

    @staticmethod
    def _path(path: str) -> str:
        if not isinstance(path, str) or not path.startswith('/') or not path:
            raise ValueError('Use an absolute POSIX path beginning with /')
        if path != '/' and (path.endswith('/') or '//' in path or any(p in ('.', '..') for p in path.split('/'))):
            raise ValueError('Invalid path: empty, dot, or parent path component')
        if '\x00' in path:
            raise ValueError('Path cannot contain NUL')
        return path

    @staticmethod
    def _parent(path: str) -> str:
        return str(PurePosixPath(path).parent)

    def _new_inode(self, path: str, inode_type: str) -> Inode:
        inode = Inode(inode_id=self.next_inode_id, path=path, inode_type=inode_type)
        self.next_inode_id += 1
        self.inodes[path] = inode
        return inode

    def _directory(self, path: str) -> Inode:
        if path not in self.inodes:
            raise ValueError('Parent directory does not exist')
        inode = self.inodes[path]
        if inode.inode_type != 'directory':
            raise ValueError('Parent is not a directory')
        return inode

    def _require_file(self, path: str) -> Inode:
        if path not in self.inodes:
            raise ValueError('File does not exist')
        inode = self.inodes[path]
        if inode.inode_type != 'file':
            raise ValueError('Operation requires a regular file')
        return inode

    def _record(self, op_type: str, ids: List[int], metadata: dict) -> Transaction:
        transaction = Transaction(
            txn_id='tx-' + uuid.uuid4().hex[:10],
            op_type=op_type,
            inodes_touched=ids,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        self.transaction_log.add(transaction)
        self._history.append(self.snapshot())
        return transaction

    def create(self, path: str, inode_type: str = 'file') -> Transaction:
        path = self._path(path)
        if inode_type not in ('file', 'directory'):
            raise ValueError('inode_type must be file or directory')
        if path in self.inodes:
            raise ValueError('Path already exists')
        parent_path = self._parent(path)
        parent = self._directory(parent_path)
        inode = self._new_inode(path, inode_type)
        inode.dirty = True
        self.dirents[path] = DirEntry(PurePosixPath(path).name, inode.inode_id, parent.inode_id)
        parent.dirty = True
        return self._record('create', [parent.inode_id, inode.inode_id],
                            {'path': path, 'inode_type': inode_type, 'parent_path': parent_path})

    def write(self, path: str, size: int) -> Transaction:
        path = self._path(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError('Write size must be a non-negative integer')
        inode = self._require_file(path)
        inode.size += size
        inode.dirty = True
        return self._record('write', [inode.inode_id], {'path': path, 'bytes_written': size})

    def link(self, existing_path: str, new_path: str) -> Transaction:
        existing_path, new_path = self._path(existing_path), self._path(new_path)
        inode = self._require_file(existing_path)
        if new_path in self.inodes:
            raise ValueError('Destination path already exists')
        parent_path = self._parent(new_path)
        parent = self._directory(parent_path)
        self.inodes[new_path] = inode  # Both pathnames now resolve to the SAME inode.
        self.dirents[new_path] = DirEntry(PurePosixPath(new_path).name, inode.inode_id, parent.inode_id)
        inode.link_count += 1
        inode.dirty = True
        parent.dirty = True
        return self._record('link', [inode.inode_id, parent.inode_id],
                            {'existing_path': existing_path, 'new_path': new_path, 'parent_path': parent_path})

    def unlink(self, path: str) -> Transaction:
        path = self._path(path)
        if path == '/':
            raise ValueError('Cannot unlink the root directory')
        if path not in self.inodes:
            raise ValueError('Path does not exist')
        inode = self.inodes[path]
        if inode.inode_type == 'directory' and any(p.startswith(path + '/') for p in self.inodes):
            raise ValueError('Cannot unlink a non-empty directory')
        parent = self.inodes[self._parent(path)]
        del self.inodes[path]
        del self.dirents[path]
        inode.link_count -= 1
        inode.dirty = True
        parent.dirty = True
        if inode.path == path and inode.link_count:
            inode.path = next(p for p, node in self.inodes.items() if node is inode)
        return self._record('unlink', [inode.inode_id, parent.inode_id],
                            {'path': path, 'remaining_link_count': inode.link_count})

    def rename(self, old_path: str, new_path: str) -> Transaction:
        old_path, new_path = self._path(old_path), self._path(new_path)
        if old_path == '/':
            raise ValueError('Cannot rename the root directory')
        if old_path not in self.inodes:
            raise ValueError('Source path does not exist')
        if new_path in self.inodes:
            raise ValueError('Destination path already exists')
        inode = self.inodes[old_path]
        if inode.inode_type == 'directory' and new_path.startswith(old_path + '/'):
            raise ValueError('Cannot move a directory into its descendant')
        parent = self._directory(self._parent(new_path))
        old_parent = self.inodes[self._parent(old_path)]
        # Moving a directory also moves its entire subtree; hard links retain their identity.
        moved = [p for p in self.inodes if p == old_path or
                 (inode.inode_type == 'directory' and p.startswith(old_path + '/'))]
        new_paths = {p: new_path + p[len(old_path):] for p in moved}
        if any(dest in self.inodes and dest not in moved for dest in new_paths.values()):
            raise ValueError('Destination subtree conflicts with existing files')
        entries = [(p, self.inodes[p], self.dirents.get(p)) for p in moved]
        for p, _, _ in entries:
            self.inodes.pop(p)
            self.dirents.pop(p, None)
        for p, child, entry in entries:
            dest = new_paths[p]
            self.inodes[dest] = child
            if child.path == p:
                child.path = dest
            if entry:
                entry.name = PurePosixPath(dest).name
                entry.parent_inode_id = self.inodes[self._parent(dest)].inode_id
                self.dirents[dest] = entry
            child.dirty = True
        old_parent.dirty = True
        parent.dirty = True
        self.rename_history.append({'inode_id': inode.inode_id, 'old_path': old_path, 'new_path': new_path})
        return self._record('rename', [inode.inode_id, old_parent.inode_id, parent.inode_id],
                            {'old_path': old_path, 'new_path': new_path, 'new_parent_path': self._parent(new_path)})

    def open(self, path: str, mode: str = 'r') -> Transaction:
        path = self._path(path)
        if mode not in ('r', 'w', 'a', 'r+', 'w+', 'a+'):
            raise ValueError('Unsupported file mode')
        inode = self._require_file(path)
        fd = self.next_fd
        self.next_fd += 1
        self.file_handles[fd] = FileHandle(fd, inode.inode_id, path, mode)
        return self._record('open', [inode.inode_id], {'fd': fd, 'path': path, 'mode': mode})

    def close(self, fd: int) -> Transaction:
        if fd not in self.file_handles:
            raise ValueError('File descriptor is not open')
        handle = self.file_handles.pop(fd)
        return self._record('close', [handle.inode_id], {'fd': fd, 'path': handle.path})

    def fsync(self, path: str = '/') -> Transaction:
        """Model-level durability barrier: persist the entire namespace at this point."""
        path = self._path(path)
        if path not in self.inodes:
            raise ValueError('Path does not exist')
        for inode in self.get_all_inodes():
            inode.dirty = False
        tx = self._record('fsync', [self.inodes[path].inode_id], {'path': path})
        self._durable_index = len(self.transaction_log.transactions)
        if self.checkpoint_path:
            self._persist_checkpoint(self.snapshot())
        return tx

    def _persist_checkpoint(self, state: dict) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.checkpoint_path.with_suffix('.tmp')
        with temp.open('w', encoding='utf-8') as file:
            json.dump(state, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, self.checkpoint_path)

    def declare_pair(self, first: str, second: str, kind: str = 'application_pair') -> dict:
        """Register a semantic dependency for graph features; not a filesystem operation.

        Declaration is application intent, not evidence of a crash violation. It
        is never used to assign labels directly: the recovery oracle determines
        whether a specific invariant actually holds after restoring a snapshot.
        """
        first, second = self._path(first), self._path(second)
        if first not in self.inodes or second not in self.inodes:
            raise ValueError('Both paths must exist before declaring an inode pair')
        if first == second or self.inodes[first] is self.inodes[second]:
            raise ValueError('Pair must contain two distinct inodes')
        pair = {'inode_a': self.inodes[first].inode_id, 'inode_b': self.inodes[second].inode_id,
                'path_a': first, 'path_b': second, 'kind': kind}
        if pair not in self.risk_pairs:
            self.risk_pairs.append(pair)
            self._history[-1] = self.snapshot()
        return pair

    def snapshot(self) -> dict:
        unique = {inode.inode_id: inode for inode in self.inodes.values()}
        return {
            'inodes': {str(i): {'inode_id': i, 'path': n.path, 'inode_type': n.inode_type,
                               'size': n.size, 'link_count': n.link_count, 'dirty': n.dirty}
                       for i, n in unique.items()},
            'paths': {p: n.inode_id for p, n in self.inodes.items()},
            'next_inode_id': self.next_inode_id,
            'rename_history': deepcopy(self.rename_history),
            'risk_pairs': deepcopy(self.risk_pairs),
        }

    def load_snapshot(self, snapshot: dict) -> None:
        nodes = {int(k): Inode(**v) for k, v in snapshot['inodes'].items()}
        self.inodes = {p: nodes[int(i)] for p, i in snapshot['paths'].items()}
        self.next_inode_id = snapshot['next_inode_id']
        self.rename_history = deepcopy(snapshot.get('rename_history', []))
        self.risk_pairs = deepcopy(snapshot.get('risk_pairs', []))
        self.dirents = {
            p: DirEntry(PurePosixPath(p).name, inode.inode_id,
                        self.inodes[self._parent(p)].inode_id)
            for p, inode in self.inodes.items() if p != '/'
        }
        self.file_handles.clear()
        self.next_fd = 3
        self.validate()

    def replay_at(self, operation_index: int) -> "VirtualFileSystem":
        """Read-only reconstruction retaining the *historical durability boundary*.

        Restoring a snapshot alone resets the durability history; a risk model
        trained on last-fsync features would then see the wrong input graph.
        """
        if not 0 <= operation_index <= len(self.transaction_log.transactions):
            raise ValueError('Operation index outside transaction history')
        replay = VirtualFileSystem()
        replay.load_snapshot(self._history[operation_index])
        replay._history = deepcopy(self._history[:operation_index + 1])
        replay._durable_index = max(
            (i for i in range(1, operation_index + 1)
             if self.transaction_log.transactions[i - 1].op_type == 'fsync'),
            default=0)
        return replay

    def validate(self) -> bool:
        if '/' not in self.inodes or self.inodes['/'].inode_type != 'directory':
            raise ValueError('Root directory is missing')
        if set(self.dirents) != set(self.inodes) - {'/'}:
            raise ValueError('Directory entries do not match path mapping')
        for path, inode in self.inodes.items():
            if path == '/':
                continue
            parent = self.inodes.get(self._parent(path))
            entry = self.dirents[path]
            if not parent or parent.inode_type != 'directory':
                raise ValueError(f'Invalid parent for {path}')
            if entry.parent_inode_id != parent.inode_id or entry.inode_id != inode.inode_id:
                raise ValueError(f'Directory entry mismatch at {path}')
        for inode in self.get_all_inodes():
            if inode.link_count != sum(other is inode for other in self.inodes.values()):
                raise ValueError(f'Incorrect link count for inode {inode.inode_id}')
        return True

    def crash_preview(self, transaction_id: str, scenario_id: str) -> dict:
        """Reconstruct and validate a modeled crash state WITHOUT changing the live VFS."""
        all_tx = self.get_transactions()
        index = next((i + 1 for i, tx in enumerate(all_tx) if tx.txn_id == transaction_id), None)
        if index is None:
            raise ValueError('Transaction not found')
        if scenario_id == 'crash-before-commit':
            cut = index - 1
        elif scenario_id == 'crash-after-commit':
            cut = index
        elif scenario_id == 'crash-last-sync':
            cut = max((i for i in range(index + 1) if i == 0 or
                       all_tx[i - 1].op_type == 'fsync'), default=0)
        else:
            raise ValueError('Unknown crash scenario')
        before = self._history[index]
        recovered = self._history[cut]
        checker = VirtualFileSystem()
        checker.load_snapshot(recovered)
        # These are *application* declarations, separate from structural VFS
        # integrity. An internally valid recovered filesystem may still violate
        # an application's two-file consistency requirement.
        violated_pairs = []
        declared = [pair for pair in self.risk_pairs
                    if pair.get('kind') in ('size_equal', 'presence_equal')]
        for pair in declared:
            first, second = pair['path_a'], pair['path_b']
            p, q = first in recovered['paths'], second in recovered['paths']
            if pair['kind'] == 'presence_equal':
                valid = p == q
            else:
                valid = p and q and (
                    recovered['inodes'][str(recovered['paths'][first])]['size'] ==
                    recovered['inodes'][str(recovered['paths'][second])]['size'])
            if not valid:
                violated_pairs.append({'paths': [first, second], 'kind': pair['kind']})
        before_paths, after_paths = set(before['paths']), set(recovered['paths'])
        lost_paths = sorted(before_paths - after_paths)
        restored_paths = sorted(after_paths - before_paths)
        changed_sizes = sorted(path for path in before_paths & after_paths if
                               before['inodes'][str(before['paths'][path])]['size'] !=
                               recovered['inodes'][str(recovered['paths'][path])]['size'])
        return {
            'scenarioId': scenario_id,
            'transactionId': transaction_id,
            'survivedOps': cut,
            'lostOps': index - cut,
            'selectedOperationSurvived': cut >= index,
            'recoveredPaths': sorted(after_paths),
            'lostPaths': lost_paths,
            'restoredPaths': restored_paths,
            'changedSizePaths': changed_sizes,
            'consistencyPassed': checker.validate(),
            'applicationInvariantsPassed': not violated_pairs if declared else None,
            'declaredInvariantCount': len(declared),
            'violatedPairs': violated_pairs,
            'recoverySteps': [
                f'Choose operation boundary {cut} of {index}',
                'Restore the saved namespace/inode state at that boundary',
                f'Validate {len(after_paths)} recovered paths and hard-link counts',
                f'Check differences: {len(lost_paths)} missing, {len(restored_paths)} restored, {len(changed_sizes)} changed-size paths',
                'Live filesystem was not modified; this is a simplified simulated crash',
            ],
            'postRecoveryRisk': None,
        }

    def get_inode(self, path: str):
        return self.inodes.get(path)

    def get_all_inodes(self):
        return list({node.inode_id: node for node in self.inodes.values()}.values())

    def get_all_dirents(self):
        return list(self.dirents.values())

    def get_rename_history(self):
        return self.rename_history

    def get_transactions(self):
        return self.transaction_log.get_all()

from pathlib import Path

import torch

from vfs.filesystem import VirtualFileSystem
from graph.builder import build_graph
from graph.subgraph import (
    get_touched_inode_indices,
    extract_2hop_subgraph,
    get_subgraph_touched_inode_indices,
)
from gnn.model import RiskGNN


class RiskGNNInference:
    """Loads the trained RiskGNN checkpoint and predicts transaction risk."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.load_error = None

    @property
    def is_loaded(self):
        return self.model is not None

    def load(self):
        self.model = None
        self.load_error = None

        try:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}"
                )

            # Use the same graph preparation process as training.
            probe_vfs = VirtualFileSystem()
            probe_vfs.create("/inference_probe.txt", "file")

            probe_transaction = probe_vfs.get_transactions()[-1]
            probe_graph = build_graph(probe_vfs)

            touched_indices = get_touched_inode_indices(
                probe_vfs,
                probe_transaction,
            )

            probe_subgraph = extract_2hop_subgraph(
                probe_graph,
                touched_indices,
            )

            subgraph_touched = get_subgraph_touched_inode_indices(
                probe_graph,
                touched_indices,
            )

            if not subgraph_touched:
                raise RuntimeError(
                    "Could not initialize model using the probe graph."
                )

            model = RiskGNN()

            # Initialize LazyLinear layers before loading checkpoint weights.
            with torch.no_grad():
                model(
                    probe_subgraph.x_dict,
                    probe_subgraph.edge_index_dict,
                    subgraph_touched,
                )

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(checkpoint, dict):
                raise ValueError("Unexpected checkpoint format.")

            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    "Checkpoint does not contain model_state_dict. "
                    f"Available keys: {list(checkpoint.keys())}"
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.model = model
            return True

        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.model = None
            raise

    def predict(self, vfs, transaction):
        if self.model is None:
            raise RuntimeError("RiskGNN model has not been loaded.")

        # Build the graph from the current VFS state.
        graph = build_graph(vfs)

        touched_indices = get_touched_inode_indices(
            vfs,
            transaction,
        )

        if not touched_indices:
            raise ValueError(
                f"No touched inode indices for {transaction.txn_id}."
            )

        # Use exactly the same 2-hop extraction as the training dataset.
        subgraph = extract_2hop_subgraph(
            graph,
            touched_indices,
        )

        subgraph_touched = get_subgraph_touched_inode_indices(
            graph,
            touched_indices,
        )

        if not subgraph_touched:
            raise ValueError(
                f"No touched inodes in subgraph for {transaction.txn_id}."
            )

        with torch.inference_mode():
            prediction = self.model.predict_risk(
                subgraph.x_dict,
                subgraph.edge_index_dict,
                subgraph_touched,
            )

        score = float(prediction.reshape(-1)[0].item())

        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid prediction score: {score}")

        return score

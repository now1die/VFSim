# Archived PyTorch Geometric RiskGNN (original Google Drive version)

This directory contains the *original* heterogeneous PyTorch Geometric architecture and its checkpoint, not a replacement for VFSim v1.0's validated Torch-native inference. It is retained for potential future ablation or fair same-task evaluation.

- `model.py`: original 2-layer HeteroConv/GATConv architecture.
- `riskgnn_original.pt`: original Google Drive checkpoint (not verified in this environment).
- The separate outdated `reference_backend/` copy is intentionally not shipped: it had a second `main.py` that confused clean installation. The original reference architecture and checkpoint remain available in this directory.
- `requirements-optional.txt`: extra dependency needed for PyG experiments.

**Do not import this package into the live backend or compare its old synthetic 0/1 labels directly against FaultLab's failure probability.** They are different prediction targets. A fair comparison must retrain both architectures on identical splits, use identical features/labels, and report seed variability. PyG was not installed here, so this optional model was not run or evaluated.

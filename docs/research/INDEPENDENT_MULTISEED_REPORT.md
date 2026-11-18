# VFSim v0.9 — Independent Synthetic Families & Five-Seed Replication

## Executive assessment
**Completed:** New scenario templates, a frozen evaluation cohort, five independently trained GNN checkpoints, seed-matched baseline comparisons, model-faithful attention traces, API integration, and browser screenshots.
**Not completed by agreement:** Linux/ext4/VM crash validation. Nothing in this report is evidence of ext4 failures.

### Scientific interpretation
The GNN outperformed the tabular baselines tested here on both probability MAE and top-10% risk-ranking for the six new synthetic workload templates in all five training runs. This is a meaningful within-simulator repeatability result, but **not proof of superior real-world crash-risk detection**. The oracle uses independent per-object writeback and synthetic application invariants; no Linux journal, block ordering, delayed allocation, actual SQLite, or actual package manager were executed.

## Design frozen for evaluation
- **Training generator:** Original six training families; old hard-link-retarget and sharded-commit families remain development diagnostics, not the pristine independent evaluation.
- **New external templates (never used for fitting or selecting checkpoints):** editor autosave; package upgrade; mirrored alias rotation; manifest publication; directory migration; multi-file snapshot batch.
- **Training seeds:** 11, 29, 47, 83, 131. Initial weights, training-case generation, and shuffling vary by seed.
- **Workloads per seed:** 5,000 generated, divided as 2,700 training / 528 validation / 522 seen-family test / 1,250 previously held-out-family diagnostics. Five maximum epochs; the best checkpoint is chosen using only the in-family validation MAE.
- **External synthetic cohort:** 1,440 distinct workloads (240 per family), fixed external seed 20261116, **exactly the same cases for every seed and baseline**.
- **Regression target:** Exact probability of violating a declared application invariant under this simulator’s enumerated writeback masks, *not* observed Linux crash probability.
- **Ranking budget:** 144 crash candidates (10% of 1,440), prioritized by predicted risk. Expected failures are sums of oracle probabilities over selected candidates, not measured numbers of Linux bugs.
- **Model baselines:** Random Forest, histogram gradient boosting, XGBoost, Ridge on graph statistics, and an explicit unsynced-group heuristic. Tabular baselines receive summary statistics computed from the same input graph but not raw graph-message passing.
- **Original systems preserved:** v0.7 and v0.8 checkpoints, original filesystem GNN, all three visualization views, hard links, and existing FaultLab analyses.

## Results — external synthetic cohort, mean ± sample SD over five independent training seeds

| Model | MAE to oracle (↓) | Expected violations in first 144 tests (↑) |
|---|---:|---:|
| GNN v0.8 operand-aware | 0.1237 ± 0.0150 | 133.24 ± 2.56 |
| Random Forest | 0.2087 ± 0.0100 | 114.90 ± 4.56 |
| HistGradientBoosting | 0.2045 ± 0.0032 | 111.23 ± 5.24 |
| XGBoost | 0.2016 ± 0.0105 | 111.52 ± 6.03 |
| Ridge (graph statistics) | 0.3491 ± 0.0072 | 69.17 ± 1.38 |
| Pending-writeback heuristic | 0.4298 ± 0.0000 | 51.54 ± 0.00 |

**Oracle-best ranking ceiling (not deployable):** 142.06 expected violations among 144 tests.

### Paired improvements, seed by seed

The paired deltas below use the same fixed evaluation set within each seed. Positive differences mean the GNN has lower MAE or more expected failures found at the 10% budget.

| Compared with | Mean MAE reduction by GNN | Mean additional expected violations @144 |
|---|---:|---:|
| Random Forest | +0.0850 | +18.34 |
| HistGradientBoosting | +0.0808 | +22.01 |
| XGBoost | +0.0780 | +21.72 |
| Ridge (graph statistics) | +0.2255 | +64.07 |
| Pending-writeback heuristic | +0.3061 | +81.70 |

### Each GNN run (no seed omitted)

| Training seed | Best epoch by validation | External MAE | Expected violations @144 |
|---:|---:|---:|---:|
| 11 | 5 | 0.13856 | 129.65 |
| 29 | 3 | 0.10355 | 134.08 |
| 47 | 5 | 0.11283 | 136.38 |
| 83 | 5 | 0.13496 | 131.87 |
| 131 | 5 | 0.12846 | 134.22 |

## Failure-mode audit: external-family MAE

| External family | GNN MAE | Random Forest MAE | XGBoost MAE |
|---|---:|---:|---:|
| editor_autosave | 0.0544 | 0.1788 | 0.1634 |
| package_upgrade | 0.2000 | 0.2989 | 0.2527 |
| mirrored_alias_rotation | 0.1711 | 0.1937 | 0.1694 |
| manifest_publication | 0.0513 | 0.1421 | 0.1526 |
| directory_migration | 0.0266 | 0.1105 | 0.1531 |
| snapshot_batch | 0.2387 | 0.3279 | 0.3186 |

**Difficult scenarios remain:** multi-file package upgrades and snapshot batches have the largest GNN errors. Mirrored alias rotation varies materially between seeds; XGBoost has a slightly lower *family-specific* average MAE than the GNN there. That exception is preserved instead of hidden.

## Dataset and validation integrity

- The six external families are defined in `backend/challenging/independent.py`, separate from `backend/challenging/generator.py`. The training generator does not import or sample them.
- The final external cases were frozen and kept constant across training seeds and baselines. Labels are recomputed from an exact independent-writeback-state oracle, and the graph never contains oracle risk, realized persistence masks, or sampled recovered outcomes.
- `backend/artifacts/independent_v09/external_cohort_manifest.json` stores cohort SHA-256 and independent generator file SHA-256. Per-seed predictions and checkpoints are saved individually.
- Checkpoint choice uses *training-family validation* only. The newly created external cohort was not used to select epochs, architecture, baselines, or hyperparameters in this run.
- The repeated-seed SD is **descriptive** across only five training runs. These seeds are not independent production environments, and this is not a confidence interval for real-world performance.
- The v0.8 operand-aware architecture was designed earlier using older held-out family diagnostics; this new cohort is the attempt to evaluate that chosen architecture on fresh templates.

### Reproducibility fingerprint

- External cohort SHA-256: `1b738e13bba0236492b970f04120f09dc7790014ad52918e2ec4928d75fc8a27`
- External generator SHA-256: `8c1ad704af9b58c69cf930f720b4152678f1f643bec81b16b3e9977d779e80cd`
- External cohort seed: `20261116`
- External risk distribution: 157 zero-risk, 1283 intermediate-risk, 0 certain-failure.

## Reproduce locally

From the extracted ZIP:

```powershell
cd VFSim\backend
python -m pip install -r requirements-research.txt
python -m pytest tests -q
python -m challenging.multiseed --seeds 11 29 47 83 131 --samples 5000 --epochs 5 --per-family 240 --outdir artifacts/reproduction_v09
python -m uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/research`, navigate to **Transaction & risk**, and view the **v0.9 independent workload family test** panel. In **GNN input graph**, choose any `v0.9 seed` checkpoint, then a new workload family, to inspect the actual attention coefficients, 32D node embeddings, model output, and separately computed oracle risk.

The included ready-made experiment artifacts can be explored without retraining. The above reproduction command intentionally writes to a separate subdirectory and does not overwrite saved v0.7/v0.8/v0.9 checkpoints.

## What v0.9 does *not* establish

1. No real ext4 filesystem, Linux VM, or actual crash-image recovery was evaluated here.
2. These workflows resemble application patterns, but are not empirically traced implementations of an editor, package manager, database, or backup software.
3. Writebacks are independently sampled by unit with one global persistence probability; real journaling and storage ordering are much more complex.
4. A GNN win against these particular baselines does not prove that the GNN is necessary: stronger graph-feature engineering, other GNN architectures, and better-tuned baselines remain open.
5. Five seeds are useful for stability, not a universal statistical guarantee. Report seed-specific variability and family-specific failures.

## Next milestone (intentionally deferred)

Create a separate Linux VM harness using disposable ext4-backed virtual disks, execute equivalent file workloads, inject crash boundaries without pretending normal shutdown is power loss, validate recovered application invariants, and compare *measured ext4* outcomes with simulator oracle and predicted rankings. That work requires an appropriate Linux/VM test environment and was not performed in this release.

---

**Research status:** Improved, reproducible synthetic evidence. Not yet a publication-ready claim about real filesystems.

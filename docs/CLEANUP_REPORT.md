# Clean-up report

Source: VFSim v1.7.1 full project archive. This is a **non-destructive** cleanup: the original archive was not modified.

## Removed

- 38 shadcn UI component files that are not imported by the live application.
- Duplicate frontend routing test and test setup under `src/test/test`.
- Unused mobile hook associated with the removed sidebar scaffold.
- Outdated version-by-version merge notes, historical visualizer instructions, stale checksum list and generic route documentation.

## Retained intentionally

- Live React/TanStack UI, styling, sprites/SVG and working source routes.
- FastAPI backend, filesystem and graph APIs, GNN checkpoints, FaultLab experiments, model metrics, and 102 backend regression tests.
- Optional archived PyTorch Geometric model (as an explicitly non-live research reference).
- `package-lock.json`, even though some scaffold dependencies may no longer be imported, to avoid silently altering working dependency resolution without a verified npm build.
- Third-party notices and the original independent-workload research evidence, organized under `docs/research`.

## Verification

The original clean archive omitted screenshots and had no verified frontend build. The GitHub preparation runs the backend checks, frontend checks and production build, and documents a browser walkthrough. [Screenshot gallery](SCREENSHOTS.md)

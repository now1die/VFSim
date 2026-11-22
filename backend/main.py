from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
import logging
import os
import sqlite3
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sessions import SessionStore, session_vfs
from vfs.filesystem import VirtualFileSystem


logger = logging.getLogger(__name__)
CHECKPOINT_PATH = Path(__file__).resolve().parent / "artifacts" / "riskgnn.pt"
vfs = VirtualFileSystem(checkpoint_path=os.environ.get("VFSIM_CHECKPOINT_PATH"))
production = os.environ.get("VFSIM_ENV") == "production"


def current_vfs() -> VirtualFileSystem:
    return session_vfs.get() or vfs


@asynccontextmanager
async def lifespan(app: FastAPI):
    proxy_token = os.environ.get("VFSIM_PROXY_TOKEN")
    if production and not proxy_token:
        raise RuntimeError("VFSIM_PROXY_TOKEN is required for hosted sessions")

    with ExitStack() as resources:
        app.state.sessions = None
        if proxy_token:
            directory = resources.enter_context(TemporaryDirectory(prefix="vfsim-"))
            app.state.sessions = SessionStore(Path(directory) / "sessions.sqlite3", proxy_token)
            app.state.sessions.initialize()
        try:
            if _trained_model() is None:
                raise RuntimeError("The main GNN checkpoint is missing or incompatible")
            from challenging.trace import V09_SEEDS, _checkpoint_files, model_and_metadata
            for version in ("v07", "v08", *(f"v09_s{seed}" for seed in V09_SEEDS)):
                model_and_metadata(version)
                if not _checkpoint_files(version)[1].is_file():
                    raise RuntimeError(f"FaultLab examples are missing for {version}")
            for artifact in (
                "baseline_metrics.json", "hardmode_metrics.json", "readout_comparison.json",
                "hardmode_audit_v07.json", "hardmode_audit.json",
                "independent_v09/external_multiseed_results.json",
            ):
                if not (CHECKPOINT_PATH.parent / artifact).is_file():
                    raise RuntimeError("A required VFSim research artifact is missing")
        except Exception:
            logger.exception("VFSim model or research artifacts unavailable")
            if production:
                raise
            app.state.models_ready = False
        else:
            app.state.models_ready = True
        try:
            yield
        finally:
            app.state.sessions = None
            del app.state.models_ready


app = FastAPI(
    title="VFSim Backend",
    description="Backend API for the VFSim risk-aware filesystem simulator",
    version="0.9.0",
    lifespan=lifespan,
    docs_url=None if production else "/docs",
    redoc_url=None if production else "/redoc",
    openapi_url=None if production else "/openapi.json",
)


@app.middleware("http")
async def scope_api(request: Request, call_next):
    store = getattr(request.app.state, "sessions", None)
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        if store is not None:
            return await store.handle(request, call_next)
        if production:
            return JSONResponse(
                {"detail": "Simulator storage is unavailable."},
                status_code=503,
                headers={"cache-control": "no-store"},
            )
    return await call_next(request)


class CreateRequest(BaseModel):
    path: str = Field(max_length=512)
    inode_type: str = Field(default="file", max_length=16)


class WriteRequest(BaseModel):
    path: str = Field(max_length=512)
    size: int = Field(le=2**53 - 1)


class PairRequest(BaseModel):
    first: str = Field(max_length=512)
    second: str = Field(max_length=512)
    kind: str = Field(default="size_equal", max_length=64)


class LinkRequest(BaseModel):
    existing_path: str = Field(max_length=512)
    new_path: str = Field(max_length=512)


class RenameRequest(BaseModel):
    old_path: str = Field(max_length=512)
    new_path: str = Field(max_length=512)


class UnlinkRequest(BaseModel):
    path: str = Field(max_length=512)


class OpenRequest(BaseModel):
    path: str = Field(max_length=512)
    mode: str = Field(default="r", max_length=8)


class CloseRequest(BaseModel):
    fd: int


class SyncRequest(BaseModel):
    path: str = Field(default="/", max_length=512)


class CrashRequest(BaseModel):
    transactionId: str = Field(max_length=64)
    scenarioId: str = Field(max_length=64)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in os.environ.get(
            "VFSIM_CORS_ORIGINS",
            "" if production else "http://127.0.0.1:3000,http://localhost:3000",
        ).split(",") if origin.strip() and origin.strip() != "*"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


def _risk_level(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.75:
        return "critical"
    if score >= 0.50:
        return "elevated"
    if score >= 0.25:
        return "watch"
    return "safe"


def _transaction_label(tx) -> str:
    path = (
        tx.metadata.get("path")
        or tx.metadata.get("new_path")
        or tx.metadata.get("old_path")
        or tx.metadata.get("existing_path")
        or ""
    )
    return f"{tx.op_type} {path}".strip()


def _bytes_for_transaction(tx) -> int:
    return int(tx.metadata.get("bytes_written", 0) or 0)


def _frontend_transaction(tx) -> Dict[str, Any]:
    bytes_written = _bytes_for_transaction(tx)
    operation = tx.op_type

    # Only a model-level fsync marks the current namespace as durable.
    op = {
        "id": f"{tx.txn_id}-op-1",
        "seq": 1,
        "op": operation,
        "target": (
            tx.metadata.get("path")
            or tx.metadata.get("new_path")
            or tx.metadata.get("old_path")
            or ""
        ),
        "bytes": bytes_written,
        "journaled": tx.op_type == "fsync",
        "riskDelta": 0.0,
    }

    assessment = _score_transaction(tx)
    return {
        "id": tx.txn_id,
        "label": _transaction_label(tx),
        "status": "committed",
        "startedAt": tx.timestamp.replace("+00:00", "Z"),
        "durationMs": 0,
        "operations": [op],
        "riskScore": assessment["score"],
        "riskLevel": _risk_level(assessment["score"]),
    }


def _find_transaction(transaction_id: str):
    tx = current_vfs().transaction_log.get_by_id(transaction_id)
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return tx


@lru_cache(maxsize=1)
def _trained_model():
    if not CHECKPOINT_PATH.exists():
        return None
    from gnn.model import RiskGNN
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    net = RiskGNN()
    try:
        net.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        # Prevent HTTP 500s if a user upgrades code before retraining.
        return None
    net.eval()
    return (net, checkpoint.get("metadata", {}))


def _score_transaction(tx):
    result = {"score": None, "confidence": None, "modelLoaded": _model_loaded(), "explanations": []}
    trained = _trained_model()
    if trained is None:
        return result
    from graph.builder import build_graph
    from graph.subgraph import get_touched_inode_indices, extract_2hop_subgraph, get_subgraph_touched_inode_indices
    idx = next((i+1 for i, t in enumerate(current_vfs().get_transactions()) if t.txn_id == tx.txn_id), None)
    if idx is None:
        return result
    replay = current_vfs().replay_at(idx)
    touched = get_touched_inode_indices(replay, tx)
    if not touched:
        return result
    graph = build_graph(replay)
    local = extract_2hop_subgraph(graph, touched)
    remapped = get_subgraph_touched_inode_indices(graph, touched)
    result["score"] = round(float(trained[0].predict_risk(local.x_dict, local.edge_index_dict, remapped).item()), 4)
    # No calibrated confidence or causal explanation: do not fabricate either.
    return result


def _graph_payload(transaction_id: Optional[str] = None) -> Dict[str, Any]:
    inodes = current_vfs().get_all_inodes()
    dirents = current_vfs().get_all_dirents()

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    inode_ids = {inode.inode_id: f"inode-{inode.inode_id}" for inode in inodes}
    latest_touch = {}
    for tx in current_vfs().get_transactions():
        for inode_id in tx.inodes_touched:
            latest_touch[inode_id] = tx.timestamp

    for inode in inodes:
        nodes.append(
            {
                "id": inode_ids[inode.inode_id],
                "name": inode.path.rsplit("/", 1)[-1] or "/",
                "path": inode.path,
                "kind": "directory" if inode.inode_type == "directory" else "file",
                "sizeKb": round(inode.size / 1024, 3),
                "riskScore": None,
                "riskLevel": "unknown",
                "lastWrite": latest_touch.get(inode.inode_id, ""),
                "dirty": inode.dirty,
                "embedding": [],
                "inodeNumber": inode.inode_id,
                "linkCount": inode.link_count,
            }
        )

    dirent_lookup = {id(entry): path for path, entry in current_vfs().dirents.items()}
    dirent_node_by_path = {}
    for i, dirent in enumerate(dirents):
        dirent_id = f"dirent-{i}"
        real_path = dirent_lookup.get(id(dirent), dirent.name)
        dirent_node_by_path[real_path] = dirent_id
        parent_id = inode_ids.get(dirent.parent_inode_id)
        inode_id = inode_ids.get(dirent.inode_id)
        nodes.append(
            {
                "id": dirent_id,
                "name": dirent.name,
                "path": real_path,
                "kind": "inode",
                "sizeKb": 0,
                "riskScore": None,
                "riskLevel": "unknown",
                "lastWrite": "",
                "dirty": False,
                "embedding": [],
                "inodeNumber": dirent.inode_id,
                "parentInodeNumber": dirent.parent_inode_id,
            }
        )
        if parent_id:
            edges.append(
                {
                    "id": f"{parent_id}->{dirent_id}",
                    "source": parent_id,
                    "target": dirent_id,
                    "relation": "contains",
                    "weight": 1.0,
                }
            )
        if inode_id:
            edges.append(
                {
                    "id": f"{dirent_id}->{inode_id}",
                    "source": dirent_id,
                    "target": inode_id,
                    "relation": "points-to",
                    "weight": 1.0,
                }
            )

    # Represent hard links as direct frontend edges between paths that share
    # the same inode. This is only a visualization; the real heterogeneous
    # graph remains in backend/graph/builder.py.
    paths_by_inode: Dict[int, List[str]] = {}
    for path, dirent in current_vfs().dirents.items():
        paths_by_inode.setdefault(dirent.inode_id, []).append(path)

    for inode_id, paths in paths_by_inode.items():
        if len(paths) > 1:
            base = paths[0]
            for other in paths[1:]:
                a = f"inode-{inode_id}"
                b = dirent_node_by_path.get(other)
                if b:
                    edges.append(
                        {
                            "id": f"{a}->{b}-hardlink",
                            "source": a,
                            "target": b,
                            "relation": "hardlink",
                            "weight": 1.0,
                        }
                    )

    touched: List[str] = []
    if transaction_id:
        tx = _find_transaction(transaction_id)
        touched = [
            inode_ids[i]
            for i in tx.inodes_touched
            if i in inode_ids
        ]

    return {
        "nodes": nodes,
        "edges": edges,
        "touched": touched,
        "transactionId": transaction_id,
    }


def _model_loaded() -> bool:
    return _trained_model() is not None


@app.get("/")
def root():
    return {
        "message": "VFSim backend is running",
        "version": app.version,
    }


@app.get("/health")
def liveness():
    return {"status": "ok"}


@app.get("/ready")
@app.get("/api/health")
def health(response: Response):
    models_ready = getattr(app.state, "models_ready", None)
    if models_ready is None:
        models_ready = not production and _model_loaded()
    store = getattr(app.state, "sessions", None)
    storage_ready = not production or store is not None
    try:
        if store:
            storage_ready = store.ready()
    except (OSError, sqlite3.Error):
        logger.exception("VFSim readiness storage check failed")
        storage_ready = False
    ready = models_ready and storage_ready
    if not ready:
        response.status_code = 503
    response.headers["cache-control"] = "no-store"
    return {
        "status": "ok" if ready else "unavailable",
        "mode": "live",
        "version": app.version,
        "gnn": "loaded" if models_ready else "not_loaded",
        "storage": "temporary" if store and storage_ready else "local" if storage_ready else "unavailable",
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/session", status_code=204)
def local_session():
    return Response(status_code=204, headers={"cache-control": "no-store"})


@app.get("/api/status")
def status():
    return {
        "backend": "online",
        "mode": "live",
        "gnn": "loaded" if _model_loaded() else "not_loaded",
    }


@app.get("/api/stats")
def stats():
    inodes = current_vfs().get_all_inodes()
    dirents = current_vfs().get_all_dirents()

    # The count is based on the actual VFS relationships, without requiring
    # PyG just to render the dashboard.
    edge_count = len(_graph_payload()["edges"])
    dirty_nodes = sum(1 for inode in inodes if inode.dirty)
    assessments = [_score_transaction(tx)["score"] for tx in current_vfs().get_transactions()[-20:]] if _model_loaded() else []
    valid_scores = [score for score in assessments if score is not None]

    return {
        "nodes": len(inodes) + len(dirents),
        "edges": edge_count,
        "dirtyNodes": dirty_nodes,
        "openTransactions": 0,
        "meanRisk": round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None,
        "journalUtilization": round((len(current_vfs().get_transactions()) - current_vfs()._durable_index) / max(1, len(current_vfs().get_transactions())), 3),
        "simulatedUptimeHrs": None,
        "crashesSurvived": 0,
    }


@app.get("/api/transactions")
def get_transactions():
    return [_frontend_transaction(tx) for tx in reversed(current_vfs().get_transactions())]


@app.get("/api/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    tx = _find_transaction(transaction_id)
    return _frontend_transaction(tx)


@app.get("/api/transactions/{transaction_id}/graph")
def transaction_graph(transaction_id: str):
    _find_transaction(transaction_id)
    return _graph_payload(transaction_id)


@app.get("/api/graph")
def graph():
    return _graph_payload()


@app.get("/api/vfs")
def get_vfs():
    return {
        "inodes": [
            {
                "inode_id": inode.inode_id,
                "path": inode.path,
                "type": inode.inode_type,
                "size": inode.size,
                "dirty": inode.dirty,
            }
            for inode in current_vfs().get_all_inodes()
        ]
    }


@app.post("/api/vfs/create")
def create_file(request: CreateRequest):
    try:
        tx = current_vfs().create(request.path, request.inode_type)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vfs/write")
def write_file(request: WriteRequest):
    try:
        tx = current_vfs().write(request.path, request.size)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vfs/link")
def link_file(request: LinkRequest):
    try:
        tx = current_vfs().link(request.existing_path, request.new_path)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vfs/pair")
def declare_application_pair(request: PairRequest):
    try:
        return {"pair": current_vfs().declare_pair(request.first, request.second, request.kind),
                "note": "Declared application dependency, not a filesystem operation"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vfs/rename")
def rename_file(request: RenameRequest):
    try:
        tx = current_vfs().rename(request.old_path, request.new_path)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/vfs/unlink")
def unlink_file(request: UnlinkRequest):
    try:
        tx = current_vfs().unlink(request.path)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vfs/open")
def open_file(request: OpenRequest):
    try:
        tx = current_vfs().open(request.path, request.mode)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vfs/close")
def close_file(request: CloseRequest):
    try:
        tx = current_vfs().close(request.fd)
        return _frontend_transaction(tx)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/risk/history")
def risk_history():
    # These are model predictions, not experimentally observed physical crashes.
    return [
        {"t": tx.timestamp, "predicted": score["score"], "observed": None, "writeAmp": None}
        for tx in current_vfs().get_transactions()
        if (score := _score_transaction(tx))["score"] is not None
    ]


@app.get("/api/risk/{transaction_id}")
def risk_score(transaction_id: str):
    return {"transactionId": transaction_id, **_score_transaction(_find_transaction(transaction_id))}


@app.get("/api/model")
def model():
    trained = _trained_model()
    metadata = trained[1] if trained else {}
    heldout = metadata.get("heldout_family_test") if trained else None
    return {
        "name": "VFSim RiskGNN",
        "architecture": "2-layer relation-aware graph attention + pair-contrast readout + MLP",
        "version": app.version,
        "status": "checkpoint_loaded" if trained else "checkpoint_not_loaded",
        "datasetNote": "Simulator oracle for application-level paired-state violations; NOT real Linux filesystem crash prediction.",
        "layers": [
            {"name": "node encoders", "type": "Linear + ReLU", "units": 32},
            {"name": "attention 1", "type": "9 typed relations", "units": 32},
            {"name": "attention 2", "type": "9 typed relations", "units": 32},
            {"name": "readout", "type": "touched inode embeddings + graph pair contrasts", "units": 41},
            {"name": "classification", "type": "MLP", "units": 1},
        ],
        "metrics": {"auc": heldout.get("roc_auc"), "precision": heldout.get("precision"),
                    "recall": heldout.get("recall"), "f1": heldout.get("f1")} if heldout else None,
        "training": [{"epoch": row["epoch"], "loss": row["train_loss"], "valLoss": row["val_loss"]}
                     for row in metadata.get("history", [])],
        "featureImportance": [],
        "lastTrained": metadata.get("trained_at"),
        "datasetSamples": metadata.get("samples"),
        "heldoutFamilies": metadata.get("heldout_families"),
        "params": sum(p.numel() for p in trained[0].parameters()) if trained else None,
    }


_CRASH_SCENARIOS = [
    {"id": "crash-last-sync", "name": "Lose work since last fsync",
     "description": "Reconstruct the namespace at the last simulated durability barrier.",
     "injectAfterSeq": 0,
     "expectedLoss": "Unsynced metadata/data changes can disappear in this simplified model.",
     "recoverySteps": ["Restore last durable snapshot", "Validate recovered namespace"]},
    {"id": "crash-before-commit", "name": "Crash before selected operation",
     "description": "Reconstruct the namespace immediately before this operation.",
     "injectAfterSeq": 0,
     "expectedLoss": "The selected operation and any later operations are excluded.",
     "recoverySteps": ["Restore pre-operation snapshot", "Validate recovered namespace"]},
    {"id": "crash-after-commit", "name": "Crash after selected operation",
     "description": "Reconstruct the namespace immediately after this operation.",
     "injectAfterSeq": 1,
     "expectedLoss": "The selected operation is represented in the recovered state.",
     "recoverySteps": ["Restore post-operation snapshot", "Validate recovered namespace"]},
]


@app.get("/api/crash/scenarios")
def crash_scenarios():
    return _CRASH_SCENARIOS


@app.post("/api/crash/simulate")
def simulate_crash(request: CrashRequest):
    _find_transaction(request.transactionId)
    try:
        return current_vfs().crash_preview(request.transactionId, request.scenarioId)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vfs/fsync")
def sync_file(request: SyncRequest):
    try:
        return _frontend_transaction(current_vfs().fsync(request.path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@app.get("/api/research/gnn")
def research_gnn(transaction_id: Optional[str] = None, scope: str = "scoring"):
    """Real typed graph, layer embeddings, and normalized inference attention."""
    from research import inspection_payload
    try:
        return inspection_payload(current_vfs(), _trained_model(), transaction_id, scope)
    except ValueError as exc:
        raise HTTPException(status_code=400 if "scope" in str(exc) else 404, detail=str(exc)) from exc


@app.get("/api/research/transactions/{transaction_id}")
def research_transaction(transaction_id: str):
    from research import transaction_analysis
    try:
        return transaction_analysis(current_vfs(), _trained_model(), transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/research/prioritize")
def prioritize_recorded_transactions(limit: int = 12):
    """Rank recorded simulated crash boundaries for further testing.

    No oracle or future recovery outcome is read here: only GNN predictions
    from each transaction's historical graph state. This is a heuristic queue,
    not a ranking of verified Linux filesystem bugs.
    """
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail='limit must be 1 to 100')
    if _trained_model() is None:
        raise HTTPException(status_code=503, detail='No compatible model checkpoint')
    candidates = []
    for position, tx in enumerate(current_vfs().get_transactions(), 1):
        score = _score_transaction(tx)['score']
        if score is not None:
            candidates.append({'transactionId': tx.txn_id, 'sequence': position,
                               'operation': tx.op_type, 'score': score})
    candidates.sort(key=lambda x: (-x['score'], x['sequence']))
    return {'candidates': [{**item, 'rank': i+1} for i, item in enumerate(candidates[:limit])],
            'totalCandidates':len(candidates),
            'disclaimer':'Heuristic simulated crash-check ordering from model scores; no verified recovery outcomes or calibrated probabilities.'}


@app.get("/api/research/benchmark")
def research_benchmark():
    import json
    trained = _trained_model()
    if trained is None:
        raise HTTPException(status_code=503, detail="No compatible trained GNN checkpoint")
    baseline_path = CHECKPOINT_PATH.parent / 'baseline_metrics.json'
    if not baseline_path.is_file():
        raise HTTPException(status_code=503, detail="No saved baseline results")
    return {"gnn": trained[1],
            "baselines": json.loads(baseline_path.read_text(encoding='utf-8')),
            "disclaimer": "All candidates and outcomes use simplified VFS snapshots and application invariants, not ext4 faults."}


@app.get("/api/research/hardmode")
def research_hardmode():
    """Saved evaluation and annotated *simulated* recovery examples.

    No recovered result is passed to an online GNN prediction, and these
    measurements must not be described as Linux filesystem crash outcomes.
    """
    import json
    folder = CHECKPOINT_PATH.parent
    metrics_file = folder / 'hardmode_metrics.json'
    examples_file = folder / 'hardmode_examples.json'
    if not metrics_file.is_file() or not examples_file.is_file():
        raise HTTPException(status_code=503, detail='FaultLab not yet trained; run python -m challenging.train --samples 5000 --epochs 5')
    return {
        'evaluation': json.loads(metrics_file.read_text(encoding='utf-8')),
        'examples': json.loads(examples_file.read_text(encoding='utf-8')),
        'boundary': 'Model observes pre-crash workload, namespace and writeback probability, NOT the realized persistence mask or recovered outcome.',
        'warning': 'Independent file/dir writeback and application oracles are simplified simulator assumptions, not ext4, POSIX, or VM evidence.',
    }


@app.get("/api/research/hardmode/readouts")
def research_hardmode_readouts():
    """Exploratory, paired comparison of both preserved FaultLab readouts."""
    import json
    path = CHECKPOINT_PATH.parent / 'readout_comparison.json'
    if not path.is_file():
        raise HTTPException(status_code=503, detail='Run python -m challenging.compare_readouts')
    return json.loads(path.read_text(encoding='utf8'))


@app.get("/api/research/hardmode/audit")
def research_hardmode_audit(version: str = 'v08'):
    """Held-out error audit, for either preserved model checkpoint."""
    import json
    if version not in ('v07','v08'):
        raise HTTPException(status_code=400, detail='Unknown FaultLab checkpoint')
    name = 'hardmode_audit_v07.json' if version == 'v07' else 'hardmode_audit.json'
    path = CHECKPOINT_PATH.parent / name
    if not path.is_file():
        raise HTTPException(status_code=503, detail='Run python -m challenging.audit from backend/')
    return json.loads(path.read_text(encoding='utf8'))


@app.get("/api/research/hardmode/trace")
def research_hardmode_trace(family: str = "hardlink_retarget", version: str = 'v08'):
    """Faithful 3-layer attention from the selected FaultLab checkpoint."""
    from challenging.trace import trace_saved_family
    try:
        return trace_saved_family(family, version=version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="FaultLab model or examples unavailable")


@app.get("/api/research/independent-v09")
def research_independent_v09():
    """Immutable, multi-seed synthetic experiment. Not Linux/ext4 validation."""
    import json
    path = CHECKPOINT_PATH.parent / "independent_v09" / "external_multiseed_results.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Independent v0.9 experiment incomplete; run python -m challenging.multiseed")
    return json.loads(path.read_text(encoding="utf8"))



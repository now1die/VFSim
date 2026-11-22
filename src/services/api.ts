/** Typed live API client. All values come from VFSim's FastAPI backend. */
import type {
  CrashScenario,
  FsEdge,
  FsNode,
  ModelInfo,
  RiskPoint,
  Transaction,
} from "@/services/types";

export const API_BASE_URL = import.meta.env.DEV
  ? (
      import.meta.env["VITE_API_URL"] ??
      import.meta.env["VITE_API_BASE_URL"] ??
      "http://127.0.0.1:8000"
    ).replace(/\/$/, "")
  : "";
export const BACKEND_MODE = "LIVE" as const;

export const ENDPOINTS = {
  health: "/api/health",
  stats: "/api/stats",
  graph: "/api/graph",
  transactionGraph: (id: string) => `/api/transactions/${encodeURIComponent(id)}/graph`,
  transactions: "/api/transactions",
  transaction: (id: string) => `/api/transactions/${encodeURIComponent(id)}`,
  riskHistory: "/api/risk/history",
  riskScore: (id: string) => `/api/risk/${encodeURIComponent(id)}`,
  model: "/api/model",
  crashScenarios: "/api/crash/scenarios",
  simulateCrash: "/api/crash/simulate",
} as const;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchApi(path: string, init?: RequestInit) {
  const timeout = AbortSignal.timeout(30_000);
  const headers = new Headers(init?.headers);
  headers.set("accept", "application/json");
  if (init?.body) headers.set("content-type", "application/json");
  try {
    return await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      credentials: "same-origin",
      signal: init?.signal ? AbortSignal.any([init.signal, timeout]) : timeout,
    });
  } catch (error) {
    if (init?.signal?.aborted) throw error;
    throw new ApiError(
      timeout.aborted
        ? "The backend took too long to respond."
        : "The backend could not be reached.",
    );
  }
}

async function responseError(response: Response) {
  let detail = `VFSim request failed (HTTP ${response.status}).`;
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") detail = payload.detail;
    else if (Array.isArray(payload.detail)) detail = "The request contains an invalid value.";
  } catch {
    /* Non-JSON errors can come from the hosting platform. */
  }
  return new ApiError(detail, response.status);
}

async function bootstrapSession() {
  const create = () => fetchApi("/api/session", { method: "POST" });
  return typeof navigator !== "undefined" && navigator.locks
    ? await navigator.locks.request("vfsim-session", create)
    : await create();
}

let session: Promise<void> | undefined;
function establishSession() {
  session ??= bootstrapSession()
    .then(async (response) => {
      if (!response.ok) throw await responseError(response);
    })
    .catch((error: unknown) => {
      session = undefined;
      throw error;
    });
  return session;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const needsSession = path !== ENDPOINTS.health && API_BASE_URL === "";
  const requestSession = needsSession ? establishSession() : undefined;
  await requestSession;
  let response = await fetchApi(path, init);
  if (needsSession && response.status === 401) {
    await response.body?.cancel();
    if (session === requestSession) session = undefined;
    if (init?.method && init.method !== "GET") {
      throw new ApiError(
        "The simulator session expired. Retry the operation to start a new session.",
        401,
      );
    }
    await establishSession();
    response = await fetchApi(path, init);
  }
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

export interface GraphPayload {
  nodes: FsNode[];
  edges: FsEdge[];
  touched: string[];
  transactionId: string | null;
}
export interface RiskAssessment {
  transactionId: string;
  score: number | null;
  confidence: number | null;
  modelLoaded?: boolean;
  explanations: { feature: string; contribution: number; note: string }[];
}
export interface CrashResult {
  scenarioId: string;
  transactionId: string;
  survivedOps: number;
  lostOps: number;
  recoverySteps: string[];
  postRecoveryRisk: number | null;
  selectedOperationSurvived?: boolean;
  recoveredPaths?: string[];
  lostPaths?: string[];
  restoredPaths?: string[];
  consistencyPassed?: boolean;
}
export interface SystemStats {
  nodes: number;
  edges: number;
  dirtyNodes: number;
  openTransactions: number;
  meanRisk: number | null;
  journalUtilization: number | null;
  simulatedUptimeHrs: number | null;
  crashesSurvived: number;
}

export const api = {
  createFile: (path: string, inodeType: "file" | "directory"): Promise<Transaction> =>
    request("/api/vfs/create", {
      method: "POST",
      body: JSON.stringify({ path, inode_type: inodeType }),
    }),
  writeFile: (path: string, size: number): Promise<Transaction> =>
    request("/api/vfs/write", { method: "POST", body: JSON.stringify({ path, size }) }),
  linkFile: (existingPath: string, newPath: string): Promise<Transaction> =>
    request("/api/vfs/link", {
      method: "POST",
      body: JSON.stringify({ existing_path: existingPath, new_path: newPath }),
    }),
  renameFile: (oldPath: string, newPath: string): Promise<Transaction> =>
    request("/api/vfs/rename", {
      method: "POST",
      body: JSON.stringify({ old_path: oldPath, new_path: newPath }),
    }),
  unlinkFile: (path: string): Promise<Transaction> =>
    request("/api/vfs/unlink", { method: "DELETE", body: JSON.stringify({ path }) }),
  syncFile: (path: string): Promise<Transaction> =>
    request("/api/vfs/fsync", { method: "POST", body: JSON.stringify({ path }) }),
  getHealth: (): Promise<{ status: string; mode: string; version: string; checkedAt: string }> =>
    request(ENDPOINTS.health),
  getStats: (): Promise<SystemStats> => request(ENDPOINTS.stats),
  getGraph: (transactionId?: string): Promise<GraphPayload> =>
    request(transactionId ? ENDPOINTS.transactionGraph(transactionId) : ENDPOINTS.graph),
  getTransactions: (): Promise<Transaction[]> => request(ENDPOINTS.transactions),
  getTransaction: (id: string): Promise<Transaction> => request(ENDPOINTS.transaction(id)),
  getRiskHistory: (): Promise<RiskPoint[]> => request(ENDPOINTS.riskHistory),
  getRiskAssessment: (id: string): Promise<RiskAssessment> => request(ENDPOINTS.riskScore(id)),
  getModel: (): Promise<ModelInfo> => request(ENDPOINTS.model),
  getCrashScenarios: (): Promise<CrashScenario[]> => request(ENDPOINTS.crashScenarios),
  simulateCrash: (transactionId: string, scenarioId: string): Promise<CrashResult> =>
    request(ENDPOINTS.simulateCrash, {
      method: "POST",
      body: JSON.stringify({ transactionId, scenarioId }),
    }),
};
export type { Transaction, FsNode, FsEdge, RiskPoint, ModelInfo, CrashScenario };

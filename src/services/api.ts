/** Typed live API client. All values come from VFSim's FastAPI backend. */
import type { CrashScenario, FsEdge, FsNode, ModelInfo, RiskPoint, Transaction } from "@/services/types";

export const API_BASE_URL: string = import.meta.env["VITE_API_BASE_URL"] ?? "http://127.0.0.1:8000";
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
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}${path}`, {
    ...init,
    headers: { Accept: "application/json", "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let detail = `Request to ${path} failed (${response.status})`;
    try {
      const payload = await response.json() as {detail?: string};
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch { /* FastAPI may return plain text for infrastructure errors. */ }
    throw new ApiError(detail, response.status);
  }
  return await response.json() as T;
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
    request("/api/vfs/create", { method: "POST", body: JSON.stringify({ path, inode_type: inodeType }) }),
  writeFile: (path: string, size: number): Promise<Transaction> =>
    request("/api/vfs/write", { method: "POST", body: JSON.stringify({ path, size }) }),
  linkFile: (existingPath: string, newPath: string): Promise<Transaction> =>
    request("/api/vfs/link", { method: "POST", body: JSON.stringify({ existing_path: existingPath, new_path: newPath }) }),
  renameFile: (oldPath: string, newPath: string): Promise<Transaction> =>
    request("/api/vfs/rename", { method: "POST", body: JSON.stringify({ old_path: oldPath, new_path: newPath }) }),
  unlinkFile: (path: string): Promise<Transaction> =>
    request("/api/vfs/unlink", { method: "DELETE", body: JSON.stringify({ path }) }),
  syncFile: (path: string): Promise<Transaction> =>
    request("/api/vfs/fsync", { method: "POST", body: JSON.stringify({ path }) }),
  getHealth: (): Promise<{status:string;mode:string;version:string;checkedAt:string}> => request(ENDPOINTS.health),
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
    request(ENDPOINTS.simulateCrash, {method:"POST",body:JSON.stringify({transactionId,scenarioId})}),
};
export type { Transaction, FsNode, FsEdge, RiskPoint, ModelInfo, CrashScenario };

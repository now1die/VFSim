/** Types for the live VFSim FastAPI JSON contract. No test fixtures or fake values. */
export type RiskLevel = "safe" | "watch" | "elevated" | "critical" | "unknown";
export interface FsNode {
  id: string;
  name: string;
  path: string;
  kind: "directory" | "file" | "inode" | "journal" | "superblock";
  sizeKb: number;
  riskScore: number | null;
  riskLevel: RiskLevel;
  lastWrite: string;
  dirty: boolean;
  embedding: number[];
  inodeNumber?: number;
  linkCount?: number;
}
export interface FsEdge {
  id: string;
  source: string;
  target: string;
  relation: "contains" | "points-to" | "journals" | "hardlink" | string;
  weight: number;
}
export interface TxOperation {
  id: string;
  seq: number;
  op: string;
  target: string;
  bytes: number;
  journaled: boolean;
  riskDelta: number;
}
export interface Transaction {
  id: string;
  label: string;
  status: "committed" | "in-flight" | "aborted" | "recovered";
  startedAt: string;
  durationMs: number;
  operations: TxOperation[];
  riskScore: number | null;
  riskLevel: RiskLevel;
}
export interface RiskPoint {
  t: string;
  predicted: number | null;
  observed: number | null;
  writeAmp: number | null;
}
export interface ModelInfo {
  name: string;
  architecture: string;
  version: string;
  status?: string;
  datasetNote?: string;
  layers: { name: string; type: string; units: number }[];
  metrics: { auc: number; precision: number; recall: number; f1: number } | null;
  training: { epoch: number; loss: number; valLoss: number }[];
  featureImportance: { feature: string; weight: number }[];
  lastTrained: string;
  params: number;
}
export interface CrashScenario {
  id: string;
  name: string;
  description: string;
  injectAfterSeq: number;
  expectedLoss: string;
  recoverySteps: string[];
}

/** VFSim v1.0 typed access to original v0.9 research endpoints.
 * No client-side fabricated scores, oracle outcomes, or model attention values.
 */
import { request as read } from "@/services/api";

export type EmbeddingStages = Record<string, number[]>;
export interface GnnNode {
  id: string;
  nodeType: string;
  label: string;
  subtitle?: string;
  path?: string;
  touched?: boolean;
  dirty?: boolean;
  features?: Record<string, number>;
  embeddings?: EmbeddingStages;
  embeddingNorms?: Record<string, number>;
}
export interface GnnEdge {
  id?: string;
  source: string;
  target: string;
  relation?: string;
  relationKey?: string;
  attention?: Record<string, number | number[]>;
  attentionByLayer?: Record<string, number | number[]>;
  [key: string]: unknown;
}
export interface FilesystemGnnTrace {
  transactionId: string | null;
  graphScope: string;
  scope: string;
  modelLoaded: boolean;
  score: number | null;
  nodes: GnnNode[];
  edges: GnnEdge[];
  counts: Record<string, number>;
  relations: Record<string, number>;
  attentionDescription?: string;
  trainingWarning?: string;
}
export interface FaultNode {
  id: number;
  role: string;
  label: string;
  features?: number[];
  embedding?: EmbeddingStages;
}
export interface FaultTrace {
  model: string;
  model_variant: string;
  family: string;
  case_id: string;
  risk_prediction: number;
  oracle_risk: number;
  absolute_error: number;
  constraint: string;
  constraint_paths: string[];
  nodes: FaultNode[];
  edges: Array<Record<string, unknown>>;
  layers: Array<Record<string, unknown>>;
  warning: string;
  attention_description: string;
  feature_width: number;
  embedding_width: number;
}
export type Metric = { n: number; mean: number; std_across_seeds: number; values: number[]; min: number; max: number };
export interface MultiseedEvidence {
  study: string;
  registered_config: { training_seeds: number[]; samples_per_seed: number; external_per_family: number };
  external_cohort: { cohort_sha256: string; workload_families: string[]; cases_per_family: number; summary: { count: number } };
  summary: { models: Record<string, Record<string, Metric>> };
  limitations: string | string[];
  runs: Array<Record<string, unknown>>;
}
export interface HardmodeReport {
  evaluation: {name: string; heldout_families: string[]; split: Record<string,number>};
  examples: unknown;
  boundary: string;
  warning: string;
}

export const researchApi = {
  filesystemTrace: (scope: "full" | "scoring" = "full", transactionId?: string): Promise<FilesystemGnnTrace> =>
    read(`/api/research/gnn?scope=${scope}${transactionId ? `&transaction_id=${encodeURIComponent(transactionId)}` : ""}`),
  faultlabTrace: (family: string, version: string): Promise<FaultTrace> =>
    read(`/api/research/hardmode/trace?family=${encodeURIComponent(family)}&version=${encodeURIComponent(version)}`),
  independent: (): Promise<MultiseedEvidence> => read("/api/research/independent-v09"),
  faultlab: (): Promise<HardmodeReport> => read("/api/research/hardmode"),
  crashPriority: (): Promise<{candidates: Array<{rank:number;transactionId:string;operation:string;score:number}>;disclaimer:string}> =>
    read("/api/research/prioritize"),
};

export const faultlabFamilies = [
  {value:"editor_autosave",label:"Editor autosave"},
  {value:"package_upgrade",label:"Package upgrade"},
  {value:"mirrored_alias_rotation",label:"Hard-link rotation"},
  {value:"manifest_publication",label:"Manifest publication"},
  {value:"directory_migration",label:"Directory migration"},
  {value:"snapshot_batch",label:"Snapshot batch"},
  {value:"sharded_commit",label:"Sharded commit"},
  {value:"hardlink_retarget",label:"Hard-link retarget"},
];
export const faultlabModels = [
  {value:"v09_s11",label:"v0.9 · seed 11"},
  {value:"v09_s29",label:"v0.9 · seed 29"},
  {value:"v09_s47",label:"v0.9 · seed 47"},
  {value:"v09_s83",label:"v0.9 · seed 83"},
  {value:"v09_s131",label:"v0.9 · seed 131"},
  {value:"v08",label:"FaultLab v0.8"},
  {value:"v07",label:"FaultLab v0.7"},
];

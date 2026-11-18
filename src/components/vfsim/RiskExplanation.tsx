import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { GitBranch, Info, Layers3, ShieldAlert } from "lucide-react";

import { api, type RiskAssessment } from "@/services/api";
import type { Transaction } from "@/services/types";

/**
 * The live RiskGNN endpoint currently exposes scores but no calibrated confidence
 * or per-transaction feature attribution. Show real context, never inferred causes.
 */
export function RiskExplanation({
  transactionId,
  assessment,
  transaction,
}: {
  transactionId: string;
  assessment: RiskAssessment;
  transaction?: Transaction;
}) {
  const model = useQuery({
    queryKey: ["model"],
    queryFn: api.getModel,
    staleTime: 60_000,
  });
  const graph = useQuery({
    queryKey: ["transaction-graph", transactionId],
    queryFn: () => api.getGraph(transactionId),
    enabled: Boolean(transactionId),
    staleTime: 15_000,
  });

  const explanations = assessment.explanations ?? [];
  const touchedIds = new Set(graph.data?.touched ?? []);
  const touchedNodes = (graph.data?.nodes ?? []).filter((node) => touchedIds.has(node.id));
  const adjacentEdges = (graph.data?.edges ?? []).filter(
    (edge) => touchedIds.has(edge.source) || touchedIds.has(edge.target),
  );
  const relationTypes = Array.from(new Set(adjacentEdges.map((edge) => edge.relation)));

  return (
    <div className="risk-explanation space-y-4" aria-live="polite">
      <div className="risk-explanation-summary">
        <div>
          <p className="editorial-label">Model prediction</p>
          <p className="risk-explanation-score tabular-nums">
            {assessment.score == null ? "Unavailable" : assessment.score.toFixed(3)}
          </p>
          <p className="text-xs text-muted-foreground">
            Score on a 0–1 scale, not a verified real-world crash probability
          </p>
        </div>
        <div>
          <p className="editorial-label">Model / checkpoint</p>
          {model.isPending ? (
            <p className="text-sm">Loading model information…</p>
          ) : model.isError ? (
            <p className="text-sm text-muted-foreground">Model information unavailable</p>
          ) : (
            <>
              <p className="text-sm font-semibold">{model.data.name}</p>
              <p className="text-xs text-muted-foreground">
                {model.data.status === "checkpoint_loaded"
                  ? "Checkpoint loaded"
                  : "Checkpoint not loaded"}
              </p>
            </>
          )}
        </div>
        <div>
          <p className="editorial-label">Selected transaction</p>
          <p className="break-words text-sm font-semibold">
            {transaction?.label ?? transactionId}
          </p>
          <p className="text-xs text-muted-foreground">
            {assessment.confidence == null
              ? "Calibrated confidence: not provided"
              : `Reported confidence: ${(assessment.confidence * 100).toFixed(1)}%`}
          </p>
        </div>
      </div>

      <section className="risk-context-panel" aria-label="Filesystem graph context">
        <h3 className="mb-2 flex items-center gap-2 text-sm font-bold">
          <GitBranch className="size-4" aria-hidden="true" />
          Related filesystem graph context
        </h3>
        {graph.isPending ? (
          <p className="text-sm text-muted-foreground">Loading graph context…</p>
        ) : graph.isError ? (
          <p className="text-sm text-muted-foreground">
            Graph context is unavailable for this transaction.
          </p>
        ) : touchedNodes.length ? (
          <>
            <p className="text-sm">
              {touchedNodes.length} touched {touchedNodes.length === 1 ? "node" : "nodes"}
              {adjacentEdges.length ? ` · ${adjacentEdges.length} adjacent relationships` : ""}
            </p>
            <ul className="mt-2 flex flex-wrap gap-2" aria-label="Touched graph nodes">
              {touchedNodes.slice(0, 8).map((node) => (
                <li key={node.id} className="risk-node-chip" title={node.path || node.id}>
                  {node.path || node.name || node.id}
                </li>
              ))}
            </ul>
            {relationTypes.length > 0 ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Visible relationship types: {relationTypes.join(", ")}
              </p>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            No touched nodes were returned for the current graph state.
          </p>
        )}
        <p className="mt-2 text-xs text-muted-foreground">
          This is graph context, not a ranking of which features caused the prediction.
        </p>
      </section>

      {explanations.length > 0 ? (
        <section aria-label="Backend-provided attributions">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-bold">
            <Layers3 className="size-4" aria-hidden="true" />
            Backend-provided feature contributions
          </h3>
          <ul className="space-y-3">
            {explanations.map((ex, index) => (
              <li key={`${ex.feature}-${index}`}>
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="break-all font-semibold">{ex.feature}</span>
                  <span className="tabular-nums">{ex.contribution >= 0 ? "+" : ""}{ex.contribution.toFixed(3)}</span>
                </div>
                <div className="mt-1 h-2 bg-surface-2">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${Math.min(100, Math.abs(ex.contribution) * 300)}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{ex.note}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <div className="risk-unavailable-note" role="note">
          <Info className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
          <div>
            <p className="text-sm font-bold">Per-transaction attribution is not available</p>
            <p className="mt-1 text-sm">
              The live RiskGNN returns a prediction but does not expose calibrated
              feature contributions or attention for this specific transaction.
              VFSim will not generate fictional weights to fill this panel.
            </p>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3 border-t-2 border-border/30 pt-3">
        <p className="flex max-w-2xl items-start gap-2 text-xs text-muted-foreground">
          <ShieldAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          Saved attention traces in Model Lab belong to their selected research
          checkpoint and workload; they are not causal explanations of this live score.
        </p>
        <div className="flex flex-wrap gap-2">
          <Link className="pixel-detail-link" to="/graph">Inspect filesystem graph</Link>
          <Link className="pixel-detail-link" to="/model">Open Model Lab</Link>
        </div>
      </div>
    </div>
  );
}

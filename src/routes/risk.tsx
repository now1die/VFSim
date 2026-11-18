import { useState } from "react";
import { MultiseedPanel } from "@/components/vfsim/MultiseedPanel";
import { ResearchGraph } from "@/components/vfsim/ResearchGraph";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Gauge, LineChart as LineChartIcon, ShieldAlert, ShieldCheck } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Slider } from "@/components/ui/slider";
import {
  ErrorState,
  LoadingState,
  PageHeader,
  Panel,
  RiskBadge,
} from "@/components/vfsim/primitives";
import { RiskGauge } from "@/components/vfsim/RiskGauge";
import { RiskExplanation } from "@/components/vfsim/RiskExplanation";
import { TransactionPicker } from "@/components/vfsim/TransactionPicker";
import { useSelection } from "@/components/vfsim/selection";
import { api } from "@/services/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/risk")({
  head: () => ({
    meta: [
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
      { title: "Risk Analysis — VFSim" },
      {
        name: "description",
        content:
          "GNN risk gauge, feature attribution and journaling decision for the selected filesystem transaction, tuned by a risk threshold.",
      },
      { property: "og:title", content: "VFSim Risk Analysis" },
      {
        property: "og:description",
        content: "Risk gauge, attribution and journaling decision per transaction.",
      },
    ],
  }),
  component: RiskPage,
});

function OriginalRiskPage() {
  const { selectedTxId, riskThreshold, setRiskThreshold } = useSelection();
  const history = useQuery({ queryKey: ["risk-history"], queryFn: () => api.getRiskHistory() });
  const tx = useQuery({
    queryKey: ["transaction", selectedTxId],
    queryFn: () => api.getTransaction(selectedTxId),
    enabled: Boolean(selectedTxId),
  });
  const assessment = useQuery({
    queryKey: ["risk", selectedTxId],
    queryFn: () => api.getRiskAssessment(selectedTxId),
    enabled: Boolean(selectedTxId),
  });

  const score = assessment.data?.score ?? null;
  const shouldJournal = score != null && score >= riskThreshold;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Risk analysis"
        description="Transaction risk from the backend model when a trained checkpoint is available. The threshold is a policy preview and does not itself enforce journaling."
        actions={<TransactionPicker />}
      />

      <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="Risk gauge" subtitle={selectedTxId} icon={Gauge}>
            {assessment.isPending ? (
              <LoadingState label="Scoring transaction" rows={2} />
            ) : assessment.isError ? (
              <ErrorState onRetry={() => void assessment.refetch()} />
            ) : (
              <div className="space-y-4">
                <RiskGauge score={score} threshold={riskThreshold} />
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>reported confidence</span>
                  <span className="tabular text-foreground">
                    {assessment.data?.confidence == null ? "—" : `${(assessment.data.confidence * 100).toFixed(1)}%`}
                  </span>
                </div>
                {tx.data ? (
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{tx.data.label}</span>
                    <RiskBadge level={tx.data.riskLevel} />
                  </div>
                ) : null}
              </div>
            )}
          </Panel>

          <Panel title="Risk threshold" subtitle="proposed policy preview">
            <Slider
              value={[riskThreshold]}
              min={0}
              max={1}
              step={0.01}
              onValueChange={([v]) => setRiskThreshold(v ?? 0)}
              aria-label="Risk threshold"
            />
            <div className="mt-2 flex justify-between text-[11px] text-muted-foreground">
              <span>0.00 journal more</span>
              <span className="tabular text-primary">{riskThreshold.toFixed(2)}</span>
              <span>1.00 journal less</span>
            </div>

            <div
              className={cn(
                "mt-4 rounded-md border p-3",
                shouldJournal
                  ? "border-critical/50 bg-critical/10"
                  : "border-safe/40 bg-safe/10",
              )}
            >
              <p
                className={cn(
                  "flex items-center gap-2 text-sm font-semibold",
                  score == null ? "text-muted-foreground" : shouldJournal ? "text-critical" : "text-safe",
                )}
              >
                {score == null ? (
                  <ShieldAlert className="size-4" aria-hidden />
                ) : shouldJournal ? (
                  <ShieldAlert className="size-4" aria-hidden />
                ) : (
                  <ShieldCheck className="size-4" aria-hidden />
                )}
                {score == null
                  ? "GNN score unavailable"
                  : shouldJournal
                    ? "Recommend full journaling"
                    : "Fast path proposed"}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {score == null
                  ? "Run the training pipeline to create backend/artifacts/riskgnn.pt before using live GNN scoring."
                  : shouldJournal
                    ? `Risk ${score.toFixed(2)} ≥ threshold ${riskThreshold.toFixed(2)} — full journaling is recommended, not executed by this control.`
                    : `Risk ${score.toFixed(2)} < threshold ${riskThreshold.toFixed(2)} — fast path is proposed; durability has not been verified.`}
              </p>
            </div>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="Risk history" subtitle="predicted vs observed, 24h" icon={LineChartIcon}>
            {history.isPending ? (
              <LoadingState label="Loading risk history" rows={3} />
            ) : history.isError || !history.data ? (
              <ErrorState onRetry={() => void history.refetch()} />
            ) : (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history.data} margin={{ left: -20, right: 8, top: 8 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="t"
                      stroke="var(--muted-foreground)"
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke="var(--muted-foreground)"
                      fontSize={11}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--popover)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine
                      y={riskThreshold}
                      stroke="var(--critical)"
                      strokeDasharray="5 4"
                      label={{
                        value: `threshold ${riskThreshold.toFixed(2)}`,
                        fill: "var(--critical)",
                        fontSize: 10,
                        position: "insideTopRight",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      stroke="var(--chart-1)"
                      strokeWidth={2}
                      dot={false}
                      name="GNN predicted"
                    />
                    <Line
                      type="monotone"
                      dataKey="observed"
                      stroke="var(--chart-3)"
                      strokeWidth={2}
                      strokeDasharray="4 3"
                      dot={false}
                      name="Observed"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>

          <Panel title="Why this score?" subtitle={selectedTxId ? `Model evidence for ${selectedTxId}` : "Select a transaction"}>
            {!selectedTxId ? (
              <p className="text-sm text-muted-foreground">Select a transaction to inspect its prediction and graph context.</p>
            ) : assessment.isPending ? (
              <LoadingState label="Loading score details" rows={3} />
            ) : assessment.isError || !assessment.data ? (
              <ErrorState message="Couldn't load the selected transaction's model result." onRetry={() => void assessment.refetch()} />
            ) : (
              <RiskExplanation
                transactionId={selectedTxId}
                assessment={assessment.data}
                {...(tx.data ? { transaction: tx.data } : {})}
              />
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function RiskPage() {
  const [tab,setTab]=useState<"live"|"evaluation"|"faultlab">("evaluation");
  return <div className="space-y-5">
    <nav aria-label="Risk analysis views" className="flex flex-wrap gap-2">
      {[{id:"live",label:"Live transactions"},{id:"evaluation",label:"Independent benchmarks"},{id:"faultlab",label:"Crash-risk graph lab"}].map(x=><button key={x.id} onClick={()=>setTab(x.id as typeof tab)} className={`rounded-md border px-4 py-2 text-sm ${tab===x.id?"border-primary bg-primary/10 text-primary":"border-border text-muted-foreground"}`}>{x.label}</button>)}
    </nav>
    {tab==="live"?<OriginalRiskPage/>:tab==="evaluation"?<><PageHeader title="Research benchmarks" description="Frozen six-family, five-seed synthetic benchmark; compare GNN with competitive baselines and inspect reproducibility."/><MultiseedPanel/></>:<><PageHeader title="FaultLab · crash-risk graph" description="Actual saved attention traces, simulator failure-probability oracle and per-case prediction error."/><ResearchGraph/></>}
  </div>;
}

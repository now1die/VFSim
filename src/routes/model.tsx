import { useState } from "react";
import { ResearchGraph } from "@/components/vfsim/ResearchGraph";
import { MultiseedPanel } from "@/components/vfsim/MultiseedPanel";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { BrainCircuit, Layers } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  ErrorState,
  LoadingState,
  PageHeader,
  Panel,
  StatCard,
} from "@/components/vfsim/primitives";
import { api } from "@/services/api";

export const Route = createFileRoute("/model")({
  head: () => ({
    meta: [
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
      { title: "GNN Model — VFSim" },
      {
        name: "description",
        content:
          "Architecture, training curves, metrics and feature importance for the VFSim graph neural network risk model.",
      },
      { property: "og:title", content: "VFSim GNN Model" },
      {
        property: "og:description",
        content: "Architecture, training curves and feature importance of the risk model.",
      },
    ],
  }),
  component: ModelPage,
});

function OriginalModelPage() {
  const model = useQuery({ queryKey: ["model"], queryFn: () => api.getModel() });

  return (
    <div className="space-y-6">
      <PageHeader
        title="GNN risk model"
        description="Current RiskGNN architecture and evaluation status from the FastAPI backend."
      />

      {model.isPending ? (
        <LoadingState label="Loading model card" rows={4} />
      ) : model.isError || !model.data ? (
        <ErrorState onRetry={() => void model.refetch()} />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard label="Checkpoint" value={model.data.lastTrained ? "loaded" : "not loaded"} />
            <StatCard label="AUC" value={model.data.metrics ? model.data.metrics.auc.toFixed(3) : "—"} />
            <StatCard label="Precision" value={model.data.metrics ? model.data.metrics.precision.toFixed(3) : "—"} />
            <StatCard label="F1" value={model.data.metrics ? model.data.metrics.f1.toFixed(3) : "—"} />
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <Panel
              title="Training curves"
              subtitle={`${model.data.version} · ${model.data.params == null ? "parameter count unavailable" : `${model.data.params.toLocaleString()} params`}`}
              icon={BrainCircuit}
              className="xl:col-span-2"
            >
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={model.data.training} margin={{ left: -20, right: 8, top: 8 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="epoch"
                      stroke="var(--muted-foreground)"
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis stroke="var(--muted-foreground)" fontSize={11} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: "var(--popover)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      stroke="var(--chart-1)"
                      strokeWidth={2}
                      dot={false}
                      name="train loss"
                    />
                    <Line
                      type="monotone"
                      dataKey="valLoss"
                      stroke="var(--chart-5)"
                      strokeWidth={2}
                      dot={false}
                      name="val loss"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel title="Architecture" subtitle={model.data.architecture} icon={Layers}>
              <ol className="space-y-2">
                {model.data.layers.map((l) => (
                  <li
                    key={l.name}
                    className="flex items-center justify-between rounded-md border border-border px-3 py-2 text-xs"
                  >
                    <span className="tabular text-foreground">{l.name}</span>
                    <span className="text-muted-foreground">{l.type}</span>
                    <span className="tabular text-primary">{l.units}</span>
                  </li>
                ))}
              </ol>
              <p className="mt-3 text-[11px] text-muted-foreground">
                {model.data.lastTrained
                  ? `Last trained ${new Date(model.data.lastTrained).toISOString().replace("T", " ").slice(0, 16)} UTC`
                  : "No saved checkpoint is loaded yet."}
              </p>
            </Panel>
          </div>

          <Panel title="Feature importance" subtitle={model.data.featureImportance.length ? "reported by the backend model" : "not available until a trained checkpoint is evaluated"}>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={model.data.featureImportance}
                  layout="vertical"
                  margin={{ left: 90, right: 16 }}
                >
                  <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" stroke="var(--muted-foreground)" fontSize={11} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    stroke="var(--muted-foreground)"
                    fontSize={10}
                    width={140}
                  />
                  <Tooltip
                    cursor={{ fill: "var(--surface-2)" }}
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="weight" fill="var(--chart-1)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

function ModelPage() {
  const [tab,setTab]=useState<"legacy"|"trace"|"evidence">("trace");
  return <div className="space-y-5">
    <nav aria-label="Model laboratory views" className="flex flex-wrap gap-2">
      {[{id:"trace",label:"Trained attention explorer"},{id:"legacy",label:"Filesystem RiskGNN"},{id:"evidence",label:"Five-seed evaluation"}].map(x=><button key={x.id} onClick={()=>setTab(x.id as typeof tab)} className={`rounded-md border px-4 py-2 text-sm ${tab===x.id?"border-primary bg-primary/10 text-primary":"border-border text-muted-foreground"}`}>{x.label}</button>)}
    </nav>
    {tab==="legacy"?<OriginalModelPage/>:tab==="trace"?<><PageHeader title="GNN Model Lab" description="Live model-specific heterogeneous graph, attention coefficients and node embeddings. Compare the saved v0.7, v0.8 and five v0.9 checkpoints."/><ResearchGraph/></>:<><PageHeader title="Repeated-seed evidence" description="Five independent fits and six entirely held-out synthetic workload families. All measurements are simulator-derived."/><MultiseedPanel/></>}
    {tab==="legacy"&&<p className="text-xs text-muted-foreground">The older PyTorch Geometric architecture is archived under backend/legacy_pyg and is not part of the reported v0.9 evaluation. The figures above belong to the Torch-native filesystem checkpoint.</p>}
  </div>;
}

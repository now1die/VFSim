import { useQuery } from "@tanstack/react-query";
import { BrainCircuit, ChartBar, ShieldAlert } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { researchApi } from "@/services/research";
import { ErrorState, LoadingState, Panel, StatCard } from "@/components/vfsim/primitives";

const mainModels = ["GNN v0.8 operand-aware","Random Forest","XGBoost","HistGradientBoosting"];
const shortLabels: Record<string,string> = {"GNN v0.8 operand-aware":"GNN","HistGradientBoosting":"HistGB"};

/** All numbers are served by the frozen synthetic 1,440-case external cohort. */
export function MultiseedPanel({compact = false}: {compact?: boolean}) {
  const query = useQuery({queryKey:["research-v09-evidence"],queryFn:researchApi.independent,staleTime:300_000});
  if (query.isPending) return <LoadingState label="Reading five-seed evidence" rows={3}/>;
  if (query.isError || !query.data) return <ErrorState message="Five-seed evidence is unavailable" onRetry={() => void query.refetch()}/>;
  const data = query.data;
  const rows = mainModels.map(name=>({
    model:shortLabels[name]??name,
    mae: data.summary.models[name]?.["mae_to_oracle"]?.mean,
    sd: data.summary.models[name]?.["mae_to_oracle"]?.std_across_seeds,
    top10: data.summary.models[name]?.["expected_failures_top10pct"]?.mean,
  })).filter(x=>typeof x.mae === "number");
  const gnn = rows[0];
  return <div className="space-y-4" data-testid="v09-evidence">
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <StatCard label="External workloads" value={data.external_cohort.summary.count} hint="fixed synthetic cohort" icon={ChartBar}/>
      <StatCard label="Training seeds" value={data.registered_config.training_seeds.length} hint="independent model fits" icon={BrainCircuit}/>
      <StatCard label="GNN MAE" value={gnn?.mae?.toFixed(4) ?? "—"} hint="lower is better" icon={ChartBar}/>
      <StatCard label="Expected top-10%" value={gnn?.top10?.toFixed(2) ?? "—"} hint="modeled violations / 144 tests" icon={ShieldAlert}/>
    </div>
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Generalization error" subtitle="MAE to simulated failure-probability oracle · mean of 5 seeds">
        <div className="h-64" data-testid="mae-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={rows} margin={{left:0,right:8,top:8,bottom:0}}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3"/><XAxis dataKey="model" tick={{fill:"var(--muted-foreground)",fontSize:11}}/><YAxis domain={[0,"auto"]} tick={{fill:"var(--muted-foreground)",fontSize:11}}/><Tooltip contentStyle={{background:"var(--popover)",border:"1px solid var(--border)"}}/><Bar dataKey="mae" name="MAE ↓" fill="var(--chart-1)" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div>
      </Panel>
      <Panel title="Crash-test prioritization" subtitle="Expected simulated violations in top 144 ranked cases">
        <div className="h-64" data-testid="priority-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={rows} margin={{left:0,right:8,top:8,bottom:0}}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3"/><XAxis dataKey="model" tick={{fill:"var(--muted-foreground)",fontSize:11}}/><YAxis domain={[0,"auto"]} tick={{fill:"var(--muted-foreground)",fontSize:11}}/><Tooltip contentStyle={{background:"var(--popover)",border:"1px solid var(--border)"}}/><Bar dataKey="top10" name="Expected violations ↑" fill="var(--chart-3)" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div>
      </Panel>
    </div>
    {!compact && <Panel title="Seed-by-seed reproducibility" subtitle="Mean ± sample standard deviation; exact metrics from saved checkpoints">
      <div className="overflow-x-auto"><table className="w-full text-sm" data-testid="multiseed-table"><thead><tr className="border-b border-border text-left text-muted-foreground"><th className="p-2">Model</th><th className="p-2">MAE ↓</th><th className="p-2">SD</th><th className="p-2">Top-10% violations ↑</th></tr></thead><tbody>{rows.map(r=><tr className="border-b border-border/60" key={r.model}><td className="p-2 font-medium">{r.model}</td><td className="p-2 tabular">{r.mae?.toFixed(4)}</td><td className="p-2 tabular">{r.sd?.toFixed(4)}</td><td className="p-2 tabular">{r.top10?.toFixed(2)}</td></tr>)}</tbody></table></div>
    </Panel>}
    <p className="rounded-md border border-watch/30 bg-watch/5 px-3 py-2 text-xs text-muted-foreground"><strong>Research boundary:</strong> All results are from a simplified independent-writeback simulator, not measured Linux ext4 recovery. Attention weights are model internals, not causal evidence.</p>
  </div>;
}

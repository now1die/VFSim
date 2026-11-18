import { useQuery } from "@tanstack/react-query";
import { BrainCircuit, ChartBar, Database, Target, Trophy, Layers } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ErrorState, LoadingState, Panel, StatCard } from "@/components/vfsim/primitives";
import { researchApi } from "@/services/research";

const GNN = "GNN v0.8 operand-aware";
const RF = "Random Forest";
const short = (n: string) => (n === GNN ? "VFSim-GNN (ours)" : n);
const pretty = (f: string) => f.replace(/_/g, " ");

type RunModel = { mae_to_oracle?: number; mean_oracle_risk?: number };
type Run = {
  training_seed: number;
  best_validation_mae: number;
  selected_epoch: number;
  models: Record<string, RunModel>;
  by_family: Record<string, Record<string, { mae: number }>>;
};

/** Benchmark view for the Crash Lab. Every number comes from the frozen v0.9 external cohort. */
export function CrashBenchmarks() {
  const q = useQuery({
    queryKey: ["research-v09-evidence"],
    queryFn: researchApi.independent,
    staleTime: 300_000,
  });
  if (q.isPending) return <LoadingState label="Reading benchmark evidence" rows={3} />;
  if (q.isError || !q.data)
    return (
      <ErrorState message="Benchmark evidence is unavailable" onRetry={() => void q.refetch()} />
    );

  const d = q.data;
  const runs = d.runs as unknown as Run[];
  const models = d.summary.models;

  const ranking = Object.entries(models)
    .map(([name, m]) => ({
      name,
      mae: m["mae_to_oracle"]?.mean ?? NaN,
      sd: m["mae_to_oracle"]?.std_across_seeds ?? NaN,
      rho: m["rank_spearman"]?.mean ?? NaN,
      top10: m["expected_failures_top10pct"]?.mean ?? NaN,
    }))
    .filter((r) => Number.isFinite(r.mae))
    .sort((a, b) => a.mae - b.mae);
  const gnnRank = ranking.findIndex((r) => r.name === GNN) + 1;
  const gnn = ranking.find((r) => r.name === GNN);

  const families = d.external_cohort.workload_families;
  const familyRows = families.map((f) => {
    const avg = (model: string) =>
      runs.reduce((s, r) => s + (r.by_family[model]?.[f]?.mae ?? 0), 0) / Math.max(runs.length, 1);
    return { family: pretty(f), GNN: +avg(GNN).toFixed(4), "Random Forest": +avg(RF).toFixed(4) };
  });

  return (
    <div className="space-y-4" data-testid="crash-benchmarks">
      <Panel
        title="Benchmark summary"
        subtitle="Frozen external cohort · simulator-derived, not ext4 measurements"
      >
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <StatCard
            label="External cases"
            value={d.external_cohort.summary.count.toLocaleString()}
            hint="six held-out families"
            icon={Database}
          />
          <StatCard
            label="GNN MAE"
            value={gnn ? gnn.mae.toFixed(4) : "—"}
            hint="to simulator oracle, lower is better"
            icon={Target}
          />
          <StatCard
            label="Rank (MAE)"
            value={gnnRank ? `#${gnnRank} / ${ranking.length}` : "—"}
            hint="vs. baselines"
            icon={Trophy}
          />
          <StatCard
            label="Spearman ρ"
            value={gnn ? gnn.rho.toFixed(3) : "—"}
            hint="risk ranking agreement"
            icon={ChartBar}
          />
          <StatCard
            label="Training seeds"
            value={d.registered_config.training_seeds.length}
            hint="independent fits"
            icon={BrainCircuit}
          />
          <StatCard
            label="Families"
            value={families.length}
            hint={`${d.external_cohort.cases_per_family} cases each`}
            icon={Layers}
          />
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel
          title="Error by workload family"
          subtitle="MAE to oracle, mean of five seeds · GNN vs Random Forest"
        >
          <div className="h-72" data-testid="family-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={familyRows} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="family"
                  tick={{ fill: "var(--muted-foreground)", fontSize: 10 }}
                  interval={0}
                  angle={-12}
                  height={50}
                />
                <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "var(--popover)", border: "2px solid var(--ink)" }}
                />
                <Legend />
                <Bar dataKey="GNN" fill="var(--chart-1)" />
                <Bar dataKey="Random Forest" fill="var(--chart-3)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel
          title="Model ranking"
          subtitle="All six models, ordered by MAE (mean ± SD over seeds)"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="ranking-table">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="p-2">#</th>
                  <th className="p-2">Model</th>
                  <th className="p-2">MAE ↓</th>
                  <th className="p-2">SD</th>
                  <th className="p-2">ρ ↑</th>
                  <th className="p-2">Top-10% ↑</th>
                </tr>
              </thead>
              <tbody>
                {ranking.map((r, i) => (
                  <tr
                    key={r.name}
                    className={
                      r.name === GNN ? "bg-primary/10 font-semibold" : "border-b border-border/60"
                    }
                  >
                    <td className="p-2">{i + 1}</td>
                    <td className="p-2">{short(r.name)}</td>
                    <td className="p-2 tabular">{r.mae.toFixed(4)}</td>
                    <td className="p-2 tabular">{r.sd.toFixed(4)}</td>
                    <td className="p-2 tabular">{r.rho.toFixed(3)}</td>
                    <td className="p-2 tabular">{r.top10.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel
        title="Seed comparison"
        subtitle="Per-seed GNN against Random Forest on the same external cohort"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="seed-table">
            <thead>
              <tr className="border-b border-border text-left text-muted-foreground">
                <th className="p-2">Seed</th>
                <th className="p-2">Best val MAE</th>
                <th className="p-2">Epoch</th>
                <th className="p-2">GNN MAE</th>
                <th className="p-2">RF MAE</th>
                <th className="p-2">Δ (RF − GNN)</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const g = r.models[GNN]?.mae_to_oracle ?? NaN;
                const f = r.models[RF]?.mae_to_oracle ?? NaN;
                return (
                  <tr key={r.training_seed} className="border-b border-border/60">
                    <td className="p-2">{r.training_seed}</td>
                    <td className="p-2 tabular">{r.best_validation_mae.toFixed(4)}</td>
                    <td className="p-2 tabular">{r.selected_epoch}</td>
                    <td className="p-2 tabular">{g.toFixed(4)}</td>
                    <td className="p-2 tabular">{f.toFixed(4)}</td>
                    <td className="p-2 tabular">{(f - g).toFixed(4)}</td>
                  </tr>
                );
              })}
              {gnn && (
                <tr className="font-semibold">
                  <td className="p-2">mean</td>
                  <td className="p-2" />
                  <td className="p-2" />
                  <td className="p-2 tabular">{gnn.mae.toFixed(4)}</td>
                  <td className="p-2 tabular">
                    {(ranking.find((r) => r.name === RF)?.mae ?? NaN).toFixed(4)}
                  </td>
                  <td className="p-2 tabular">
                    {((ranking.find((r) => r.name === RF)?.mae ?? NaN) - gnn.mae).toFixed(4)}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

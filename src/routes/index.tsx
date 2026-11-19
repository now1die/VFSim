import { MultiseedPanel } from "@/components/vfsim/MultiseedPanel";
import { LiveOperations } from "@/components/vfsim/LiveOperations";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  GitBranch,
  Layers,
  ListTree,
  BrainCircuit,
  ShieldAlert,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

import { Button } from "@/components/ui/button";
import {
  EmptyState,
  ErrorState,
  LoadingState,
  Panel,
  RiskBadge,
  StatCard,
} from "@/components/vfsim/primitives";
import { useSelection } from "@/components/vfsim/selection";
import { OverviewFilesystem } from "@/components/vfsim/OverviewFilesystem";
import { api } from "@/services/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
      { title: "VFSim Overview — Risk-Aware Filesystem Simulator" },
      {
        name: "description",
        content:
          "Live overview of simulated filesystem health: graph size, dirty nodes, open transactions and GNN-predicted risk.",
      },
      { property: "og:title", content: "VFSim Overview" },
      {
        property: "og:description",
        content: "Graph size, dirty nodes, open transactions and GNN-predicted filesystem risk.",
      },
    ],
  }),
  component: Overview,
});

const skylineDestinations = [
  { to: "/graph", label: "Graph Explorer", building: "graph", icon: GitBranch },
  { to: "/transactions", label: "Transactions", building: "transactions", icon: ListTree },
  { to: "/risk", label: "Risk Analysis", building: "risk", icon: ShieldAlert },
  { to: "/simulation", label: "Crash Lab", building: "crash", icon: Zap },
  { to: "/model", label: "GNN Model", building: "model", icon: BrainCircuit },
] as const;

const chartAxis = { stroke: "var(--muted-foreground)", fontSize: 11 };

function OriginalOverview() {
  const { setSelectedTxId, riskThreshold } = useSelection();
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => api.getStats() });
  const risk = useQuery({ queryKey: ["risk-history"], queryFn: () => api.getRiskHistory() });
  const txs = useQuery({ queryKey: ["transactions"], queryFn: () => api.getTransactions() });
  const graph = useQuery({ queryKey: ["graph", null], queryFn: () => api.getGraph() });

  return (
    <div className="space-y-6">
      <section className="overview-hero" aria-label="VFSim research console">
        <div className="pixel-depth-frame"><div className="pixel-window overview-hero-copy">
          <div className="pixel-window-title"><span>VFSim Research Console</span><span className="pixel-window-controls" aria-hidden="true"><span /><span /><span /></span></div>
          <div className="pixel-window-body">
            <h1 className="overview-brand-title">VFSim</h1>
            <p className="overview-subtitle">Graph-powered filesystem crash-risk research</p>
            <div className="flex flex-wrap gap-3">
              <Button asChild className="pixel-button h-12 px-5"><Link to="/graph"><GitBranch className="size-4" aria-hidden="true" />Explore graph</Link></Button>
              <Button asChild variant="outline" className="pixel-button pixel-button-secondary h-12 px-5"><Link to="/simulation"><Zap className="size-4" aria-hidden="true" />Run simulation</Link></Button>
            </div>
          </div>
        </div>
        </div>
        <div className="pixel-depth-frame"><div className="pixel-window overview-skyline" aria-label="Pixel sunset over a city">
          <div className="pixel-window-title"><span>Research environment</span><span className="pixel-window-controls" aria-hidden="true"><span /><span /><span /></span></div>
          <nav className="overview-skyline-scene" aria-label="Explore VFSim sections from the city skyline">
            <div className="scene-backdrop" aria-hidden="true">
              <span className="scene-sun" />
              <span className="scene-cloud scene-cloud-one" />
              <span className="scene-cloud scene-cloud-two" />
              <span className="scene-cloud scene-cloud-three" />
              <span className="scene-mountain scene-mountain-back" />
              <span className="scene-mountain scene-mountain-front" />
              <span className="scene-city-base" />
            </div>
            {skylineDestinations.map(({ to, label, building, icon: Icon }) => (
              <Link key={to} to={to} className={`scene-city scene-city-${building}`} aria-label={`Open ${label}`}>
                <span className="scene-city-sign" aria-hidden="true"><Icon /></span>
                <span className="scene-city-tooltip" aria-hidden="true">{label}</span>
              </Link>
            ))}
          </nav>
        </div></div>
      </section>

      {stats.isPending ? (
        <LoadingState label="Fetching simulator stats" rows={2} />
      ) : stats.isError || !stats.data ? (
        <ErrorState onRetry={() => void stats.refetch()} />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label="Graph size"
            value={`${stats.data.nodes}/${stats.data.edges}`}
            hint="nodes / edges in the FS graph"
            icon={Layers}
          />
          <StatCard
            label="Dirty nodes"
            value={stats.data.dirtyNodes}
            hint="unflushed pages awaiting fsync"
            icon={AlertTriangle}
            tone="watch"
          />
          <StatCard
            label="Open transactions"
            value={stats.data.openTransactions}
            hint="in-flight, not yet committed"
            icon={Activity}
          />
          <StatCard
            label="Mean predicted risk"
            value={stats.data.meanRisk == null ? "—" : stats.data.meanRisk.toFixed(3)}
            hint={
              stats.data.journalUtilization == null
                ? "Average of recorded transaction risk predictions"
                : `journal at ${(stats.data.journalUtilization * 100).toFixed(0)}% utilization`
            }
            icon={ShieldAlert}
            tone="critical"
          />
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.9fr)]">
        <Panel title="Filesystem structure" subtitle="Live graph topology · directories, files, and inode links" icon={GitBranch}>
          {graph.isPending ? <LoadingState label="Loading filesystem graph" rows={3} /> : graph.isError || !graph.data ? <ErrorState message="Filesystem graph is unavailable." onRetry={() => void graph.refetch()} /> : graph.data.nodes.length === 0 ? <EmptyState title="Filesystem graph is empty" /> : <OverviewFilesystem nodes={graph.data.nodes} edges={graph.data.edges} />}
        </Panel>
        <Panel title="Crash-risk trend" subtitle="Predicted risk; observed outcome appears only when measured" icon={ShieldAlert}>
          {risk.isPending ? (
            <LoadingState label="Loading risk history" rows={3} />
          ) : risk.isError || !risk.data ? (
            <ErrorState onRetry={() => void risk.refetch()} />
          ) : risk.data.length === 0 ? (
            <EmptyState title="No risk samples recorded" />
          ) : (
            <div className="h-[430px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={risk.data} margin={{ left: -20, right: 8, top: 8 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                  <XAxis dataKey="t" {...chartAxis} tickLine={false} />
                  <YAxis domain={[0, 1]} {...chartAxis} tickLine={false} />
                  <ReferenceLine y={riskThreshold} stroke="var(--orange)" strokeDasharray="5 4" />
                  <Tooltip
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 0,
                      fontSize: 12,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar
                    dataKey="predicted"
                    stroke="var(--chart-1)"
                    fill="var(--chart-1)"
                    name="GNN predicted"
                  />
                  <Bar dataKey="observed" fill="var(--orange)" name="Observed outcome (when available)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel title="Recent transactions" subtitle="click to make active" icon={Activity}>
          {txs.isPending ? (
            <LoadingState label="Loading transactions" rows={4} />
          ) : txs.isError || !txs.data ? (
            <ErrorState onRetry={() => void txs.refetch()} />
          ) : txs.data.length === 0 ? (
            <EmptyState title="No transactions replayed yet" />
          ) : (
            <ul className="space-y-2">
              {txs.data.map((tx) => (
                <li key={tx.id}>
                  <Link
                    to="/transactions"
                    onClick={() => setSelectedTxId(tx.id)}
                    className="flex items-center justify-between gap-2 rounded-md border border-border px-3 py-2 transition-colors hover:border-primary/50 hover:bg-surface-2"
                  >
                    <span>
                      <span className="tabular block text-xs text-foreground">{tx.id}</span>
                      <span className="text-[11px] text-muted-foreground">{tx.label}</span>
                    </span>
                    <RiskBadge level={tx.riskLevel} score={tx.riskScore} />
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Filesystem research" subtitle="About the simulator and its model" icon={Layers}>
          <Accordion type="multiple" className="faq-list">
            <AccordionItem value="hard-links"><AccordionTrigger>What is a hard link?</AccordionTrigger><AccordionContent>A hard link is another directory entry pointing to the same inode. Both names share the inode and its file data.</AccordionContent></AccordionItem>
            <AccordionItem value="faultlab"><AccordionTrigger>How does FaultLab work?</AccordionTrigger><AccordionContent>FaultLab evaluates saved model checkpoints against simulated filesystem workloads and a simulator-derived recovery oracle. These results are not Linux ext4 measurements.</AccordionContent></AccordionItem>
            <AccordionItem value="attention"><AccordionTrigger>What does GNN attention show?</AccordionTrigger><AccordionContent>Attention values describe learned message-passing weights inside a selected model checkpoint. They are not proof of causal explanations.</AccordionContent></AccordionItem>
          </Accordion>
        </Panel>
      </div>
    </div>
  );
}

function Overview() {
  return (
    <div className="space-y-6">
      <OriginalOverview />
      <LiveOperations />
      <section aria-label="Research evidence" className="space-y-3">
        <div className="border-b-2 border-border pb-3">
          <h2 className="text-xl font-semibold text-foreground">VFSim research snapshot</h2>
          <p className="mt-1 text-sm text-muted-foreground">Five-seed, six-family simulator-based evaluation from saved v0.9 experiments. Not Linux ext4 measurements, and not live crash outcomes from your current session.</p>
        </div>
        <MultiseedPanel compact />
      </section>
    </div>
  );
}

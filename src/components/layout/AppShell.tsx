import { Link } from "@tanstack/react-router";
import {
  BrainCircuit,
  GitBranch,
  LayoutDashboard,
  Layers,
  ListTree,
  Settings,
  ShieldAlert,
  Zap,
} from "lucide-react";
import type { ReactNode } from "react";

import { BackendStatusBadge } from "@/components/vfsim/BackendStatus";
import { PixelLandscapeBackground } from "@/components/vfsim/PixelLandscapeBackground";
import { TransactionPicker } from "@/components/vfsim/TransactionPicker";

const nav = [
  { to: "/", label: "Overview", icon: LayoutDashboard, exact: true },
  { to: "/graph", label: "Graph Explorer", icon: GitBranch, exact: false },
  { to: "/transactions", label: "Transactions", icon: ListTree, exact: false },
  { to: "/risk", label: "Risk Analysis", icon: ShieldAlert, exact: false },
  { to: "/simulation", label: "Crash Lab", icon: Zap, exact: false },
  { to: "/model", label: "GNN Model", icon: BrainCircuit, exact: false },
  { to: "/settings", label: "Settings", icon: Settings, exact: false },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <PixelLandscapeBackground />
      <header className="app-header">
        <Link to="/" className="brand-mark" aria-label="VFSim overview">
          <span className="brand-icon" aria-hidden="true">
            <Layers fill="var(--header-logo-fill)" />
          </span>
          <span className="brand-name">VFSim</span>
        </Link>
        <nav aria-label="Main navigation" className="main-nav">
          {nav.map(({ to, label, icon: Icon, exact }) => (
            <Link key={to} to={to} activeOptions={{ exact }} className="nav-tab">
              <Icon className="size-4" aria-hidden="true" />
              <span>{label}</span>
            </Link>
          ))}
        </nav>
        <div className="header-status">
          <div className="header-transaction header-pixel-frame">
            <TransactionPicker />
          </div>
          <div className="header-status-frame header-pixel-frame"><BackendStatusBadge /></div>
        </div>
      </header>
      <main className="app-content">{children}</main>
      <footer className="app-footer">
        VFSim · Filesystem simulation and model processing remain on the research service.
      </footer>
    </div>
  );
}
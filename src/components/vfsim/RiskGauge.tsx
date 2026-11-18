import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/services/types";

const levelColor = {
  safe: "var(--safe)",
  watch: "var(--watch)",
  elevated: "var(--elevated)",
  critical: "var(--critical)",
  unknown: "var(--muted-foreground)",
} as const;

export function RiskGauge({
  score,
  threshold,
  size = 180,
}: {
  score: number | null;
  threshold: number;
  size?: number;
}) {
  const level: RiskLevel | "unknown" =
    score == null ? "unknown" : score >= 0.75 ? "critical" : score >= 0.5 ? "elevated" : score >= 0.25 ? "watch" : "safe";
  const radius = size / 2 - 14;
  const circumference = Math.PI * radius;
  const dash = circumference * (score == null ? 0 : Math.min(1, Math.max(0, score)));
  const thresholdAngle = 180 * Math.min(1, Math.max(0, threshold));

  return (
    <div className="flex flex-col items-center" role="img" aria-label={score == null ? "Risk score unavailable" : `Risk score ${score == null ? "—" : score.toFixed(2)}, ${level}`}>
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        <path
          d={`M 14 ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 14} ${size / 2}`}
          fill="none"
          stroke="var(--surface-2)"
          strokeWidth={12}
          strokeLinecap="round"
        />
        <path
          d={`M 14 ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 14} ${size / 2}`}
          fill="none"
          stroke={levelColor[level]}
          strokeWidth={12}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          className="transition-all duration-500"
        />
        <line
          x1={size / 2}
          y1={size / 2}
          x2={size / 2 + radius * Math.cos(Math.PI - (thresholdAngle * Math.PI) / 180)}
          y2={size / 2 - radius * Math.sin(Math.PI - (thresholdAngle * Math.PI) / 180)}
          stroke="var(--foreground)"
          strokeWidth={2}
          strokeDasharray="3 3"
        />
      </svg>
      <p className="tabular -mt-6 text-3xl font-semibold" style={{ color: levelColor[level] }}>
        {score == null ? "—" : score.toFixed(2)}
      </p>
      <p className={cn("text-xs uppercase tracking-widest text-muted-foreground")}>
        {level} · threshold {threshold.toFixed(2)}
      </p>
    </div>
  );
}

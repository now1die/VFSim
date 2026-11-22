import { useQuery } from "@tanstack/react-query";
import { Radio } from "lucide-react";

import { api } from "@/services/api";
import { cn } from "@/lib/utils";

export function BackendStatusBadge({ className }: { className?: string }) {
  const { data, isError, isPending } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.getHealth(),
    refetchInterval: 30_000,
  });

  const Icon = Radio;

  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wider",
        isError
          ? "border-critical/50 bg-critical/15 text-critical"
          : isPending
            ? "border-watch/40 bg-watch/10 text-watch"
            : "border-live/50 bg-live/15 text-live",
        className,
      )}
      role="status"
    >
      <Icon className="size-3.5" aria-hidden />
        {isError ? "Backend offline" : isPending ? "Connecting…" : "Backend live"}
      <span className="hidden font-normal normal-case tracking-normal text-muted-foreground sm:inline">
        {isError ? "Unavailable" : (data?.version ?? "connecting…")}
      </span>
    </div>
  );
}


import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { FilePlus2, HardDrive } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Panel } from "@/components/vfsim/primitives";
import { useSelection } from "@/components/vfsim/selection";
import { api } from "@/services/api";

const operations = [
  ["create_file", "Create file"],
  ["create_directory", "Create directory"],
  ["write", "Write data"],
  ["link", "Create hard link"],
  ["rename", "Rename / move"],
  ["unlink", "Unlink name"],
  ["fsync", "Synchronize"],
] as const;
type Action = (typeof operations)[number][0];

export function LiveOperations() {
  const queryClient = useQueryClient();
  const { setSelectedTxId } = useSelection();
  const [action, setAction] = useState<Action>("create_file");
  const [path, setPath] = useState("/report.txt");
  const [secondPath, setSecondPath] = useState("/report_link.txt");
  const [size, setSize] = useState("256");
  const [notice, setNotice] = useState("");

  const operation = useMutation({
    mutationFn: async () => {
      const target = path.trim();
      if (!target.startsWith("/")) throw new Error("Use an absolute path, for example /report.txt");
      if (action === "create_file") return api.createFile(target, "file");
      if (action === "create_directory") return api.createFile(target, "directory");
      if (action === "write") {
        const bytes = Number(size);
        if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Bytes must be a nonnegative integer");
        return api.writeFile(target, bytes);
      }
      if (action === "link" || action === "rename") {
        const destination = secondPath.trim();
        if (!destination.startsWith("/")) throw new Error("The second path must also start with /");
        return action === "link" ? api.linkFile(target, destination) : api.renameFile(target, destination);
      }
      if (action === "unlink") return api.unlinkFile(target);
      return api.syncFile(target);
    },
    onSuccess: async (transaction) => {
      setSelectedTxId(transaction.id);
      setNotice(`Recorded ${transaction.label} (transaction ${transaction.id})`);
      await queryClient.invalidateQueries();
    },
    onError: (error) => {
      setNotice(`Operation failed: ${error instanceof Error ? error.message : String(error)}`);
    },
  });

  const needsSecond = action === "link" || action === "rename";
  const isWrite = action === "write";
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    operation.mutate();
  }

  return (
    <Panel title="Live virtual filesystem operations" subtitle="Execute your own operations against the connected VFSim backend" icon={HardDrive}>
      <form className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)_auto] lg:items-end" onSubmit={handleSubmit}>
        <label className="flex min-w-0 flex-col gap-1 text-sm font-semibold">
          Operation
          <select className="h-10 min-w-0 border-2 border-foreground bg-card px-2 text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring" value={action} onChange={(e) => { setAction(e.target.value as Action); setNotice(""); }}>
            {operations.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-sm font-semibold">
          {action === "link" ? "Existing file path" : action === "rename" ? "Old path" : "Absolute path"}
          <Input className="h-10 border-2 border-foreground bg-card" value={path} onChange={(e) => setPath(e.target.value)} autoComplete="off" required aria-label="Filesystem path" />
        </label>
        <Button className="pixel-button h-10" type="submit" disabled={operation.isPending}>
          <FilePlus2 className="size-4" aria-hidden="true" />
          {operation.isPending ? "Executing…" : "Execute operation"}
        </Button>
        {needsSecond && (
          <label className="flex min-w-0 flex-col gap-1 text-sm font-semibold lg:col-span-2">
            {action === "link" ? "New hard-link name" : "New path"}
            <Input className="h-10 border-2 border-foreground bg-card" value={secondPath} onChange={(e) => setSecondPath(e.target.value)} autoComplete="off" required aria-label="Destination path" />
          </label>
        )}
        {isWrite && (
          <label className="flex min-w-0 flex-col gap-1 text-sm font-semibold lg:col-span-2">
            Bytes to write
            <Input className="h-10 border-2 border-foreground bg-card" type="number" min={0} step={1} value={size} onChange={(e) => setSize(e.target.value)} required />
          </label>
        )}
      </form>
      <p className="mt-3 text-xs text-muted-foreground">These are real operations in VFSim's in-memory virtual filesystem, not scripted examples or physical-disk writes. Start by creating a file; then select Write, Hard link, or Fsync. Its state resets when the backend process restarts.</p>
      {notice && <p className="mt-2 break-words text-sm text-foreground" role="status" aria-live="polite">{notice}</p>}
    </Panel>
  );
}

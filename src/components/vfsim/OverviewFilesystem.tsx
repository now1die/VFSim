import { Link } from "@tanstack/react-router";
import { Database, File, Folder } from "lucide-react";
import { useId } from "react";

import type { FsEdge, FsNode } from "@/services/types";

const nodeHeight = 34;
const columnWidth = 154;
const rowHeight = 76;
const top = 18;

interface PositionedNode {
  id: string;
  label: string;
  path: string;
  kind: "directory" | "file" | "shared";
  x: number;
  y: number;
}

interface Branch {
  parent: PositionedNode;
  children: PositionedNode[];
}

function layoutFilesystem(nodes: FsNode[], edges: FsEdge[]) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const root = nodes.find((node) => node.kind === "directory" && node.path === "/");
  if (!root) return null;

  const entriesByDirectory = new Map<string, FsNode[]>();
  const targetByEntry = new Map<string, FsNode>();
  for (const edge of edges) {
    const target = byId.get(edge.target);
    if (!target) continue;
    if (edge.relation === "contains") {
      const entries = entriesByDirectory.get(edge.source) ?? [];
      entries.push(target);
      entriesByDirectory.set(edge.source, entries);
    }
    if (edge.relation === "points-to") targetByEntry.set(edge.source, target);
  }

  const namespace: PositionedNode[] = [];
  const branches: Branch[] = [];
  const filesByInode = new Map<FsNode, PositionedNode[]>();
  let columns = 0;
  let maxDepth = 0;

  function place(entry: FsNode, inode: FsNode, depth: number): PositionedNode {
    const node: PositionedNode = {
      id: entry.id,
      label: depth === 1 && inode.kind === "directory" ? entry.path : entry.name,
      path: entry.path,
      kind: inode.kind === "directory" ? "directory" : "file",
      x: 0,
      y: top + depth * rowHeight,
    };
    namespace.push(node);
    maxDepth = Math.max(maxDepth, depth);

    const entries =
      node.kind === "directory"
        ? [...(entriesByDirectory.get(inode.id) ?? [])].sort((a, b) => a.name.localeCompare(b.name))
        : [];
    const children = entries.map((child) =>
      place(child, targetByEntry.get(child.id) ?? child, depth + 1),
    );
    const first = children[0];
    const last = children.at(-1);
    if (first && last) {
      node.x = (first.x + last.x) / 2;
      branches.push({ parent: node, children });
    } else {
      node.x = (columns + 0.5) * columnWidth;
      columns += 1;
    }

    if (node.kind === "file" && inode.kind === "file") {
      const files = filesByInode.get(inode) ?? [];
      files.push(node);
      filesByInode.set(inode, files);
    }
    return node;
  }

  place(root, root, 0);

  const sharedInodes: { inode: PositionedNode; files: PositionedNode[] }[] = [];
  const namespaceBottom = top + maxDepth * rowHeight + nodeHeight;
  for (const [inode, files] of filesByInode) {
    const first = files[0];
    const last = files.at(-1);
    if (files.length < 2 || !first || !last) continue;
    sharedInodes.push({
      inode: {
        id: inode.id,
        label: inode.inodeNumber === undefined ? "Shared inode" : `Inode ${inode.inodeNumber}`,
        path: "",
        kind: "shared",
        x: (first.x + last.x) / 2,
        y: namespaceBottom + 64 + sharedInodes.length * rowHeight,
      },
      files,
    });
  }

  return {
    namespace,
    branches,
    sharedInodes,
    width: columns * columnWidth,
    height: (sharedInodes.at(-1)?.inode.y ?? namespaceBottom - nodeHeight) + nodeHeight + top,
  };
}

function FilesystemNode({ node, linked = false }: { node: PositionedNode; linked?: boolean }) {
  const Icon = node.kind === "directory" ? Folder : node.kind === "shared" ? Database : File;
  return (
    <div
      className={`overview-fs-node overview-fs-node-${node.kind}`}
      style={{ left: node.x, top: node.y, height: nodeHeight }}
      title={node.path || "Shared inode for the connected filenames"}
      data-path={node.path || undefined}
      data-linked={linked || undefined}
    >
      <Icon size={16} aria-hidden="true" />
      <span>{node.label}</span>
    </div>
  );
}

export function OverviewFilesystem({ nodes, edges }: { nodes: FsNode[]; edges: FsEdge[] }) {
  const arrowId = useId();
  const layout = layoutFilesystem(nodes, edges);
  if (!layout)
    return <p className="text-sm text-muted-foreground">The root directory is unavailable.</p>;

  const linkedIds = new Set(layout.sharedInodes.flatMap(({ files }) => files.map(({ id }) => id)));

  return (
    <figure className="overview-fs">
      <div
        className="overview-fs-scroll"
        tabIndex={0}
        role="region"
        aria-label="Filesystem diagram"
      >
        <div className="overview-fs-canvas" style={{ width: layout.width, height: layout.height }}>
          <svg
            className="overview-fs-edges"
            width={layout.width}
            height={layout.height}
            aria-hidden="true"
          >
            <defs>
              <marker
                id={arrowId}
                viewBox="0 0 8 8"
                markerWidth="6"
                markerHeight="6"
                refX="7"
                refY="4"
                orient="auto"
              >
                <path d="M 1 1 L 7 4 L 1 7" fill="none" stroke="context-stroke" strokeWidth="1.5" />
              </marker>
            </defs>
            {layout.branches.map(({ parent, children }) => {
              const start = parent.y + nodeHeight;
              const rail = start + 21;
              const first = children[0]!;
              const last = children.at(-1)!;
              return (
                <g key={parent.id}>
                  <path d={`M ${parent.x} ${start} V ${rail} M ${first.x} ${rail} H ${last.x}`} />
                  {children.map((child) => (
                    <path
                      key={child.id}
                      d={`M ${child.x} ${rail} V ${child.y - 3}`}
                      markerEnd={`url(#${arrowId})`}
                    />
                  ))}
                </g>
              );
            })}
            {layout.sharedInodes.map(({ inode, files }) => (
              <g className="overview-fs-link" key={inode.id}>
                {files.map((file) => (
                  <path
                    key={file.id}
                    d={`M ${file.x} ${file.y + nodeHeight} V ${inode.y - 20} H ${inode.x}`}
                  />
                ))}
                <path
                  d={`M ${inode.x} ${inode.y - 20} V ${inode.y - 3}`}
                  markerEnd={`url(#${arrowId})`}
                />
              </g>
            ))}
          </svg>
          {layout.branches.map(({ parent, children }) => (
            <span
              key={parent.id}
              className="overview-fs-edge-label"
              style={{
                left: children.length === 1 ? parent.x + 40 : parent.x,
                top: parent.y + nodeHeight + 21,
              }}
              aria-hidden="true"
            >
              contains
            </span>
          ))}
          {layout.namespace.map((node) => (
            <FilesystemNode key={node.id} node={node} linked={linkedIds.has(node.id)} />
          ))}
          {layout.sharedInodes.map(({ inode }) => (
            <FilesystemNode key={inode.id} node={inode} />
          ))}
        </div>
      </div>
      {layout.sharedInodes.map(({ inode, files }) => (
        <span className="sr-only" key={inode.id}>
          {files.map((file) => file.path).join(", ")} share {inode.label}.
        </span>
      ))}
      <figcaption className="overview-fs-caption">
        {layout.sharedInodes.length > 0 && (
          <span>
            <i aria-hidden="true" /> Hard links share an inode
          </span>
        )}
        <Link to="/graph">
          Open Graph Explorer <span aria-hidden="true">→</span>
        </Link>
      </figcaption>
    </figure>
  );
}

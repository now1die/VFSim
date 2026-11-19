import { ArrowRight, Database, File, Folder, Maximize, Minus, Plus } from "lucide-react";
import { useId, useRef, useState } from "react";

import type { FsEdge, FsNode } from "@/services/types";

const nodeHeight = 47;
const columnWidth = 180;
const rowHeight = 91;
const minimumZoom = 0.25;
const maximumZoom = 2;
const top = 18;

interface PositionedNode {
  id: string;
  label: string;
  inodeNumber?: number | undefined;
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
      inodeNumber: inode.inodeNumber,
      kind: inode.kind === "directory" ? "directory" : "file",
      x: 0,
      y: top + depth * rowHeight,
    };
    namespace.push(node);
    maxDepth = Math.max(maxDepth, depth);

    const entries = node.kind === "directory" ? (entriesByDirectory.get(inode.id) ?? []) : [];
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

  const sharedInodes: {
    inode: PositionedNode;
    files: PositionedNode[];
    rail: number;
    primaryPath: string;
  }[] = [];
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
        y: namespaceBottom + 55 + sharedInodes.length * rowHeight,
      },
      files,
      primaryPath: files.find((file) => file.path === inode.path)?.path ?? first.path,
      rail: namespaceBottom + 30 + sharedInodes.length * rowHeight,
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

function FilesystemNode({
  node,
  selected,
  onSelect,
  linkStyle,
}: {
  node: PositionedNode;
  selected: boolean;
  onSelect: () => void;
  linkStyle?: "first" | "alias" | undefined;
}) {
  const Icon = node.kind === "directory" ? Folder : node.kind === "shared" ? Database : File;
  return (
    <button
      type="button"
      className={`overview-fs-node overview-fs-node-${node.kind}`}
      style={{ left: node.x, top: node.y, height: nodeHeight }}
      title={node.path || "Shared inode for the connected filenames"}
      data-path={node.path || undefined}
      data-long-name={node.label.length > 14 || undefined}
      data-linked={linkStyle}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span className="overview-fs-node-name">
        <Icon size={15} aria-hidden="true" />
        <span>{node.label}</span>
      </span>
      {node.inodeNumber !== undefined && <small>Inode {node.inodeNumber}</small>}
    </button>
  );
}

export function OverviewFilesystem({ nodes, edges }: { nodes: FsNode[]; edges: FsEdge[] }) {
  const arrowId = useId();
  const viewport = useRef<HTMLDivElement>(null);
  const [selectedNode, select] = useState("/");
  const [zoom, setZoom] = useState(1);
  const layout = layoutFilesystem(nodes, edges);
  if (!layout)
    return <p className="text-sm text-muted-foreground">The root directory is unavailable.</p>;

  const allNodes = [...layout.namespace, ...layout.sharedInodes.map(({ inode }) => inode)];
  const activeNode = allNodes.some(({ id, path }) => (path || id) === selectedNode)
    ? selectedNode
    : "/";
  const linkedIds = new Map(
    layout.sharedInodes.flatMap(({ files, primaryPath }) =>
      files.map(({ id, path }) => [id, path === primaryPath ? "first" : "alias"] as const),
    ),
  );
  const edgeCount = layout.branches.reduce(
    (count, { children }) => count + children.length,
    linkedIds.size,
  );
  const typeCount = new Set(allNodes.map(({ kind }) => kind)).size;

  function changeZoom(factor: number) {
    setZoom((current) => Math.min(maximumZoom, Math.max(minimumZoom, current * factor)));
  }

  function fitGraph() {
    const area = viewport.current;
    if (!area || !layout) return;
    setZoom(
      Math.max(
        minimumZoom,
        Math.min(1, (area.clientWidth - 12) / layout.width, area.clientHeight / layout.height),
      ),
    );
    area.scrollTo(0, 0);
  }

  return (
    <figure className="overview-fs">
      <div
        className="overview-fs-scroll"
        ref={viewport}
        tabIndex={0}
        role="region"
        aria-label="Filesystem diagram"
      >
        <div
          className="overview-fs-canvas"
          style={{ width: layout.width, height: layout.height, zoom }}
        >
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
                <path d="M 0 0 L 8 4 L 0 8 Z" fill="context-stroke" stroke="none" />
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
            {layout.sharedInodes.map(({ inode, files, rail }) => (
              <g className="overview-fs-link" key={inode.id}>
                {files.map((file) => (
                  <path
                    key={file.id}
                    d={`M ${file.x} ${file.y + nodeHeight} V ${rail} H ${inode.x}`}
                  />
                ))}
                <path d={`M ${inode.x} ${rail} V ${inode.y - 3}`} markerEnd={`url(#${arrowId})`} />
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
          {layout.sharedInodes.map(({ inode, rail }) => (
            <span
              key={inode.id}
              className="overview-fs-edge-label overview-fs-hardlink-label"
              style={{ left: inode.x, top: rail }}
              aria-hidden="true"
            >
              hard link
            </span>
          ))}
          {allNodes.map((node) => (
            <FilesystemNode
              key={node.id}
              node={node}
              selected={activeNode === (node.path || node.id)}
              onSelect={() => select(node.path || node.id)}
              linkStyle={linkedIds.get(node.id)}
            />
          ))}
        </div>
      </div>
      {layout.sharedInodes.map(({ inode, files }) => (
        <span className="sr-only" key={inode.id}>
          {files.map((file) => file.path).join(", ")} share {inode.label}.
        </span>
      ))}
      <figcaption className="overview-fs-caption">
        <div className="overview-fs-legend">
          <span className="overview-fs-caption-label">Legend</span>
          <span>
            <Folder size={14} className="overview-fs-folder-key" aria-hidden="true" /> Directory
          </span>
          <span>
            <File size={14} aria-hidden="true" /> File
          </span>
          <span>
            <ArrowRight size={14} aria-hidden="true" /> Contains
          </span>
          <span>
            <i className="overview-fs-dashed-key" aria-hidden="true" /> Hard link
          </span>
          <span>
            <i className="overview-fs-selection-key" aria-hidden="true" /> Selected
          </span>
        </div>
        <div className="overview-fs-status">
          <dl>
            <div>
              <dt>Nodes</dt>
              <dd>{allNodes.length}</dd>
            </div>
            <div>
              <dt>Edges</dt>
              <dd>{edgeCount}</dd>
            </div>
            <div>
              <dt>Types</dt>
              <dd>{typeCount}</dd>
            </div>
          </dl>
          <div className="overview-fs-controls" role="group" aria-label="Graph zoom controls">
            <button type="button" title="Fit graph" aria-label="Fit graph" onClick={fitGraph}>
              <Maximize size={15} />
            </button>
            <button
              type="button"
              title="Zoom out"
              aria-label="Zoom out"
              disabled={zoom <= minimumZoom}
              onClick={() => changeZoom(0.8)}
            >
              <Minus size={15} />
            </button>
            <button
              type="button"
              title="Zoom in"
              aria-label="Zoom in"
              disabled={zoom >= maximumZoom}
              onClick={() => changeZoom(1.25)}
            >
              <Plus size={15} />
            </button>
          </div>
        </div>
      </figcaption>
    </figure>
  );
}

import "@xyflow/react/dist/style.css";
import {
  Background, BackgroundVariant, Controls, Handle, MiniMap, Position,
  ReactFlow, type Node, type Edge, type NodeProps,
} from "@xyflow/react";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BrainCircuit, GitBranch, Layers, Search } from "lucide-react";
import { ErrorState, LoadingState, Panel } from "@/components/vfsim/primitives";
import { researchApi, faultlabModels, faultlabFamilies, type GnnNode, type FaultNode } from "@/services/research";

const palette: Record<string,string> = {
  inode:"var(--node-inode)", dirent:"var(--node-dirent)", fd:"var(--node-fd)", link:"var(--node-link)", rename:"var(--node-rename)",
  constraint:"var(--node-constraint)", path:"var(--node-path)", object:"var(--node-object)", operation:"var(--node-operation)", sync:"var(--node-sync)", writeback:"var(--node-writeback)"
};

interface ResearchCard {
  [key: string]: unknown;
  id: string;
  label: string;
  kind: string;
  detail?: string | undefined;
  touched?: boolean | undefined;
  selected?: boolean;
}
function GraphCard({data}:NodeProps) {
  const d = data as unknown as ResearchCard;
  const color = palette[d.kind] ?? "var(--muted-foreground)";
  return <div className="relative w-[176px] border-2 border-border bg-card p-2.5 text-foreground shadow-md" style={{borderColor:d.selected?"var(--orange)":undefined,boxShadow:d.selected?"0 0 0 2px var(--orange)":undefined}}>
    <Handle type="target" position={Position.Top} className="!bg-primary"/>
    <div className="flex items-center gap-2"><span className="size-2.5 shrink-0 rounded-full" style={{background:color}}/><span className="truncate text-[11px] font-semibold">{d.label}</span></div>
    <div className="mt-1 flex justify-between gap-1 text-[10px] text-muted-foreground"><span className="uppercase">{d.kind}</span>{d.touched && <span className="font-semibold text-critical">TOUCHED</span>}</div>
    <p className="mt-1 truncate text-[9px] text-muted-foreground">{d.detail || d.id}</p>
    <Handle type="source" position={Position.Bottom} className="!bg-primary"/>
  </div>;
}
const nodeTypes = {researchCard:GraphCard};
function arrange(rows:ResearchCard[]):Node[] {
  const kinds = [...new Set(rows.map(x=>x.kind))];
  const groups = new Map(kinds.map(t=>[t,rows.filter(x=>x.kind===t)]));
  return kinds.flatMap((type,row)=> (groups.get(type)??[]).map((n,i)=>({
    id:n.id,type:"researchCard",position:{x:row*225,y:i*116},data:n,
  })));
}
function fmt(value:number|null|undefined,digits=3){return typeof value==="number"?value.toFixed(digits):"—";}

export function ResearchGraph({initialTransactionId}: {initialTransactionId?:string}) {
  const [source,setSource]=useState<"filesystem"|"faultlab">("faultlab");
  const [graphScope,setGraphScope]=useState<"full"|"scoring">("full");
  const [model,setModel]=useState("v09_s11");
  const [family,setFamily]=useState("package_upgrade");
  const [layer,setLayer]=useState("layer2");
  const [selected,setSelected]=useState<string|null>(null);
  const [search,setSearch]=useState("");
  const [showAll,setShowAll]=useState(true);
  const [minWeight,setMinWeight]=useState(0);
  const fs = useQuery({queryKey:["research-gnn",graphScope,initialTransactionId],queryFn:()=>researchApi.filesystemTrace(graphScope,graphScope==="scoring"?initialTransactionId:undefined),enabled:source==="filesystem"});
  const fault = useQuery({queryKey:["fault-trace",model,family],queryFn:()=>researchApi.faultlabTrace(family,model),enabled:source==="faultlab",staleTime:60_000,retry:1});
  const availableFamilies = model.startsWith("v09_") ? faultlabFamilies : faultlabFamilies.filter(f=>["sharded_commit","hardlink_retarget"].includes(f.value));
  const data=source==="filesystem"?fs.data:fault.data;
  const busy=source==="filesystem"?fs.isPending:fault.isPending;
  const failed=source==="filesystem"?fs.isError:fault.isError;
  const error=source==="filesystem"?fs.error:fault.error;

  const graph = useMemo(()=>{
    if (!data) return {nodes:[] as ResearchCard[], edges:[] as Array<{id:string;source:string;target:string;relation:string;weight:number|null}>, nodeDetails:new Map<string,GnnNode|FaultNode>()};
    const nodes:ResearchCard[]=[];const nodeDetails=new Map<string,GnnNode|FaultNode>();
    const edges:Array<{id:string;source:string;target:string;relation:string;weight:number|null}>=[];
    if (source==="filesystem" && fs.data){
      for(const n of fs.data.nodes){const id=String(n.id);nodes.push({id,kind:n.nodeType,label:n.label,detail:n.path||n.subtitle,touched:n.touched,selected:selected===id});nodeDetails.set(id,n);}
      for(const e of fs.data.edges){const weight=e.attention?.[layer];edges.push({id:e.id||`${e.source}-${e.target}`,source:String(e.source),target:String(e.target),relation:e.relation||"relation",weight:typeof weight==="number"?weight:null});}
    }
    if(source==="faultlab" && fault.data){
      for(const n of fault.data.nodes){const id=String(n.id);nodes.push({id,kind:n.role,label:n.label,detail:`node ${n.id}`,selected:selected===id});nodeDetails.set(id,n);}
      for(const e of fault.data.edges){const weight=e[layer];edges.push({id:String(e["id"]??`${e["source"]}-${e["target"]}`),source:String(e["source"]),target:String(e["target"]),relation:String(e["relation"]??"relation"),weight:typeof weight==="number"?weight:null});}
    }
    return {nodes,edges,nodeDetails};
  },[source,fs.data,fault.data,layer,selected,data]);

  const visual=useMemo(()=>{
    const filter=search.trim().toLowerCase();
    const visible=filter?graph.nodes.filter(n=>n.label.toLowerCase().includes(filter)||n.kind.includes(filter)||n.id.includes(filter)):graph.nodes;
    const allowed=new Set(visible.map(x=>x.id));
    let edges=graph.edges.filter(x=>allowed.has(x.source)&&allowed.has(x.target));
    if (minWeight>0 && layer!=="encoder" && layer!=="encoded") {
      edges=edges.filter(e=>e.weight===null || e.weight>=minWeight);
    }
    if(!showAll&&edges.length>75){edges=[...edges].sort((a,b)=>(b.weight??-1)-(a.weight??-1)).slice(0,75);}
    const flowEdges:Edge[]=edges.map(e=>({id:e["id"] as string,source:e["source"] as string,target:e["target"] as string,label: e["weight"]===null?String(e["relation"]):`${String(e["relation"])} ${(e["weight"] as number).toFixed(2)}`,style:{stroke:"var(--blue)",strokeWidth:e["weight"]==null?1.2:1+(e["weight"] as number)*2.6,opacity:e["weight"]==null?.45:Math.max(.25,Math.min(1,e["weight"] as number))},labelStyle:{fill:"var(--ink)",fontSize:9},labelBgStyle:{fill:"var(--cream)"},animated:e["weight"]!==null&&(e["weight"] as number)>.7}));
    return {nodes:arrange(visible),edges:flowEdges,visibleEdges:edges.length};
  },[graph,search,showAll,minWeight,layer]);

  const detail=selected?graph.nodeDetails.get(selected):null;
  const embeddings=detail?("embeddings" in detail?detail.embeddings:("embedding" in detail?detail.embedding:undefined)):undefined;
  const vector=embeddings?.[layer]??embeddings?.["layer2"]??embeddings?.["encoder"]??[];
  const selectedIncoming = useMemo(() => selected ? graph.edges
    .filter(e => e.target===selected && e.weight!==null)
    .sort((a,b)=>(b.weight??0)-(a.weight??0)) : [], [graph.edges,selected]);
  const selectedNodeInfo = selected ? graph.nodes.find(n => n.id===selected) : null;
  const fullFeatures = detail && "features" in detail ? detail.features : undefined;
  const layers=source==="filesystem"?["encoded","layer1","layer2"]:["encoder","layer1","layer2","layer3"];
  return <div className="space-y-4" data-testid="react-research-gnn">
    <div className="grid gap-3 lg:grid-cols-2">
      <label className="text-xs font-medium">Graph and model
        <select className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm" value={source} onChange={e=>{setSource(e.target.value as typeof source);setSelected(null);setLayer("layer2")}}><option value="filesystem">Filesystem GNN (2-layer)</option><option value="faultlab">FaultLab GNN (3-layer)</option></select>
      </label>
      <label className="text-xs font-medium">{source==="filesystem"?"Graph scope":"Checkpoint"}
        {source==="filesystem"?<select className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm" value={graphScope} onChange={e=>setGraphScope(e.target.value as typeof graphScope)}><option value="full">Full heterogeneous graph</option><option value="scoring">Selected transaction scoring graph</option></select>
          :<select className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm" value={model} onChange={e=>{setModel(e.target.value);if(!e.target.value.startsWith("v09_") && !["sharded_commit","hardlink_retarget"].includes(family)) setFamily("hardlink_retarget");setSelected(null)}}>{faultlabModels.map(x=><option key={x.value} value={x.value}>{x.label}</option>)}</select>}
      </label>
    </div>
    {source==="faultlab"&&<label className="block text-xs font-medium">Workload family
      <select className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm" value={family} onChange={e=>{setFamily(e.target.value);setSelected(null)}}>{availableFamilies.map(x=><option key={x.value} value={x.value}>{x.label}</option>)}</select>
    </label>}
    {busy?<LoadingState label="Loading model-specific attention trace" rows={5}/>:failed||!data?<ErrorState message={`Trace unavailable: ${String(error??"No response")}`} onRetry={()=>void (source==="filesystem"?fs.refetch():fault.refetch())}/>:<>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard icon="graph" label="Nodes" value={String(graph.nodes.length)}/>
        <MetricCard icon="links" label="Relations" value={String(graph.edges.length)}/>
        <MetricCard icon="risk" label="Predicted risk" value={source==="faultlab"?fmt(fault.data?.risk_prediction):fmt(fs.data?.score)}/>
        <MetricCard icon="oracle" label="Simulator oracle" value={source==="faultlab"?fmt(fault.data?.oracle_risk):"Not applicable"}/>
      </div>
      <Panel title="Model-internal attention explorer" subtitle={`${source==="faultlab"?model:"Filesystem GNN"} · ${graph.nodes.length} nodes / ${graph.edges.length} directed relations`} icon={GitBranch}>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-border bg-background px-2"><Search className="size-3 text-muted-foreground"/><input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Find nodes" aria-label="Find graph nodes" className="w-36 bg-transparent py-2 text-xs outline-none"/></div>
          <label className="flex items-center gap-2 text-xs">Layer <select className="rounded border border-border bg-background px-2 py-1.5 text-xs" value={layer} onChange={e=>setLayer(e.target.value)}>{layers.map(x=><option key={x} value={x}>{x}</option>)}</select></label>
          <label className="flex items-center gap-2 text-xs">Minimum attention
            <input aria-label="Minimum attention weight" type="range" min="0" max="1" step="0.01" value={minWeight} onChange={e=>setMinWeight(Number(e.target.value))} className="w-28 accent-orange" />
            <output className="tabular w-10">{minWeight.toFixed(2)}</output>
          </label>
          <label className="ml-auto flex items-center gap-1 text-xs text-muted-foreground"><input type="checkbox" checked={showAll} onChange={e=>setShowAll(e.target.checked)}/> Show all edges</label>
          <button type="button" className="pixel-button pixel-button-secondary px-2 py-1.5 text-[10px]" onClick={()=>{setSelected(null);setSearch("");setMinWeight(0);setShowAll(true)}}>Reset selection</button>
          <span className="text-[10px] text-muted-foreground">{visual.visibleEdges} displayed</span>
        </div>
        <div className="h-[540px] overflow-hidden rounded-md border border-border bg-background" data-testid="gnn-reactflow">
          <ReactFlow nodes={visual.nodes} edges={visual.edges} nodeTypes={nodeTypes} onNodeClick={(_,node)=>setSelected(node.id)} minZoom={0.1} fitView fitViewOptions={{padding:.16}} proOptions={{hideAttribution:true}}>
            <Background variant={BackgroundVariant.Dots} color="rgba(119, 93, 76, 0.22)" gap={24}/><Controls/><MiniMap pannable zoomable nodeColor={n=>palette[(n.data as unknown as ResearchCard).kind]??"var(--muted-foreground)"}/>
          </ReactFlow>
        </div>
        {graph.edges.length>75&&!showAll&&<p className="mt-2 text-xs text-muted-foreground">For legibility the top 75 visible relationships are shown by {layer} weight. Toggle “Show all edges” for the complete input graph.</p>}
      </Panel>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,1fr)]">
        <Panel title="FaultLab node inspector" subtitle={selectedNodeInfo ? `${selectedNodeInfo.kind.toUpperCase()} · ${selectedNodeInfo.label}` : "Select a node to inspect actual model inputs"} icon={BrainCircuit}>
          {detail && selectedNodeInfo ? <div className="space-y-4 text-xs">
            <div className="grid grid-cols-2 gap-2 border-b border-border pb-3">
              <div><span className="text-muted-foreground">Node ID</span><div className="tabular font-bold">{selected}</div></div>
              <div><span className="text-muted-foreground">Category</span><div className="font-bold">{selectedNodeInfo.kind}</div></div>
              <div><span className="text-muted-foreground">Input features</span><div className="tabular font-bold">{Array.isArray(fullFeatures) ? fullFeatures.length : Object.keys(fullFeatures??{}).length}</div></div>
              <div><span className="text-muted-foreground">Embedding dimensions</span><div className="tabular font-bold">{vector.length}</div></div>
              <div><span className="text-muted-foreground">Incoming messages</span><div className="tabular font-bold">{selectedIncoming.length}</div></div>
              <div><span className="text-muted-foreground">Selected layer</span><div className="tabular font-bold">{layer}</div></div>
            </div>
            <div>
              <h3 className="mb-2 font-bold uppercase">{vector.length}D {layer} embedding</h3>
              {vector.length ? <><div className="grid h-24 grid-cols-16 items-end gap-[3px] border border-border bg-surface-2 p-2" role="img" aria-label={`Signed ${vector.length}-dimensional node embedding`}>{vector.map((v,i)=>
                <span key={i} title={`dimension ${i}: ${v.toFixed(5)}`} className="block min-h-0 bg-blue" style={{height:`${Math.max(3,Math.min(100,Math.abs(v)*100))}%`,background:v>=0?"var(--blue)":"var(--orange)"}}/>
              )}</div><p className="mt-1 text-muted-foreground">Blue = positive; orange = negative. Hover a bar to see the signed value.</p></> : <p className="text-muted-foreground">No embedding for this layer.</p>}
            </div>
            <div>
              <h3 className="mb-2 font-bold uppercase">Incoming effective attention</h3>
              {selectedIncoming.length ? <div className="max-h-56 overflow-auto border border-border">{selectedIncoming.map(e=><div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 border-b border-border p-2 last:border-b-0" key={e.id}>
                <span className="min-w-0 break-words">{graph.nodes.find(n=>n.id===e.source)?.label??e.source} · {e.relation}</span><strong className="tabular text-orange-dark">{fmt(e.weight,4)}</strong>
              </div>)}</div>:<p className="text-muted-foreground">No attention values for incoming edges at this layer.</p>}
            </div>
            <details className="border border-border p-2"><summary className="cursor-pointer font-bold">Raw input features</summary>
              <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap break-all text-[10px]">{JSON.stringify(fullFeatures??{},null,2)}</pre>
            </details>
            <details className="border border-border p-2"><summary className="cursor-pointer font-bold">Full embedding vector</summary>
              <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap break-all text-[10px]">{JSON.stringify(vector,null,2)}</pre>
            </details>
          </div> : <p className="py-8 text-center text-xs text-muted-foreground">Select a graph node to inspect the selected checkpoint's real features, message weights, and embedding.</p>}
        </Panel>
        <Panel title="Trace integrity and interpretation" subtitle="Predictions and attention are checkpoint-specific" icon={Layers}>
          <p className="text-xs leading-6 text-muted-foreground">{source==="faultlab"?fault.data?.attention_description:fs.data?.attentionDescription}</p>
          {source==="faultlab" && <div className="mt-3 space-y-2 text-xs">
            <p>Selected checkpoint: <strong className="tabular">{fault.data?.model}</strong></p>
            <p>Family: <strong>{fault.data?.family}</strong></p>
            <p>Invariant: <strong>{fault.data?.constraint}</strong></p>
            <p>GNN risk: <strong className="tabular">{fmt(fault.data?.risk_prediction,4)}</strong> · Simulator oracle: <strong className="tabular">{fmt(fault.data?.oracle_risk,4)}</strong></p>
            <p>Absolute error: <strong className="tabular">{fmt(fault.data?.absolute_error,4)}</strong></p>
          </div>}
          <p className="mt-4 border-l-4 border-watch bg-watch/10 p-3 text-xs leading-5 text-foreground">Learned attention is not proof of causal attribution. These are synthetic durability-model results, not real ext4 crash-recovery measurements.</p>
          <div className="mt-3 flex flex-wrap gap-2 text-[10px]">{Object.entries(palette).map(([kind,color])=><span key={kind} className="inline-flex items-center gap-1 border border-border bg-card px-2 py-1"><i className="inline-block size-2.5" style={{background:color}}/>{kind}</span>)}</div>
        </Panel>
      </div>
    </>}
  </div>;
}
function MetricCard({label,value,icon}:{label:string;value:string;icon:string}){return <div className="pixel-depth-frame" data-testid={`metric-${icon}`}><div className="research-stat"><p className="text-[11px] text-muted-foreground">{label}</p><p className="mt-1 tabular text-xl font-semibold">{value}</p></div></div>}

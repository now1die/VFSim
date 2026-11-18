"""Static merge checks that do not require npm; not a substitute for build/tests."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
required={
  'CrashLab benchmark':('src/routes/simulation.tsx','<CrashBenchmarks />'),
  'Benchmark API':('src/components/vfsim/CrashBenchmarks.tsx','researchApi.independent'),
  'Graph full-state':('src/routes/graph.tsx','getGraph(selectedTxId || undefined)'),
  'All edges default':('src/components/vfsim/ResearchGraph.tsx','useState(true)'),
  'Attention slider':('src/components/vfsim/ResearchGraph.tsx','Minimum attention'),
  'Incoming weights':('src/components/vfsim/ResearchGraph.tsx','Incoming effective attention'),
  'Full embeddings':('src/components/vfsim/ResearchGraph.tsx','Full embedding vector'),
  'Live API':('src/services/api.ts','API_BASE_URL'),
  'Prerender build':('vite.config.ts','prerender:'),
  'Single-port real build':('backend/static_host.py','_send_asset'),
  'Reproducible npm lock':('package-lock.json','"lockfileVersion"'),
}
for name,(p,s) in required.items():
  f=r/p
  assert f.is_file() and s in f.read_text(),f'Missing {name}: {p}'
  print('OK',name)
for banned in ('frontend-preview','src/lib/mock-data.ts'):
  assert not (r/banned).exists(),f'Stale demo or preview: {banned}'
assert not (r/'frontend-dist').exists(), 'Do not ship old v1.1 built assets as v1.4'
print('All source merge checks passed.')

# Live-only interface: no scripted example transactions or demo fallback.
assert 'isDemo' not in (r/'src/components/vfsim/BackendStatus.tsx').read_text(encoding='utf8')
assert '/api/research/sample' not in (r/'backend/main.py').read_text(encoding='utf8')
assert 'loadResearchSample' not in (r/'src/services/api.ts').read_text(encoding='utf8')
assert 'LiveOperations' in (r/'src/routes/index.tsx').read_text(encoding='utf8')

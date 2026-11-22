# Deploy VFSim with Vercel Hobby and Render Free

## Architecture

Browser → Vercel TanStack Start / Nitro → HTTPS → Render Free FastAPI, CPU PyTorch, and a private temporary SQLite file.

The Blueprint selects `plan: free` for a single Python web service with 512 MB RAM and 0.1 CPU. It creates no persistent disk or external database. Vercel Hobby runs the frontend and uses Fluid Compute with a 90-second function budget. Stay within both providers’ included usage. Render Free shares 750 instance hours per workspace each month. Render can charge for bandwidth and build-minute overages when a payment method is on file. Use an account without a payment method, or confirm the workspace’s applicable spend controls before deploying. The application requires no paid resources. [Render Free limits](https://render.com/docs/free) · [Vercel Hobby](https://vercel.com/docs/plans/hobby)

Render starts the worker with the original filesystem GNN and all seven FaultLab checkpoints. Hosted simulation requests run one at a time to bound simultaneous inference and session decoding. FaultLab reconstructs its original oracle-selected example for each external family without keeping the full 1,440-case cohort in memory. The selection, checkpoint weights, predictions, and recovery semantics stay intact.

### Temporary sessions and cold starts

Each process creates a private directory for its SQLite session store. Each browser gets an opaque random session ID in a secure, host-only, HTTP-only cookie. The namespace, file handles, transactions and simulated fsync boundary remain available during the session. **All hosted session state resets when Render sleeps, restarts, or redeploys.** Sessions also expire after seven days. During one process lifetime, the worker holds up to 64 active sessions, with a limit of 12 per network, 150 operations per session, and 2 MiB of serialized state per session. Requests without a verified Vercel ingress address share a fallback admission limit. Old cookies cannot retrieve data after a restart.

Render Free sleeps after 15 idle minutes. An inbound request wakes it, which can take about a minute. VFSim shows this on the dashboard. The frontend shares an in-flight health probe before the first API call or after a long idle period. The Vercel proxy retries only this read-only probe when Render returns a transient response or loading page, within one 75-second deadline. The browser allows 85 seconds. Ordinary API calls retain a 25-second upstream timeout. Mutations and session creation have no automatic retries. A lost session can be reopened automatically for a read. A write against an expired session returns a message before any operation occurs. The dashboard has no periodic keep-alive requests.

The browser calls the same-origin `/api` proxy. The proxy requires the exact frontend origin on writes, forwards only the VFSim cookie, and sends a server-only shared secret to Render. The backend requires that secret for API requests and fails closed when the temporary store is unavailable. Public `/health` and `/ready` remain available for probes. The proxy does not require cross-origin browser requests.

## Environment variables

| Where                         | Name                 | Value or source                                                              |
| ----------------------------- | -------------------- | ---------------------------------------------------------------------------- |
| Vercel runtime                | `VFSIM_API_URL`      | The HTTPS origin assigned to the Render service                              |
| Vercel and Render             | `VFSIM_PROXY_TOKEN`  | The same secret in both dashboards. Generate with `openssl rand -hex 32`.    |
| Render                        | `VFSIM_CORS_ORIGINS` | The exact Vercel origin. Separate explicitly authorized origins with commas. |
| Render Blueprint              | `VFSIM_ENV`          | `production`                                                                 |
| Render Blueprint              | `PYTHON_VERSION`     | `3.12.14`                                                                    |
| Optional development override | `VITE_API_URL`       | For example, `http://127.0.0.1:8000`                                         |

Keep provider secrets in the dashboards under server-only names. The Render startup creates the temporary database automatically.

## Deploy and verify

1. Import the repo into the intended Vercel Hobby project. Use the repository root, TanStack Start, Node.js 24, and the checked-in build configuration. Leave Output Directory unset. Note the intended frontend origin.
2. In Render, create a Blueprint from the branch containing the updated `render.yaml`. Confirm the plan is **Free**, no disk or database appears in the plan, and the review shows no paid resources. Set `VFSIM_PROXY_TOKEN` and `VFSIM_CORS_ORIGINS` when prompted.
3. Open the actual Render HTTPS `/ready` endpoint. Check that it returns `gnn: loaded` and `storage: temporary`. Check startup logs and RAM use while loading and tracing models.
4. Set the Vercel project’s `VFSIM_API_URL` and matching `VFSIM_PROXY_TOKEN`, then deploy. Use the corresponding environment scope for preview or production. Verify a cold and a warm run against the resulting frontend URL.

```bash
python -m pip install playwright
python -m playwright install chromium
python scripts/smoke_hosted.py "$FRONTEND_ORIGIN"
```

The smoke test checks all seven routes, actual browser file creation, hard links, fsync and crash recovery, graph data, prediction, FaultLab, and isolation between browsers. Verify the cold path after Render has spun down. The deployment should remain gated on the hosted check and a review of Render’s memory metrics.

## Local development and acceptance

Use Node.js 24 and Python 3.12.14 for production parity. The default local simulator keeps its state in memory. The optional local `VFSIM_CHECKPOINT_PATH` keeps the simulator’s existing fsync file behavior.

```bash
npm ci
python -m pip install -r backend/requirements.txt
(cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000)
```

In another terminal, run `npm run dev`. For a production-like test, run `npm run build:vercel` and serve with `npm run preview -- --mode vercel`. Start Python with `VFSIM_ENV=production` and give both processes the same local test token. Point the preview’s `VFSIM_API_URL` at the local Python service. The offline smoke test accepts `--offline` with a separately configured preview aimed at an unreachable local port.

```bash
npm run verify:frontend
npm run bundle:static
npm run build:vercel
(cd backend && python -m pytest -q)
```

## Free-tier test evidence

- **PASS** — Render’s public Blueprint schema accepted one Free web service with no disk or database. Nitro emitted a Node.js 24 Vercel function with a 90-second duration.
- **PASS** — 107 backend tests, seven frontend tests, scoped ESLint and formatting checks, and the static and Vercel builds. The npm audit reported zero vulnerabilities.
- **PASS** — Chromium tested the local Vercel build with the real backend over HTTPS, an unavailable upstream, and a simulated Render wake-up that returned a loading page and a 503 before becoming ready.
- **PASS** — A local Linux worker loaded eight checkpoints, ran a 150-operation filesystem session with hard links, scored it with the GNN, verified fsync recovery, and served 32 FaultLab traces covering all seven FaultLab checkpoints and six external families. Peak backend RSS was 292,216,832 bytes (278.7 MiB). A watchdog monitored the process against a 512,000,000-byte threshold. A worker restart rejected the old session cookie and opened an empty session.
- **BLOCKED** — Kernel-enforced Render-equivalent memory and CPU limits and live cloud verification. This environment’s rootless Docker daemon lacks cgroup delegation, so it did not enforce its requested limits. The measured RSS excludes kernel accounting such as filesystem caches. Verify the cap and cold-start behavior against a real Render Free instance before publishing.

# Deploy VFSim on Vercel and Render

## Findings

| Finding | Fix |
| --- | --- |
| The frontend locked TanStack Start to 1.168.32, inside the affected range for CVE-2026-102989. The original npm audit also flagged Vite and Zod. | Upgrade Start and align Router with Start's dependency graph. Upgrade Vite, remove unused direct dependencies, and regenerate the npm lockfile. |
| The original HTML had no visible content until the client mounted. The production JavaScript used a loopback API default, so a remote visitor's API calls targeted their own machine. | Render a loading shell in HTML and use a relative `/api` URL in production. Both frontend API clients share timeouts, session initialization, and failure handling. |
| The existing build produced local prerendered assets without a Vercel adapter. | Build with Nitro's Vercel preset. The generated route config serves assets directly and dispatches nested URLs and API requests to a Node.js 24 function. |
| FastAPI used one process-global simulator, allowed every CORS origin, and could report healthy before the full model set was usable. | Scope hosted simulator state to a browser cookie, restrict origins and upstream access, validate the shipped models at startup, and expose separate liveness and readiness checks. |
| The existing optional checkpoint file saves the namespace at simulated fsync. It does not preserve each browser's transaction history on a host restart. | Store hosted sessions, transaction history, and durability boundaries in SQLite on a Render persistent disk. Keep simulated crash semantics and local checkpoint behavior. |

The original hosted white screen was not independently reproduced on Vercel because no deployment was available for inspection. The missing initial markup and loopback API fallback were reproduced from the original local build.

## Architecture

Browser → Vercel TanStack Start / Nitro → HTTPS → Render FastAPI → VFSim, shipped CPU PyTorch checkpoints, and SQLite on a persistent disk.

The browser makes same-origin requests, so its host-only session cookie works with browsers that block third-party cookies. Only the Vercel function knows the Render origin and shared proxy secret. It forwards VFSim's session cookie and rejects cross-origin browser writes. Render rejects direct API callers that lack the proxy secret. Public `/health` and `/ready` expose service status without credentials. Configure CORS with exact frontend origins. The proxy does not require cross-origin browser requests.

The CPU-only PyTorch wheel was 209.5 MB compressed and approximately 809 MiB installed before other dependencies. Vercel Functions would also require external state for this workload. Render Standard, a single Uvicorn worker, and a persistent disk keep the existing Python engine and checkpoint assets together. Render's standard instance has 2 GB of RAM. Check peak memory with realistic workloads in the Render dashboard before increasing concurrency; local smoke measurements do not establish cloud capacity.

A hosted browser session expires seven days after creation. Sessions preserve the namespace, open handles, transaction IDs and replay history, and the simulated fsync boundary across service restarts. Each session is capped at 150 operations and 2 MiB of serialized state, and the instance admits up to 256 unexpired sessions. Expired sessions are purged on session creation. The cookies identify anonymous simulator sessions and should be treated as bearer credentials. Design for one instance and one worker with this SQLite disk. Before offering unrestricted public access, configure traffic rate limits for `POST /api/session` and other write requests in the Vercel project. For an unattended demo, deployment protection can restrict access to invited users.

## Environment variables

| Where | Name | Value or source |
| --- | --- | --- |
| Vercel runtime | `VFSIM_API_URL` | The HTTPS origin shown after Render creates the Python service. Use the origin alone. |
| Vercel and Render | `VFSIM_PROXY_TOKEN` | The same newly generated secret in both dashboards. Generate with `openssl rand -hex 32`. |
| Render | `VFSIM_CORS_ORIGINS` | The exact production Vercel origin. Separate multiple explicitly authorized origins with commas. |
| Render Blueprint | `VFSIM_ENV` | `production` |
| Render Blueprint | `VFSIM_SESSION_DB` | `/var/data/vfsim.sqlite3` |
| Render Blueprint | `PYTHON_VERSION` | `3.12.14` |
| Optional local development | `VITE_API_URL` | Override the default local backend origin, for example `http://127.0.0.1:8000`. |

Set production variables before deployment. Keep the shared token and upstream URL under server-only names. The frontend does not read these from a `VITE_` variable. For a preview deployment, use a separately configured backend if preview testers need independent data and access. Keep secrets in provider environment settings.

## Local development

Use Node.js 24 and Python 3.12.14 for production parity. Install the backend development requirements and start Uvicorn from `backend/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
(cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000)
```

From the repository root in another terminal:

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:3000`. Local development defaults to the existing single in-memory simulator. Set the optional `VFSIM_CHECKPOINT_PATH` on the local Python process to persist the existing fsync checkpoint. The compiled single-port demo remains available with `npm run bundle:static` and `uvicorn static_host:app` from `backend/`.

## Deploy

1. Import this repository into a Vercel project with the root directory set to the repository root. Use the TanStack Start preset and Node.js 24. Keep the install and build commands from `vercel.json` and leave Output Directory unset so Nitro supplies the Build Output API config. Obtain the intended production hostname from the project.
2. In Render, create a Blueprint from this repository and select the commit containing `render.yaml`. This provisions a Standard Python service with its working directory set to `backend`, a CPU-only build, `/ready` health checks, and a 1 GB persistent disk. Supply `VFSIM_PROXY_TOKEN` and `VFSIM_CORS_ORIGINS` when prompted. For an existing service, apply equivalent settings and use its current durable disk if it already contains compatible session data.
3. Confirm Render reported healthy. Open its actual HTTPS `/health` and `/ready` endpoints. Verify that readiness reports `gnn: loaded` and `storage: durable`. Check startup logs for model or disk errors. Keep backups of the SQLite state using SQLite's backup API and the Render disk backup process.
4. In Vercel set `VFSIM_API_URL` to the actual Render HTTPS origin and `VFSIM_PROXY_TOKEN` to the matching secret. Deploy the production branch. Check the Vercel build log for the TanStack and Nitro build and confirm no advisory block.
5. Run the browser smoke against the actual Vercel origin. It exercises the UI, direct routes and refreshes, API operations, hard links, transaction history, crash recovery, GNN, FaultLab, and isolation. Verify Render memory during this run and check both providers' runtime logs.

```bash
python -m pip install playwright
python -m playwright install chromium
python scripts/smoke_hosted.py "$FRONTEND_ORIGIN"
```

For local acceptance of the Vercel artifact, build with `npm run build:vercel` and serve with `npm run preview -- --mode vercel`. Pass the local API origin and an identical test proxy token to the preview and Python environments respectively. Set `VFSIM_ENV=production` and a scratch `VFSIM_SESSION_DB` in an existing directory on Python. Loopback HTTP is allowed by the proxy outside Vercel for this test. Use a second preview configured to an unreachable local API port to verify the offline screen with `scripts/smoke_hosted.py --offline`.

## Verification

| Result | Check |
| --- | --- |
| PASS | Clean `npm ci`, dependency audit with zero reported advisories, TypeScript, and four Vitest tests. Resolved versions include TanStack Start 1.169.7, Router 1.170.46, React 19.3.1, Vite 8.3.8, and Nitro 3.0.261103-rc on Node.js 24. |
| PASS | Local static build and Vercel Build Output API artifact. The Vercel function specifies `nodejs24.x` and the browser assets contain no loopback API default or proxy credential header. |
| PASS | Clean Python 3.12.14 environment with CPU-only PyTorch 2.15.0, FastAPI 0.141.1, Uvicorn 0.52.1, Pydantic 2.13.4, and NumPy 2.3.5. `pip check` passed and the audit of the resolved environment reported no known vulnerabilities. |
| PASS | Backend import, eight model checkpoints, and 104 backend tests including session isolation, recovery, and restart persistence. |
| PASS | Local Chromium smoke through the Vercel build and a locally trusted HTTPS FastAPI upstream. All seven routes loaded directly and refreshed. Core simulation, prediction, and FaultLab worked with no browser console or network errors in the online run. The offline retry path also passed. |
| PASS | A Uvicorn process restart preserved hard links, the fsync checkpoint, and transaction history on disk. The backend's local peak RSS was about 310 MiB during smoke testing. |
| FAIL | Repository-wide ESLint still reports existing format errors. The baseline reported 386 errors and two warnings and the updated tree reports 373 errors and two warnings. |
| BLOCKED | Actual Render and Vercel deployment, the platform vulnerability gate, public browser verification, and hosted memory measurements. There were no authenticated Render or Vercel credentials or project links in the execution environment. |

Platform references: [Vercel TanStack Start guide](https://vercel.com/docs/frameworks/full-stack/tanstack-start), [Render Blueprint specification](https://render.com/docs/blueprint-spec), [Render persistent disks](https://render.com/docs/disks), [Render Python version](https://render.com/docs/python-version), [Render instance pricing and memory](https://render.com/pricing), [TanStack advisory](https://github.com/advisories/GHSA-qx66-fv34-fjm8).

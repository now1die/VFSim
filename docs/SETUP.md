# Installation, usage and troubleshooting

Clone the repository into a fresh directory. Run the setup commands from the directory containing `backend/`, `package.json`, and `verify_install.py`.

## Windows

```powershell
python -m pip install -r backend\requirements.txt
python verify_install.py
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal:

```powershell
npm.cmd ci
npm.cmd run dev
```

Visit `http://127.0.0.1:3000`. The backend documentation is at `http://127.0.0.1:8000/docs`.

## Linux / macOS development

From the project root:

```bash
python3 -m pip install -r backend/requirements.txt
python3 verify_install.py
(cd backend && python3 -m uvicorn main:app --host 127.0.0.1 --port 8000)
```

In another terminal:

```bash
npm ci
npm run dev
```

VFSim remains an in-memory simulator on both operating systems; running it on Linux does **not** automatically validate ext4 behavior.

## Test the release

```bash
python verify_install.py
(cd backend && python -m pytest -q tests)
npm run verify:frontend
npm run build
```

The first two commands check the backend. The last two require installed Node dependencies. Keep the test output with your release notes.

## Common errors

- `No module named 'gnn.inference'`: the active backend `main.py` belongs to an older, incompatible reference copy. Use a fresh extraction and check `python verify_install.py`.
- Frontend empty or offline: ensure both processes are running; verify `/api/health` and inspect the browser's network panel.
- Empty filesystem graph: the virtual filesystem starts with its root. Execute operations in the UI to create nodes.
- Blank per-transaction attribution: the current live model API does not expose causal feature attribution; the UI explains that limitation rather than inventing weights.
- Missing npm packages: run `npm ci` on a machine with npm access. Never substitute an unrelated preview interface for the real React app.

## Compiled local hosting (optional)

`npm run bundle:static` attempts to build the real frontend and copy the generated client assets to `frontend-dist`. If the build succeeds, `cd backend && python -m uvicorn static_host:app --port 8000` can serve the API and compiled site together. A failed build must not be presented as verified deployment.

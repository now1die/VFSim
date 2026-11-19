# VFSim screenshots

## Updated Overview

The desktop and mobile screenshots show the React app running against FastAPI. The skyline buildings open five app sections. The graph shows live filesystem operations in `/research_demo_1`, where `report.txt` and `alias.txt` share one inode. The mobile captures use a viewport 390 CSS pixels wide. The graph close-up shows the result of tapping **Fit graph**.

| View | Desktop | Mobile |
| --- | --- | --- |
| Overview | [Dashboard and clickable skyline](screenshots/overview-desktop.png) | [Dashboard and clickable skyline](screenshots/overview-mobile.png) |
| Filesystem graph | [Directory tree and shared inode](screenshots/filesystem-desktop.png) | [Tree fitted to phone width](screenshots/filesystem-mobile.png) |

## Original v1.7.1 gallery

Chromium captured the original compiled frontend running with the FastAPI backend. The capture run created a file, wrote 256 bytes to it, and made a hard link through the browser. [Capture log](screenshots/verification.json)

| View | Screenshot |
| --- | --- |
| Overview | [Dashboard and live operations](screenshots/01_overview.png) |
| Graph Explorer | [Filesystem topology](screenshots/02_graph_explorer.png) |
| Transactions | [Operation history](screenshots/03_transactions.png) |
| Risk Analysis | [Model evaluation](screenshots/04_risk_analysis.png) |
| Crash Lab | [Simulated crash recovery](screenshots/05_crash_lab.png) |
| GNN Model | [Model trace and checkpoint controls](screenshots/06_gnn_model.png) |
| Settings | [Application settings](screenshots/07_settings.png) |
| GNN node inspector | [Selected node features, attention, and embedding](screenshots/08_gnn_selected_node.png) |
| GNN graph mode | [Alternate Graph Explorer view](screenshots/09_graph_gnn_mode.png) |

## Run the route capture script

Start a fresh backend so the demo can create `/report.txt`. With both development services running, install Playwright and capture the pages from the project root.

```bash
python -m pip install playwright
python -m playwright install chromium
python scripts/capture_real_pages.py --exercise-demo
```

For the compiled single-port app, add `--base http://127.0.0.1:8000`. Use `--chromium /path/to/chromium` to select a system browser.

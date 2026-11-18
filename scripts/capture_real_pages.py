"""Capture the ACTUAL TanStack/React app, never the deleted preview.

Preconditions:
    1. FastAPI running at http://127.0.0.1:8000
    2. npm dependencies installed and `npm run dev` running at http://127.0.0.1:3000
    3. `pip install playwright` and a Chromium browser available

Usage:
    python scripts/capture_real_pages.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROUTES = [
    ("01_overview.png", "/"),
    ("02_graph_explorer.png", "/graph"),
    ("03_transactions.png", "/transactions"),
    ("04_risk_analysis.png", "/risk"),
    ("05_crash_lab.png", "/simulation"),
    ("06_gnn_model.png", "/model"),
    ("07_settings.png", "/settings"),
]


def http(url: str, method="GET") -> dict:
    req = Request(url, method=method, headers={"Accept": "application/json"})
    with urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screenshot the full original TanStack VFSim interface")
    parser.add_argument("--base", default="http://127.0.0.1:3000", help="Actual TanStack app URL, NOT a static preview")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Real backend URL")
    parser.add_argument("--output", default="screenshots_verified", help="Destination for REAL screenshots")
    parser.add_argument("--chromium", help="Optional path to a system Chromium executable")
    parser.add_argument(
        "--exercise-demo",
        action="store_true",
        help="Create, write, and hard-link /report.txt through the UI on a fresh backend",
    )
    args = parser.parse_args()

    # Refuse to accidentally screenshot an offline/dummy page.
    status = http(args.api.rstrip("/") + "/api/health")
    if status.get("status") not in ("ok", "healthy"):
        raise RuntimeError(f"FastAPI health check failed: {status}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Install Playwright: python -m pip install playwright") from exc

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    screenshots = []
    js_errors = []

    with sync_playwright() as playwright:
        # Prefer installed Chrome/Edge on Windows; otherwise Playwright's Chromium.
        browser = None
        errors = []
        options = ({"channel": "chrome"}, {"channel": "msedge"}, {})
        if args.chromium:
            options = ({"executable_path": args.chromium}, *options)
        for opts in options:
            try:
                browser = playwright.chromium.launch(headless=True, **opts)
                break
            except Exception as exc:
                errors.append(str(exc))
        if browser is None:
            raise RuntimeError("No supported Chromium browser found. Run: python -m playwright install chromium\n" + "\n".join(errors))
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: js_errors.append(str(error)))
        if args.exercise_demo:
            page.goto(args.base.rstrip("/") + "/", wait_until="domcontentloaded")
            form = page.locator("form").filter(
                has=page.get_by_role("button", name="Execute operation")
            )
            for action, endpoint, message in (
                ("create_file", "create", "Recorded create"),
                ("write", "write", "Recorded write"),
                ("link", "link", "Recorded link"),
            ):
                form.get_by_role("combobox").select_option(action)
                with page.expect_response(
                    lambda response: response.url.endswith("/api/vfs/" + endpoint)
                    and response.request.method == "POST"
                ) as pending:
                    form.get_by_role("button", name="Execute operation").click()
                if not pending.value.ok:
                    raise RuntimeError(f"Demo operation {action} failed: {pending.value.text()}")
                page.get_by_text(message, exact=False).wait_for()
                print("Verified live browser operation:", action)

        for filename, path in ROUTES:
            response = page.goto(args.base.rstrip("/")+path, wait_until="domcontentloaded", timeout=30000)
            if response is None or not response.ok:
                raise RuntimeError(f"Cannot load actual TanStack page {path}: {response.status if response else 'no response'}")
            page.locator("header.app-header").wait_for(timeout=20000)
            page.locator("main.app-content").wait_for(timeout=20000)
            if not page.locator(".pixel-landscape").count():
                raise RuntimeError(f"The pixel landscape from the ORIGINAL source is absent on {path}; aborting")
            if not page.locator(".brand-name").count():
                raise RuntimeError(f"The original app shell is absent on {path}; aborting")
            palette = page.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--orange').trim()")
            font = page.evaluate("() => getComputedStyle(document.body).fontFamily")
            if not palette or "Pixelify Sans" not in font:
                raise RuntimeError(f"The original pixel CSS/fonts are not applied on {path}; aborting")
            # Allow React Query / React Flow / Recharts to settle.
            page.evaluate("() => document.fonts.ready")
            page.wait_for_timeout(1800)
            if js_errors:
                raise RuntimeError(f"JavaScript exceptions while loading {path}: {js_errors}")
            page.evaluate("window.scrollTo({ top: 0, behavior: 'instant' })")
            page.screenshot(path=str(root/filename), full_page=True, animations="disabled")
            screenshots.append(filename)
            print("Captured actual React route:",path,"->",filename)

        # Capture genuine GNN graph and selected-node inspector if fully loaded.
        page.goto(args.base.rstrip("/")+"/model",wait_until="domcontentloaded")
        page.locator("[data-testid='react-research-gnn']").wait_for(timeout=20000)
        page.locator("[data-testid='gnn-reactflow'] .react-flow__node").first.wait_for(timeout=25000)
        graph_nodes = page.locator("[data-testid='gnn-reactflow'] .react-flow__node").count()
        if graph_nodes == 0:
            raise RuntimeError("The actual GNN React Flow component had no model nodes")
        page.locator("[data-testid='gnn-reactflow'] .react-flow__node").first.click(timeout=10000)
        page.get_by_role("heading", name="Incoming effective attention").wait_for(timeout=10000)
        page.evaluate("window.scrollTo({ top: 0, behavior: 'instant' })")
        page.screenshot(path=str(root/"08_gnn_selected_node.png"),full_page=True,animations="disabled")
        screenshots.append("08_gnn_selected_node.png")

        # Capture the alternate GNN graph mode on the original Graph Explorer route.
        page.goto(args.base.rstrip("/")+"/graph", wait_until="domcontentloaded")
        page.get_by_role("button",name="Heterogeneous GNN + attention").click(timeout=10000)
        page.locator("[data-testid='gnn-reactflow'] .react-flow__node").first.wait_for(timeout=25000)
        page.evaluate("window.scrollTo({ top: 0, behavior: 'instant' })")
        page.screenshot(path=str(root/"09_graph_gnn_mode.png"),full_page=True,animations="disabled")
        screenshots.append("09_graph_gnn_mode.png")
        browser.close()

    manifest = {
        "frontend": "Source-built TanStack React application",
        "app_url": args.base,
        "api_url": args.api,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "javascript_exceptions": js_errors,
        "screenshots": screenshots,
        "exercised_live_operations": args.exercise_demo,
    }
    (root/"verification.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Verified {len(screenshots)} genuine frontend captures at {root}")

if __name__ == "__main__":
    main()

"""Exercise a hosted or local Vercel build and its real API using an isolated browser."""
from argparse import ArgumentParser
import re
from uuid import uuid4

from playwright.sync_api import expect, sync_playwright

ROUTES = ("/", "/graph", "/transactions", "/risk", "/simulation", "/model", "/settings")


def smoke(browser, origin, offline):
    with browser.new_context() as context:
        page = context.new_page()
        page_errors = []
        console_errors = []
        bad_responses = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("response", lambda response: bad_responses.append(response.url) if response.status >= 400 else None)
        url = f"{origin}/"
        assert page.goto(url, wait_until="domcontentloaded").ok
        expect(page.get_by_role("heading", name="VFSim", exact=True)).to_be_visible(timeout=25_000)
        expect(page.get_by_text(re.compile("sessions are temporary"))).to_be_visible()
        if offline:
            expect(page.get_by_role("status").filter(has_text="Backend offline")).to_be_visible(timeout=90_000)
            retry = page.get_by_role("button", name="Retry request").first
            expect(retry).to_be_visible()
            with page.expect_response(lambda response: response.url.endswith("/api/health"), timeout=90_000) as retried:
                retry.click()
            assert retried.value.status >= 500
            assert all("/api/" in url for url in bad_responses), bad_responses
            assert not page_errors, page_errors
            print("PASS: offline dashboard, status, and retry controls")
            return

        expect(page.get_by_text(re.compile("Backend live", re.I))).to_be_visible(timeout=90_000)
        for route in ROUTES:
            assert page.goto(origin + route, wait_until="domcontentloaded").ok, route
            expect(page.locator("main.app-content")).to_be_visible(timeout=25_000)
            assert page.reload(wait_until="domcontentloaded").ok, route
            expect(page.locator("main.app-content")).to_be_visible(timeout=25_000)

        page.goto(url, wait_until="domcontentloaded")
        first_path = f"/smoke-{uuid4().hex[:8]}.txt"
        alias_path = first_path.replace(".txt", "-alias.txt")
        form = page.locator("form").filter(has=page.get_by_role("button", name="Execute operation"))
        form.get_by_label("Filesystem path").fill(first_path)
        with page.expect_response(lambda response: response.url.endswith("/api/vfs/create")) as create:
            form.get_by_role("button", name="Execute operation").click()
        assert create.value.ok, create.value.text()
        transaction = create.value.json()
        expect(page.get_by_text(re.compile("Recorded create"))).to_be_visible()

        def api_call(path, payload=None):
            result = page.evaluate("""async ({path, payload}) => {
                const init = payload === null ? {} : {
                    method: 'POST', headers: {'content-type': 'application/json'},
                    body: JSON.stringify(payload),
                };
                const response = await fetch(path, {...init, signal: AbortSignal.timeout(30000)});
                return {status: response.status, body: await response.json()};
            }""", {"path": path, "payload": payload})
            assert result["status"] == 200, (path, result)
            return result["body"]

        api_call("/api/vfs/write", {"path": first_path, "size": 128})
        api_call("/api/vfs/link", {"existing_path": first_path, "new_path": alias_path})
        api_call("/api/vfs/fsync", {"path": "/"})
        rename = api_call("/api/vfs/rename", {"old_path": first_path, "new_path": first_path + ".moved"})
        graph = api_call("/api/graph")
        entries = {node["path"]: node["id"] for node in graph["nodes"] if node["id"].startswith("dirent-")}
        targets = {edge["source"]: edge["target"] for edge in graph["edges"] if edge["relation"] == "points-to"}
        assert targets[entries[alias_path]] == targets[entries[first_path + ".moved"]]
        assert any(row["id"] == transaction["id"] for row in api_call("/api/transactions"))
        assert api_call(f"/api/risk/{rename['id']}")["score"] is not None
        recovery = api_call("/api/crash/simulate", {
            "transactionId": rename["id"], "scenarioId": "crash-last-sync",
        })
        assert recovery["consistencyPassed"] and first_path in recovery["recoveredPaths"]
        assert api_call("/api/model")["status"] == "checkpoint_loaded"
        assert api_call("/api/research/gnn?scope=full")["nodes"]
        faultlab = api_call("/api/research/hardmode/trace?family=hardlink_retarget&version=v08")
        assert faultlab["nodes"] and faultlab["edges"]
        assert api_call("/api/research/hardmode/trace?family=package_upgrade&version=v09_s11")["nodes"]
        ready = api_call("/api/health")
        assert ready["gnn"] == "loaded" and ready["storage"] == "temporary"

        # A fresh browser must see a different namespace and transaction log.
        with browser.new_context() as other:
            isolated = other.new_page()
            assert isolated.goto(url, wait_until="domcontentloaded").ok
            expect(isolated.get_by_text("No transactions replayed yet")).to_be_visible(timeout=25_000)
            assert isolated.evaluate("async () => await (await fetch('/api/transactions')).json()") == []
        assert not page_errors, page_errors
        assert not console_errors, console_errors
        assert not bad_responses, bad_responses
        print("PASS: route loads and refreshes, UI mutation, live graph, hard link, risk, recovery, GNN, FaultLab, and session isolation")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("origin")
    parser.add_argument("--chromium", help="Optional path to a system Chromium executable")
    parser.add_argument("--offline", action="store_true", help="Expect the API to be unreachable")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        options = {"executable_path": args.chromium} if args.chromium else {}
        browser = playwright.chromium.launch(**options)
        smoke(browser, args.origin.rstrip("/"), args.offline)

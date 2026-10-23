"""Capture real MLflow UI screenshots for the usage PPT.

Default: http://127.0.0.1:5001 (project sqlite on that port).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mlflow.tracking import MlflowClient
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts" / "assets" / "dvc_mlflow_ppt"


def _pick_and_carry(tracking_uri: str) -> tuple[str | None, str | None]:
    client = MlflowClient(tracking_uri=tracking_uri)
    exps = [e for e in client.search_experiments() if e.name == "pick-and-carry"]
    if not exps:
        return None, None
    exp_id = exps[0].experiment_id
    runs = client.search_runs([exp_id], max_results=50, order_by=["attributes.start_time DESC"])
    scored = []
    for r in runs:
        name = r.data.tags.get("mlflow.runName", "")
        reward = r.data.metrics.get("reward")
        if reward is None:
            continue
        prefer = 2 if name == "carry" else 1 if "carry" in name else 0
        scored.append((prefer, reward, r.info.run_id))
    if not scored:
        return exp_id, (runs[0].info.run_id if runs else None)
    scored.sort(reverse=True)
    return exp_id, scored[0][2]


def _goto(page, url: str, settle: float = 2.0) -> None:
    # SPA + websockets rarely reach networkidle; load + settle is enough for screenshots.
    page.goto(url, wait_until="load", timeout=60000)
    time.sleep(settle)


def _dismiss(page) -> None:
    for label in ("Got it", "Close", "Close alert", "Close guidance", "Dismiss"):
        try:
            page.get_by_role("button", name=label).first.click(timeout=800)
        except Exception:
            pass


def capture(base_url: str, tracking_uri: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp_id, run_id = _pick_and_carry(tracking_uri)
    if not exp_id:
        raise SystemExit("pick-and-carry experiment not found in tracking DB")
    print(f"experiment_id={exp_id} run_id={run_id}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Home → Model training (not GenAI welcome)
        _goto(page, f"{base_url}/", settle=2.0)
        _dismiss(page)
        try:
            page.get_by_role("button", name="Model training").click(timeout=2000)
            time.sleep(1.5)
        except Exception:
            pass
        _dismiss(page)
        # Experiments list is a clearer "home" for RL usage slides
        try:
            page.get_by_role("link", name="Experiments").first.click(timeout=2000)
            time.sleep(2.0)
        except Exception:
            pass
        page.screenshot(path=str(OUT / "mlflow_home.png"), full_page=False)
        print("saved mlflow_home.png")

        # Experiment runs (classic ML training view) — clear any leftover search filter
        runs_url = (
            f"{base_url}/#/experiments/{exp_id}/runs"
            "?workflowType=machine_learning"
            "&searchFilter="
            "&orderByKey=attributes.start_time&orderByAsc=false"
            "&startTime=ALL&lifecycleFilter=Active"
        )
        _goto(page, runs_url, settle=3.0)
        try:
            page.get_by_role("button", name="Model training").click(timeout=2000)
            time.sleep(1.5)
        except Exception:
            pass
        _dismiss(page)
        # Clear search box if a demo filter is still visible
        try:
            search = page.locator('input[placeholder*="Search"], input[type="search"]').first
            if search.count():
                search.click(timeout=1000)
                search.fill("")
                page.keyboard.press("Enter")
                time.sleep(1.5)
        except Exception:
            pass
        try:
            page.get_by_role("button", name="Columns").click(timeout=2000)
            time.sleep(0.5)
            for metric in ("reward", "grasp", "ee_dist", "approach", "success", "walk_score"):
                loc = page.get_by_text(metric, exact=True)
                if loc.count():
                    try:
                        loc.first.click(timeout=500)
                    except Exception:
                        pass
            page.keyboard.press("Escape")
            time.sleep(0.8)
        except Exception:
            pass
        page.screenshot(path=str(OUT / "mlflow_experiment.png"), full_page=False)
        print("saved mlflow_experiment.png")

        # Chart view of runs (second angle for runs table slide)
        # View toggle (table | chart | evaluation) sits left of the search box: 2nd icon = chart view.
        try:
            page.mouse.click(295, 118)
            time.sleep(3.0)
        except Exception:
            pass
        page.screenshot(path=str(OUT / "mlflow_runs_table.png"), full_page=False)
        print("saved mlflow_runs_table.png")

        if run_id:
            detail = f"{base_url}/#/experiments/{exp_id}/runs/{run_id}?workflowType=machine_learning"
            _goto(page, detail, settle=3.0)
            _dismiss(page)
            # Overview = params / summary for run_detail slide
            for tab in ("Overview",):
                try:
                    page.get_by_role("tab", name=tab).click(timeout=1500)
                    time.sleep(1.5)
                except Exception:
                    try:
                        page.get_by_text(tab, exact=True).first.click(timeout=1000)
                        time.sleep(1.5)
                    except Exception:
                        pass
            page.screenshot(path=str(OUT / "mlflow_run_detail.png"), full_page=False)
            print(f"saved mlflow_run_detail.png ({run_id[:8]})")

            # Model metrics charts for curves slide
            for tab in ("Model metrics", "Metrics", "Charts"):
                try:
                    page.get_by_role("tab", name=tab).click(timeout=1500)
                    time.sleep(2.0)
                    break
                except Exception:
                    try:
                        page.get_by_text(tab, exact=True).first.click(timeout=1000)
                        time.sleep(2.0)
                        break
                    except Exception:
                        pass
            page.screenshot(path=str(OUT / "mlflow_metric_curves.png"), full_page=False)
            print("saved mlflow_metric_curves.png")

        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{(ROOT / 'logs' / 'mlflow' / 'mlflow.db').as_posix()}")
    args = parser.parse_args()
    capture(args.base_url.rstrip("/"), args.tracking_uri)


if __name__ == "__main__":
    main()

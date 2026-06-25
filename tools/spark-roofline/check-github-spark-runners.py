#!/usr/bin/env python3
"""Read-only GitHub runner check for the DGX Spark / GB10 gate.

This helper does not create workflows, dispatch jobs, register runners, or
modify repository settings. It scans checked-in workflow `runs-on` labels and,
when permitted, queries GitHub's repository self-hosted runner inventory through
`gh api`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SPARK_MARKERS = (
    "sm_121",
    "sm121",
    "121a",
    "gb10",
    "dgx spark",
    "dgx-spark",
    "dgx_spark",
    "spark",
)


RUNS_ON_RE = re.compile(r"^\s*runs-on:\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repo for runner inventory, e.g. jethac/llama.cpp")
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path(".github/workflows"),
        help="local workflow directory to scan",
    )
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON")
    return parser.parse_args()


def run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def default_repo() -> str:
    result = run_gh(["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or "gh repo view failed")
    repo = result.stdout.strip()
    if not repo:
        raise RuntimeError("gh repo view returned an empty repository name")
    return repo


def normalize_runs_on(value: str) -> list[str]:
    value = value.split("#", 1)[0].strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    if value.startswith("${{"):
        return [value]
    return [value.strip().strip("'\"")]


def scan_workflows(workflow_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not workflow_dir.is_dir():
        return rows
    for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            match = RUNS_ON_RE.match(line)
            if not match:
                continue
            labels = normalize_runs_on(match.group(1))
            rows.append(
                {
                    "path": str(path),
                    "line": lineno,
                    "raw": match.group(1).strip(),
                    "labels": labels,
                    "spark_like": is_spark_like(labels),
                }
            )
    return rows


def is_spark_like(labels: list[str]) -> bool:
    text = " ".join(labels).lower()
    return any(marker in text for marker in SPARK_MARKERS)


def query_runners(repo: str) -> tuple[bool, str, list[dict[str, Any]]]:
    result = run_gh(["api", f"repos/{repo}/actions/runners", "--paginate"])
    if result.returncode != 0:
        return False, result.stdout.strip(), []
    runners: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            return False, f"invalid gh api JSON: {exc}", []
        for runner in payload.get("runners", []):
            if not isinstance(runner, dict):
                continue
            labels = [str(label.get("name", "")) for label in runner.get("labels", []) if isinstance(label, dict)]
            runners.append(
                {
                    "name": runner.get("name"),
                    "os": runner.get("os"),
                    "status": runner.get("status"),
                    "busy": runner.get("busy"),
                    "labels": labels,
                    "spark_like": is_spark_like(labels + [str(runner.get("name", ""))]),
                }
            )
    return True, "", runners


def main() -> int:
    args = parse_args()
    try:
        repo = args.repo or default_repo()
    except RuntimeError as exc:
        print(f"failed_to_resolve_repo={exc}", file=sys.stderr)
        return 3

    workflows = scan_workflows(args.workflow_dir)
    runner_query_complete, runner_error, runners = query_runners(repo)
    spark_workflows = [row for row in workflows if row["spark_like"]]
    spark_runners = [runner for runner in runners if runner["spark_like"]]

    payload = {
        "repo": repo,
        "workflow_dir": str(args.workflow_dir),
        "workflow_runs_on_count": len(workflows),
        "spark_like_workflow_runs_on": spark_workflows,
        "runner_query_complete": runner_query_complete,
        "runner_error": runner_error,
        "runner_count": len(runners),
        "spark_like_runners": spark_runners,
        "passed": bool(spark_runners),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"repo={payload['repo']}")
        print(f"workflow_runs_on_count={payload['workflow_runs_on_count']}")
        print(f"spark_like_workflow_runs_on_count={len(spark_workflows)}")
        for row in spark_workflows:
            print(f"  workflow={row['path']}:{row['line']} runs-on={row['raw']}")
        print(f"runner_query_complete={str(runner_query_complete).lower()}")
        if runner_error:
            print(f"runner_error={runner_error}")
        print(f"runner_count={len(runners)}")
        print(f"spark_like_runner_count={len(spark_runners)}")
        for runner in spark_runners:
            print(
                f"  runner={runner['name']} os={runner['os']} status={runner['status']} "
                f"busy={runner['busy']} labels={','.join(runner['labels'])}"
            )
        print(f"passed={str(payload['passed']).lower()}")

    if spark_runners:
        return 0
    if runner_query_complete:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only GitHub search for upstream llama.cpp NVFP4 KV/Spark work.

This helper never writes to GitHub. It shells out to `gh` search commands and
classifies current public PRs/issues so the Spark NVFP4 KV lane can distinguish
directly competing work from adjacent NVFP4 substrate work.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


REPO = "ggml-org/llama.cpp"

DEFAULT_SEARCHES = [
    ("prs", "NVFP4 KV"),
    ("issues", "NVFP4 KV"),
    ("prs", "Spark NVFP4"),
    ("issues", "Spark NVFP4"),
    ("prs", "FP4 KV"),
    ("issues", "FP4 KV"),
    ("prs", "NVFP4"),
    ("issues", "NVFP4"),
]

EXHAUSTIVE_SEARCHES = [
    ("prs", "NVFP4 KV"),
    ("issues", "NVFP4 KV"),
    ("prs", "Spark NVFP4"),
    ("issues", "Spark NVFP4"),
    ("prs", "Blackwell NVFP4 FlashAttention"),
    ("issues", "Blackwell NVFP4 FlashAttention"),
    ("prs", "Gemma NVFP4"),
    ("issues", "Gemma NVFP4"),
    ("prs", "FP4 KV"),
    ("issues", "FP4 KV"),
    ("prs", "quantized KV FlashAttention"),
    ("issues", "quantized KV FlashAttention"),
    ("prs", "NVFP4"),
    ("issues", "NVFP4"),
]

DIRECT_MARKERS = (
    ("nvfp4", "kv", "flashattention"),
    ("nvfp4", "kv", "fattn"),
    ("nvfp4", "kv-cache", "flashattention"),
    ("nvfp4", "kv cache", "flashattention"),
    ("fp4", "kv", "flashattention"),
    ("fp4", "kv-cache", "flashattention"),
    ("spark", "nvfp4", "kv"),
    ("gb10", "nvfp4", "kv"),
)

ADJACENT_MARKERS = (
    "nvfp4",
    "fp4",
    "blackwell",
    "gemma4",
    "gemma 4",
    "dgx spark",
    "gb10",
)


@dataclass
class GitHubItem:
    kind: str
    number: int
    title: str
    state: str
    author: str
    updated_at: str
    url: str
    matched_queries: list[str]
    classification: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO, help=f"GitHub repository to search (default: {REPO})")
    parser.add_argument("--limit", type=int, default=50, help="maximum results per query")
    parser.add_argument("--exhaustive", action="store_true", help="run the broader query set; uses more GitHub search quota")
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON")
    return parser.parse_args()


def run_gh(kind: str, query: str, repo: str, limit: int) -> list[dict[str, Any]]:
    cmd = [
        "gh",
        "search",
        kind,
        "--repo",
        repo,
        query,
        "--limit",
        str(limit),
        "--json",
        "number,title,state,author,updatedAt,url",
    ]
    completed = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"gh search {kind} failed for {query!r}")
    payload = json.loads(completed.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected gh JSON payload for {kind} {query!r}: {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def classify(title: str) -> tuple[str, str]:
    text = " ".join(title.lower().replace("_", " ").replace("-", " ").split())
    for markers in DIRECT_MARKERS:
        if all(marker in text for marker in markers):
            return "direct", "title matches NVFP4/FP4 KV FlashAttention/Spark markers"
    if any(marker in text for marker in ADJACENT_MARKERS):
        return "adjacent", "title is NVFP4/FP4/Blackwell/Gemma/Spark substrate, not direct KV FlashAttention"
    return "other", "matched broad search but title is not obviously relevant"


def make_key(kind: str, item: dict[str, Any]) -> tuple[str, int]:
    return kind, int(item.get("number") or 0)


def item_author(item: dict[str, Any]) -> str:
    author = item.get("author")
    if isinstance(author, dict):
        return str(author.get("login") or "")
    return ""


def main() -> int:
    args = parse_args()
    if shutil.which("gh") is None:
        print("missing gh CLI; install GitHub CLI or run from an environment with gh on PATH", file=sys.stderr)
        return 2

    items: dict[tuple[str, int], GitHubItem] = {}
    errors: list[str] = []
    searches = EXHAUSTIVE_SEARCHES if args.exhaustive else DEFAULT_SEARCHES

    for kind, query in searches:
        try:
            rows = run_gh(kind, query, args.repo, args.limit)
        except Exception as exc:  # noqa: BLE001 - discovery should collect all query failures.
            errors.append(f"{kind} {query!r}: {exc}")
            continue

        for row in rows:
            key = make_key(kind, row)
            classification, reason = classify(str(row.get("title") or ""))
            if key not in items:
                items[key] = GitHubItem(
                    kind=kind,
                    number=key[1],
                    title=str(row.get("title") or ""),
                    state=str(row.get("state") or ""),
                    author=item_author(row),
                    updated_at=str(row.get("updatedAt") or ""),
                    url=str(row.get("url") or ""),
                    matched_queries=[],
                    classification=classification,
                    reason=reason,
                )
            items[key].matched_queries.append(query)
            if classification == "direct":
                items[key].classification = classification
                items[key].reason = reason

    rows = sorted(items.values(), key=lambda item: (item.classification != "direct", item.kind, item.number))
    direct = [item for item in rows if item.classification == "direct"]
    adjacent = [item for item in rows if item.classification == "adjacent"]

    payload = {
        "repo": args.repo,
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "search_mode": "exhaustive" if args.exhaustive else "default",
        "query_count": len(searches),
        "errors": errors,
        "complete": not errors,
        "direct_count": len(direct),
        "adjacent_count": len(adjacent),
        "direct": [item.__dict__ for item in direct],
        "adjacent": [item.__dict__ for item in adjacent],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"repo={payload['repo']}")
        print(f"checked_at_utc={payload['checked_at_utc']}")
        print(f"search_mode={payload['search_mode']}")
        print(f"complete={str(payload['complete']).lower()}")
        print(f"direct_count={payload['direct_count']}")
        print(f"adjacent_count={payload['adjacent_count']}")
        if errors:
            for error in errors:
                print(f"error={error}")
        if direct:
            print("direct:")
            for item in direct:
                print(f"  {item.kind[:-1]} #{item.number} {item.state} {item.title} {item.url}")
        else:
            print("direct: none")
        print("adjacent:")
        for item in adjacent[:30]:
            print(
                f"  {item.kind[:-1]} #{item.number} {item.state} {item.title} "
                f"author={item.author} updated={item.updated_at}"
            )

    if errors:
        return 2
    return 0 if not direct else 1


if __name__ == "__main__":
    raise SystemExit(main())

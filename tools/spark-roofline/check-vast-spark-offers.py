#!/usr/bin/env python3
"""Read-only Vast.ai offer check for the DGX Spark / GB10 gate.

This script never creates, launches, starts, stops, or modifies instances. It
only searches the public offer catalog and classifies whether any result looks
usable for the llama.cpp sm_121a Spark gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any


SPARK_NAME_MARKERS = (
    "gb10",
    "dgx spark",
    "dgx-spark",
    "dgx_spark",
)


@dataclass
class OfferSummary:
    offer_id: Any
    gpu_name: str
    compute_cap: Any
    num_gpus: Any
    gpu_ram_gb: float | None
    dph_total: Any
    cuda_max_good: Any
    driver_version: Any
    geolocation: Any
    reliability: Any
    static_ip: Any
    reason: str


def load_vast_client():
    try:
        import vastai  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The Python package 'vastai' is not installed. Install it or run the check from an environment "
            "with Vast.ai SDK support."
        ) from exc

    return vastai.VastAI(raw=True, quiet=True)


def as_gb(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    # Vast reports GPU RAM in MiB-like units in the current API payload.
    return numeric / 1024.0 if numeric > 1024 else numeric


def summarize(offer: dict[str, Any], reason: str) -> OfferSummary:
    return OfferSummary(
        offer_id=offer.get("id"),
        gpu_name=str(offer.get("gpu_name", "")),
        compute_cap=offer.get("compute_cap"),
        num_gpus=offer.get("num_gpus"),
        gpu_ram_gb=as_gb(offer.get("gpu_ram")),
        dph_total=offer.get("dph_total"),
        cuda_max_good=offer.get("cuda_max_good"),
        driver_version=offer.get("driver_version"),
        geolocation=offer.get("geolocation"),
        reliability=offer.get("reliability"),
        static_ip=offer.get("static_ip"),
        reason=reason,
    )


def is_spark_candidate(offer: dict[str, Any]) -> tuple[bool, str]:
    name = str(offer.get("gpu_name", "")).lower()
    compute_cap = offer.get("compute_cap")

    if any(marker in name for marker in SPARK_NAME_MARKERS):
        return True, "gpu name looks like DGX Spark/GB10"

    try:
        cc_int = int(compute_cap)
    except (TypeError, ValueError):
        return False, "missing or non-numeric compute_cap"

    if cc_int >= 1210:
        return True, "compute_cap is at least 1210, matching sm_121-class target"
    if cc_int == 1200:
        return False, "compute_cap=1200 is Blackwell sm_120-class, not the sm_121a Spark target"
    if cc_int == 1000:
        return False, "compute_cap=1000 is Blackwell datacenter sm_100-class, not the sm_121a Spark target"
    return False, f"compute_cap={cc_int} is not sm_121-class"


def search(client: Any, query: str, limit: int, order: str) -> list[dict[str, Any]]:
    results = client.search_offers(query=query, limit=limit, order=order, type="on-demand")
    if not isinstance(results, list):
        raise RuntimeError(f"unexpected Vast.ai search result type for {query!r}: {type(results).__name__}")
    return [result for result in results if isinstance(result, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="maximum offers per search query")
    parser.add_argument("--order", default="score-", help="Vast.ai offer sort order")
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON")
    args = parser.parse_args()

    client = load_vast_client()
    queries = [
        "compute_cap>=1210 rentable=true",
        "compute_cap>=1200 rentable=true",
        "gpu_name in [GB10,DGX_Spark,DGX-Spark,DGX_Spark] rentable=true",
    ]

    offers_by_id: dict[Any, dict[str, Any]] = {}
    errors: list[str] = []
    for query in queries:
        try:
            for offer in search(client, query, args.limit, args.order):
                offer.setdefault("_matched_queries", []).append(query)
                offers_by_id[offer.get("id")] = offer
        except Exception as exc:  # noqa: BLE001 - keep discovery failure visible.
            errors.append(f"{query}: {exc}")

    candidates: list[OfferSummary] = []
    near_misses: list[OfferSummary] = []
    counts = Counter()
    for offer in offers_by_id.values():
        name = str(offer.get("gpu_name", "unknown"))
        counts[(name, offer.get("compute_cap"))] += 1
        ok, reason = is_spark_candidate(offer)
        summary = summarize(offer, reason)
        if ok:
            candidates.append(summary)
        else:
            near_misses.append(summary)

    candidates.sort(key=lambda item: (float(item.dph_total or 0), str(item.gpu_name)))
    near_misses.sort(key=lambda item: (str(item.compute_cap), str(item.gpu_name), float(item.dph_total or 0)))

    payload = {
        "passed": bool(candidates),
        "candidate_count": len(candidates),
        "searched_offer_count": len(offers_by_id),
        "errors": errors,
        "counts_by_gpu_name_and_compute_cap": [
            {"gpu_name": name, "compute_cap": cc, "count": count}
            for (name, cc), count in sorted(counts.items(), key=lambda item: (str(item[0][1]), item[0][0]))
        ],
        "spark_candidates": [summary.__dict__ for summary in candidates[:20]],
        "near_misses": [summary.__dict__ for summary in near_misses[:20]],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"passed={str(payload['passed']).lower()}")
        print(f"candidate_count={payload['candidate_count']}")
        print(f"searched_offer_count={payload['searched_offer_count']}")
        if errors:
            for error in errors:
                print(f"error={error}")
        print("counts_by_gpu_name_and_compute_cap:")
        for row in payload["counts_by_gpu_name_and_compute_cap"]:
            print(f"  {row['gpu_name']} compute_cap={row['compute_cap']} count={row['count']}")
        if candidates:
            print("spark_candidates:")
            for item in candidates[:20]:
                print(
                    f"  id={item.offer_id} gpu={item.gpu_name} compute_cap={item.compute_cap} "
                    f"num_gpus={item.num_gpus} dph_total={item.dph_total} reason={item.reason}"
                )
        else:
            print("spark_candidates: none")
            print("near_misses:")
            for item in near_misses[:10]:
                print(
                    f"  id={item.offer_id} gpu={item.gpu_name} compute_cap={item.compute_cap} "
                    f"num_gpus={item.num_gpus} dph_total={item.dph_total} reason={item.reason}"
                )

    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())

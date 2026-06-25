#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path


RUN_RE = re.compile(r"^kq256-mixedpv-threads-(\d+)\.log$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize llama-spark-kq256-mixedpv gate logs.")
    parser.add_argument("--dir", required=True, type=Path, help="Artifact directory from the mixed-PV run")
    parser.add_argument("--csv", required=True, type=Path, help="CSV summary output path")
    parser.add_argument("--json", required=True, type=Path, help="JSON summary output path")
    parser.add_argument("--text", required=True, type=Path, help="Human-readable summary output path")
    parser.add_argument("--requested-device", default="", help="Device id requested by the run script")
    parser.add_argument("--require-spark", action="store_true", help="Fail if any parsed run is not spark_target: yes")
    return parser.parse_args()


def parse_bool(value: str) -> bool | None:
    value = value.strip().lower()
    if value in {"yes", "true"}:
        return True
    if value in {"no", "false"}:
        return False
    return None


def parse_log(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "log": str(path),
    }

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^device:\s+(\d+)\s+(.+)$", stripped)
        if match:
            row["device"] = int(match.group(1))
            row["device_name"] = match.group(2)
            continue

        match = re.match(r"^compute_capability:\s+sm_(\d+)\s+cc=(\d+)$", stripped)
        if match:
            row["sm"] = f"sm_{match.group(1)}"
            row["cc"] = int(match.group(2))
            continue

        match = re.match(r"^spark_target:\s+(\S+)$", stripped)
        if match:
            row["spark_target"] = parse_bool(match.group(1))
            continue

        match = re.match(r"^blackwell_fp4_target:\s+(\S+)$", stripped)
        if match:
            row["blackwell_fp4_target"] = parse_bool(match.group(1))
            continue

        match = re.match(r"^sms:\s+(\d+)$", stripped)
        if match:
            row["sms"] = int(match.group(1))
            continue

        match = re.match(
            r"^config:\s+blocks=(\d+)\s+threads=(\d+)\s+mtp_rows=(\d+)\s+"
            r"useful_m16=([0-9.]+)%\s+iters=(\d+)$",
            stripped,
        )
        if match:
            row["blocks"] = int(match.group(1))
            row["threads"] = int(match.group(2))
            row["mtp_rows"] = int(match.group(3))
            row["useful_m16_pct"] = float(match.group(4))
            row["iters"] = int(match.group(5))
            continue

        match = re.match(
            r"^(kq256_only|pv256_mixed|combined256_mixed) occupancy:\s+"
            r"active_blocks_per_sm=(\d+)\s+active_warps_per_sm=(\d+)\s+"
            r"occupancy=([0-9.]+)%\s+shared=([0-9.]+)\s+KiB$",
            stripped,
        )
        if match:
            prefix = match.group(1)
            row[f"{prefix}_active_blocks_per_sm"] = int(match.group(2))
            row[f"{prefix}_active_warps_per_sm"] = int(match.group(3))
            row[f"{prefix}_occupancy_pct"] = float(match.group(4))
            row[f"{prefix}_shared_kib"] = float(match.group(5))
            continue

        match = re.match(
            r"^q_quant:\s+([0-9.]+)\s+GB/s-input\s+([0-9.]+)\s+GB/s-output\s+"
            r"([0-9.]+)\s+GB/s-total\s+blocks=(\d+)\s+repeats=(\d+)\s+time=([0-9.]+)\s+ms$",
            stripped,
        )
        if match:
            row["q_quant_gbps_input"] = float(match.group(1))
            row["q_quant_gbps_output"] = float(match.group(2))
            row["q_quant_gbps_total"] = float(match.group(3))
            row["q_quant_blocks"] = int(match.group(4))
            row["q_quant_repeats"] = int(match.group(5))
            row["q_quant_ms"] = float(match.group(6))
            continue

        match = re.match(
            r"^kq256:\s+([0-9.]+)\s+KQ-TOPS\s+([0-9.]+)\s+useful-mtp-KQ-TOPS\s+"
            r"([0-9.]+)\s+GB/s-K-compact-read\s+([0-9.]+)\s+GB-Q-compact-once\s+"
            r"blocks=(\d+)\s+warps=(\d+)\s+iters=(\d+)\s+time=([0-9.]+)\s+ms$",
            stripped,
        )
        if match:
            row["kq_tops"] = float(match.group(1))
            row["useful_mtp_kq_tops"] = float(match.group(2))
            row["k_compact_read_gbps"] = float(match.group(3))
            row["q_compact_once_gb"] = float(match.group(4))
            row["kq_blocks"] = int(match.group(5))
            row["warps"] = int(match.group(6))
            row["kq_iters"] = int(match.group(7))
            row["kq_ms"] = float(match.group(8))
            continue

        match = re.match(
            r"^pv256_mixed:\s+([0-9.]+)\s+mixedPV-TOPS\s+([0-9.]+)\s+"
            r"useful-mtp-mixedPV-TOPS\s+([0-9.]+)\s+GB/s-V-compact-read\s+"
            r"blocks=(\d+)\s+warps=(\d+)\s+iters=(\d+)\s+time=([0-9.]+)\s+ms$",
            stripped,
        )
        if match:
            row["mixedpv_tops"] = float(match.group(1))
            row["useful_mtp_mixedpv_tops"] = float(match.group(2))
            row["v_compact_read_gbps"] = float(match.group(3))
            row["pv_blocks"] = int(match.group(4))
            row["pv_warps"] = int(match.group(5))
            row["pv_iters"] = int(match.group(6))
            row["pv_ms"] = float(match.group(7))
            continue

        match = re.match(
            r"^combined256_mixed:\s+([0-9.]+)\s+modeled-total-TOPS\s+([0-9.]+)\s+"
            r"useful-mtp-modeled-total-TOPS\s+([0-9.]+)\s+KQ-issue-TOPS\s+"
            r"([0-9.]+)\s+mixedPV-issue-TOPS\s+([0-9.]+)\s+GB/s-K-compact-read\s+"
            r"([0-9.]+)\s+GB/s-V-compact-read\s+blocks=(\d+)\s+warps=(\d+)\s+"
            r"iters=(\d+)\s+time=([0-9.]+)\s+ms$",
            stripped,
        )
        if match:
            row["combined_total_tops"] = float(match.group(1))
            row["useful_mtp_combined_total_tops"] = float(match.group(2))
            row["combined_kq_issue_tops"] = float(match.group(3))
            row["combined_mixedpv_issue_tops"] = float(match.group(4))
            row["combined_k_compact_read_gbps"] = float(match.group(5))
            row["combined_v_compact_read_gbps"] = float(match.group(6))
            row["combined_blocks"] = int(match.group(7))
            row["combined_warps"] = int(match.group(8))
            row["combined_iters"] = int(match.group(9))
            row["combined_ms"] = float(match.group(10))
            continue

    return row


def numeric_row_value(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def main() -> int:
    args = parse_args()
    if not args.dir.is_dir():
        print(f"missing artifact directory: {args.dir}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for path in sorted(args.dir.iterdir()):
        match = RUN_RE.match(path.name)
        if not match:
            continue

        row = parse_log(path)
        row.setdefault("threads", int(match.group(1)))
        rows.append(row)

    if not rows:
        print(f"no kq256 mixed-PV thread logs found under {args.dir}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for row in rows:
        threads = row.get("threads", "?")
        if "kq_tops" not in row:
            failures.append(f"threads={threads}: missing kq256 result line")
        if "mixedpv_tops" not in row:
            failures.append(f"threads={threads}: missing pv256_mixed result line")
        if "combined_total_tops" not in row:
            failures.append(f"threads={threads}: missing combined256_mixed result line")
        if args.require_spark and row.get("spark_target") is not True:
            failures.append(f"threads={threads}: spark_target is not yes")

    best_row = max(rows, key=lambda row: numeric_row_value(row, "useful_mtp_combined_total_tops"))
    best = {
        "threads": best_row.get("threads", ""),
        "useful_mtp_combined_total_tops": best_row.get("useful_mtp_combined_total_tops", ""),
        "combined_total_tops": best_row.get("combined_total_tops", ""),
        "combined_kq_issue_tops": best_row.get("combined_kq_issue_tops", ""),
        "combined_mixedpv_issue_tops": best_row.get("combined_mixedpv_issue_tops", ""),
        "combined_k_compact_read_gbps": best_row.get("combined_k_compact_read_gbps", ""),
        "combined_v_compact_read_gbps": best_row.get("combined_v_compact_read_gbps", ""),
    }
    passed = len(failures) == 0
    all_spark = all(row.get("spark_target") is True for row in rows)
    if not passed:
        gate_decision = "no-go"
    elif all_spark:
        gate_decision = "needs-interpretation"
    else:
        gate_decision = "proxy-pass"

    if passed and all_spark:
        next_action = "Compare mixed-PV combined rows against KQ-only and decide whether D=256 should move to backend POC."
    elif passed:
        next_action = "Proxy mixed-PV run passed relaxed checks; repeat on Spark before making a D=256 go/no-go decision."
    else:
        next_action = "Resolve the listed failures before interpreting D=256 mixed-PV."

    fieldnames = [
        "requested_device",
        "device",
        "threads",
        "mtp_rows",
        "iters",
        "spark_target",
        "blackwell_fp4_target",
        "sm",
        "sms",
        "kq_tops",
        "useful_mtp_kq_tops",
        "mixedpv_tops",
        "useful_mtp_mixedpv_tops",
        "combined_total_tops",
        "useful_mtp_combined_total_tops",
        "combined_kq_issue_tops",
        "combined_mixedpv_issue_tops",
        "k_compact_read_gbps",
        "v_compact_read_gbps",
        "combined_k_compact_read_gbps",
        "combined_v_compact_read_gbps",
        "q_quant_gbps_total",
        "kq256_only_occupancy_pct",
        "pv256_mixed_occupancy_pct",
        "combined256_mixed_occupancy_pct",
        "kq_ms",
        "pv_ms",
        "combined_ms",
        "device_name",
        "log",
    ]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["requested_device"] = args.requested_device
            writer.writerow(row)

    summary = {
        "artifact_dir": str(args.dir),
        "requested_device": args.requested_device,
        "rows": rows,
        "best": best,
        "failures": failures,
        "passed": passed,
        "all_spark": all_spark,
        "gate_decision": gate_decision,
        "next_action": next_action,
    }
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"artifact_dir={args.dir}",
        f"requested_device={args.requested_device}",
        f"rows={len(rows)}",
        f"passed={str(summary['passed']).lower()}",
        f"gate_decision={gate_decision}",
        f"next_action={next_action}",
        f"best_threads={best['threads']}",
        f"best_useful_mtp_combined_total_tops={best['useful_mtp_combined_total_tops']}",
        f"best_combined_total_tops={best['combined_total_tops']}",
        f"best_combined_kq_issue_tops={best['combined_kq_issue_tops']}",
        f"best_combined_mixedpv_issue_tops={best['combined_mixedpv_issue_tops']}",
        f"best_combined_k_read_gbps={best['combined_k_compact_read_gbps']}",
        f"best_combined_v_read_gbps={best['combined_v_compact_read_gbps']}",
    ]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for failure in failures:
            lines.append(f"- {failure}")
    lines.append("")
    lines.append("KQ256 mixed-PV rows:")
    for row in rows:
        lines.append(
            "threads={threads} spark_target={spark_target} kq_tops={kq_tops} "
            "mixedpv_tops={mixedpv_tops} combined_total_tops={combined_total_tops} "
            "useful_mtp_combined_total_tops={useful_mtp_combined_total_tops} "
            "combined_kq_issue_tops={combined_kq_issue_tops} "
            "combined_mixedpv_issue_tops={combined_mixedpv_issue_tops} "
            "combined_k_read_gbps={combined_k_compact_read_gbps} "
            "combined_v_read_gbps={combined_v_compact_read_gbps} device={device} sm={sm}".format(
                threads=row.get("threads", ""),
                spark_target=row.get("spark_target", ""),
                kq_tops=row.get("kq_tops", ""),
                mixedpv_tops=row.get("mixedpv_tops", ""),
                combined_total_tops=row.get("combined_total_tops", ""),
                useful_mtp_combined_total_tops=row.get("useful_mtp_combined_total_tops", ""),
                combined_kq_issue_tops=row.get("combined_kq_issue_tops", ""),
                combined_mixedpv_issue_tops=row.get("combined_mixedpv_issue_tops", ""),
                combined_k_compact_read_gbps=row.get("combined_k_compact_read_gbps", ""),
                combined_v_compact_read_gbps=row.get("combined_v_compact_read_gbps", ""),
                device=row.get("device", ""),
                sm=row.get("sm", ""),
            )
        )
    args.text.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

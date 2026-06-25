#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path


RUN_RE = re.compile(r"^kq256-threads-(\d+)\.log$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize llama-spark-kq256 gate logs.")
    parser.add_argument("--dir", required=True, type=Path, help="Artifact directory from run-spark-kq256-gate.sh")
    parser.add_argument("--csv", required=True, type=Path, help="CSV summary output path")
    parser.add_argument("--json", required=True, type=Path, help="JSON summary output path")
    parser.add_argument("--text", required=True, type=Path, help="Human-readable summary output path")
    parser.add_argument("--requested-device", default="", help="Device id requested by run-spark-kq256-gate.sh")
    parser.add_argument("--require-spark", action="store_true", help="Fail if any parsed run is not spark_target: yes")
    parser.add_argument("--require-ncu", action="store_true", help="Fail if any parsed run lacks passing NCU FP4 evidence")
    return parser.parse_args()


def parse_bool(value: str) -> bool | None:
    value = value.strip().lower()
    if value == "yes" or value == "true":
        return True
    if value == "no" or value == "false":
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
            r"^kq256 occupancy:\s+active_blocks_per_sm=(\d+)\s+active_warps_per_sm=(\d+)\s+"
            r"occupancy=([0-9.]+)%\s+shared=([0-9.]+)\s+KiB$",
            stripped,
        )
        if match:
            row["active_blocks_per_sm"] = int(match.group(1))
            row["active_warps_per_sm"] = int(match.group(2))
            row["occupancy_pct"] = float(match.group(3))
            row["shared_kib"] = float(match.group(4))
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

    return row


def load_ncu_evidence(out_dir: Path, threads: int) -> dict[str, object]:
    path = out_dir / f"ncu-kq256-threads-{threads}-fp4-evidence.json"
    if not path.is_file():
        return {
            "ncu_evidence_path": "",
            "ncu_passed": "",
            "ncu_nonzero_fp4_rows": "",
            "ncu_nonzero_tensor_rows": "",
        }

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {
        "ncu_evidence_path": str(path),
        "ncu_passed": bool(data.get("passed", False)),
        "ncu_nonzero_fp4_rows": int(data.get("nonzero_fp4_rows", 0)),
        "ncu_nonzero_tensor_rows": int(data.get("nonzero_tensor_rows", 0)),
    }


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
        threads = int(match.group(1))
        row.setdefault("threads", threads)
        row.update(load_ncu_evidence(args.dir, threads))
        rows.append(row)

    if not rows:
        print(f"no kq256 thread logs found under {args.dir}", file=sys.stderr)
        return 2

    failures: list[str] = []
    any_ncu_passed = any(row.get("ncu_passed") is True for row in rows)
    for row in rows:
        threads = row.get("threads", "?")
        if "kq_tops" not in row:
            failures.append(f"threads={threads}: missing kq256 result line")
        if args.require_spark and row.get("spark_target") is not True:
            failures.append(f"threads={threads}: spark_target is not yes")
        if args.require_ncu and row.get("ncu_evidence_path") and row.get("ncu_passed") is not True:
            failures.append(f"threads={threads}: NCU FP4 evidence did not pass")
    if args.require_ncu and not any_ncu_passed:
        failures.append("no passing NCU FP4 evidence artifact found")

    best_row = max(rows, key=lambda row: numeric_row_value(row, "useful_mtp_kq_tops"))
    best = {
        "threads": best_row.get("threads", ""),
        "useful_mtp_kq_tops": best_row.get("useful_mtp_kq_tops", ""),
        "kq_tops": best_row.get("kq_tops", ""),
        "k_compact_read_gbps": best_row.get("k_compact_read_gbps", ""),
        "q_quant_gbps_total": best_row.get("q_quant_gbps_total", ""),
        "occupancy_pct": best_row.get("occupancy_pct", ""),
        "ncu_passed": best_row.get("ncu_passed", ""),
    }
    passed = len(failures) == 0
    all_spark = all(row.get("spark_target") is True for row in rows)
    if not passed:
        gate_decision = "no-go"
    elif all_spark:
        gate_decision = "go"
    else:
        gate_decision = "proxy-pass"

    if passed and all_spark:
        next_action = "Record Spark sm_121a numbers and decide whether to proceed to D=256 mixed PV."
    elif passed:
        next_action = "Proxy run passed relaxed checks; do not treat D=256 KQ as green until a Spark run passes."
    else:
        next_action = "Resolve the listed failures before treating D=256 KQ as green."

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
        "k_compact_read_gbps",
        "q_quant_gbps_total",
        "occupancy_pct",
        "active_blocks_per_sm",
        "active_warps_per_sm",
        "ncu_passed",
        "ncu_nonzero_fp4_rows",
        "ncu_nonzero_tensor_rows",
        "kq_ms",
        "q_quant_ms",
        "device_name",
        "log",
        "ncu_evidence_path",
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
        f"best_useful_mtp_kq_tops={best['useful_mtp_kq_tops']}",
        f"best_kq_tops={best['kq_tops']}",
        f"best_k_read_gbps={best['k_compact_read_gbps']}",
        f"best_q_quant_gbps={best['q_quant_gbps_total']}",
    ]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for failure in failures:
            lines.append(f"- {failure}")
    lines.append("")
    lines.append("KQ256 rows:")
    for row in rows:
        lines.append(
            "threads={threads} spark_target={spark_target} kq_tops={kq_tops} "
            "useful_mtp_kq_tops={useful_mtp_kq_tops} k_read_gbps={k_compact_read_gbps} "
            "q_quant_gbps={q_quant_gbps_total} device={device} sm={sm} ncu_passed={ncu_passed}".format(
                threads=row.get("threads", ""),
                spark_target=row.get("spark_target", ""),
                kq_tops=row.get("kq_tops", ""),
                useful_mtp_kq_tops=row.get("useful_mtp_kq_tops", ""),
                k_compact_read_gbps=row.get("k_compact_read_gbps", ""),
                q_quant_gbps_total=row.get("q_quant_gbps_total", ""),
                device=row.get("device", ""),
                sm=row.get("sm", ""),
                ncu_passed=row.get("ncu_passed", ""),
            )
        )
    args.text.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

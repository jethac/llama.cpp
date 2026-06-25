#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path


DEFAULT_EVIDENCE_RE = (
    r"fp4|mxf4|mxf8|f6f4|e2m1|"
    r"pipe_tensor|tcgen|wgmma|hmma|"
    r"mma|tensor"
)
DEFAULT_FP4_RE = r"fp4|mxf4|mxf8|f6f4|e2m1"
DEFAULT_TENSOR_RE = r"pipe_tensor|tcgen|wgmma|hmma|mma|tensor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Nsight Compute raw CSV evidence for FP4/tensor-core activity.",
    )
    parser.add_argument("--csv", required=True, type=Path, help="Nsight Compute raw CSV from ncu --import --page raw --csv")
    parser.add_argument("--out", required=True, type=Path, help="JSON summary output path")
    parser.add_argument("--text", required=True, type=Path, help="Human-readable summary output path")
    parser.add_argument("--kernel-regex", default=r"kq256_operand_kernel", help="Kernel name regex to look for")
    parser.add_argument("--evidence-regex", default=DEFAULT_EVIDENCE_RE, help="Metric-name evidence regex")
    parser.add_argument("--fp4-regex", default=DEFAULT_FP4_RE, help="FP4-specific evidence regex")
    parser.add_argument("--tensor-regex", default=DEFAULT_TENSOR_RE, help="Tensor/MMA evidence regex")
    parser.add_argument("--max-rows", default=80, type=int, help="Maximum candidate rows to include")
    parser.add_argument("--allow-tensor-only", action="store_true", help="Allow generic tensor/MMA evidence without FP4-specific evidence")
    parser.add_argument("--allow-empty", action="store_true", help="Exit 0 even when no nonzero evidence is found")
    return parser.parse_args()


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None

    # Nsight can emit instanced metrics as "1; 2; 3". Any nonzero instance is evidence.
    if ";" in value:
        nums = [parse_number(part) for part in value.split(";")]
        nums = [num for num in nums if num is not None]
        if not nums:
            return None
        return max(nums, key=abs)

    value = value.replace(",", "")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def row_numbers(row: list[str]) -> list[float]:
    nums: list[float] = []
    for cell in row:
        num = parse_number(cell)
        if num is not None:
            nums.append(num)
    return nums


def main() -> int:
    args = parse_args()
    evidence_re = re.compile(args.evidence_regex, re.IGNORECASE)
    fp4_re = re.compile(args.fp4_regex, re.IGNORECASE)
    tensor_re = re.compile(args.tensor_regex, re.IGNORECASE)
    kernel_re = re.compile(args.kernel_regex)

    if not args.csv.is_file():
        print(f"missing raw CSV: {args.csv}", file=sys.stderr)
        return 2

    rows_seen = 0
    kernel_rows: list[list[str]] = []
    evidence_rows: list[dict[str, object]] = []
    fp4_text_rows: list[list[str]] = []
    nonzero_fp4_rows: list[dict[str, object]] = []
    nonzero_tensor_rows: list[dict[str, object]] = []
    named_metric_rows = 0

    with args.csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows_seen += 1
            joined = " ".join(cell.strip() for cell in row if cell.strip())
            if not joined:
                continue

            if kernel_re.search(joined):
                kernel_rows.append(row)

            has_fp4_text = fp4_re.search(joined) is not None
            has_tensor_text = tensor_re.search(joined) is not None
            if has_fp4_text:
                fp4_text_rows.append(row)

            if not evidence_re.search(joined):
                continue

            named_metric_rows += 1
            nums = row_numbers(row)
            nonzero = [num for num in nums if abs(num) > 0.0]
            if not nonzero:
                continue

            evidence_row = {
                "row": row,
                "max_abs_value": max(abs(num) for num in nonzero),
            }
            evidence_rows.append(evidence_row)
            if has_fp4_text:
                nonzero_fp4_rows.append(evidence_row)
            if has_tensor_text:
                nonzero_tensor_rows.append(evidence_row)

    passed = len(nonzero_fp4_rows) > 0 or (len(fp4_text_rows) > 0 and len(nonzero_tensor_rows) > 0) or (args.allow_tensor_only and len(nonzero_tensor_rows) > 0)

    summary = {
        "csv": str(args.csv),
        "rows_seen": rows_seen,
        "kernel_regex": args.kernel_regex,
        "kernel_rows_seen": len(kernel_rows),
        "evidence_regex": args.evidence_regex,
        "fp4_regex": args.fp4_regex,
        "tensor_regex": args.tensor_regex,
        "named_metric_rows": named_metric_rows,
        "nonzero_evidence_rows": len(evidence_rows),
        "fp4_text_rows": len(fp4_text_rows),
        "nonzero_fp4_rows": len(nonzero_fp4_rows),
        "nonzero_tensor_rows": len(nonzero_tensor_rows),
        "allow_tensor_only": args.allow_tensor_only,
        "passed": passed,
        "evidence_rows": evidence_rows[: args.max_rows],
        "fp4_evidence_rows": nonzero_fp4_rows[: args.max_rows],
        "tensor_evidence_rows": nonzero_tensor_rows[: args.max_rows],
        "kernel_rows": kernel_rows[: min(args.max_rows, 20)],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"csv={args.csv}",
        f"rows_seen={rows_seen}",
        f"kernel_regex={args.kernel_regex}",
        f"kernel_rows_seen={len(kernel_rows)}",
        f"evidence_regex={args.evidence_regex}",
        f"fp4_regex={args.fp4_regex}",
        f"tensor_regex={args.tensor_regex}",
        f"named_metric_rows={named_metric_rows}",
        f"nonzero_evidence_rows={len(evidence_rows)}",
        f"fp4_text_rows={len(fp4_text_rows)}",
        f"nonzero_fp4_rows={len(nonzero_fp4_rows)}",
        f"nonzero_tensor_rows={len(nonzero_tensor_rows)}",
        f"allow_tensor_only={str(args.allow_tensor_only).lower()}",
        f"passed={str(summary['passed']).lower()}",
    ]
    if nonzero_fp4_rows:
        lines.append("")
        lines.append("Top FP4-specific evidence rows:")
        for item in nonzero_fp4_rows[: args.max_rows]:
            lines.append(f"- max_abs_value={item['max_abs_value']}: {item['row']}")
    elif fp4_text_rows and nonzero_tensor_rows:
        lines.append("")
        lines.append("FP4 text was present and generic tensor/MMA counters were nonzero.")
    elif evidence_rows:
        lines.append("")
        lines.append("Top generic evidence rows:")
        for item in evidence_rows[: args.max_rows]:
            lines.append(f"- max_abs_value={item['max_abs_value']}: {item['row']}")
    else:
        lines.append("")
        lines.append("No nonzero FP4/tensor/MMA evidence rows found.")

    args.text.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if summary["passed"] or args.allow_empty:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

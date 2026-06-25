#!/usr/bin/env python3
"""Local self-test for the KQ256 Spark handoff helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = SCRIPT_DIR / "analyze-ncu-fp4-evidence.py"
SUMMARIZER = SCRIPT_DIR / "summarize-kq256-gate.py"
VERIFIER = SCRIPT_DIR / "verify-kq256-artifacts.py"
BUNDLER = SCRIPT_DIR / "bundle-kq256-artifacts.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="Directory for self-test artifacts; kept after the run")
    return parser.parse_args()


def run(cmd: list[str], *, expect: int = 0) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != expect:
        raise RuntimeError(f"expected exit {expect}, got {completed.returncode}: {' '.join(cmd)}")
    return completed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(artifact_dir: Path) -> None:
    manifest = artifact_dir / "artifact-manifest.tsv"
    rows = [("sha256", "size_bytes", "path")]
    for path in sorted(item for item in artifact_dir.iterdir() if item.is_file() and item.name != manifest.name):
        rows.append((sha256_file(path), str(path.stat().st_size), path.name))
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def write_positive_raw_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ID,Kernel Name,Metric Name,Metric Unit,Metric Value",
                "1,kq256_operand_kernel,smsp__inst_executed_pipe_tensor_op_mxf4.sum,inst,128",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_tensor_only_raw_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ID,Kernel Name,Metric Name,Metric Unit,Metric Value",
                "1,kq256_operand_kernel,smsp__inst_executed_pipe_tensor_op_hmma.sum,inst,128",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_kq_log(path: Path, *, threads: int, kq_tops: float, useful: float, k_read: float) -> None:
    path.write_text(
        "\n".join(
            [
                "device:               0 NVIDIA GB10",
                "compute_capability:   sm_121 cc=1210",
                "spark_target:         yes",
                "blackwell_fp4_target: yes",
                "sms:                  20",
                f"config:               blocks=80 threads={threads} mtp_rows=4 useful_m16=25.0% iters=20000",
                "kq256 occupancy: active_blocks_per_sm=8 active_warps_per_sm=32 occupancy=66.7% shared=0.000 KiB",
                "q_quant: 240.000 GB/s-input  30.000 GB/s-output  270.000 GB/s-total  blocks=4194304 repeats=4 time=10.000 ms",
                f"kq256: {kq_tops:.3f} KQ-TOPS  {useful:.3f} useful-mtp-KQ-TOPS  {k_read:.3f} GB/s-K-compact-read  0.001000 GB-Q-compact-once  blocks=4194304 warps=80 iters=20000 time=1.000 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_complete_artifact(root: Path) -> Path:
    artifact_dir = root / "synthetic-kq256-complete"
    artifact_dir.mkdir(parents=True)

    write_kq_log(artifact_dir / "kq256-threads-128.log", threads=128, kq_tops=10.0, useful=2.5, k_read=700.0)
    write_kq_log(artifact_dir / "kq256-threads-256.log", threads=256, kq_tops=12.0, useful=3.0, k_read=840.0)

    raw_csv = artifact_dir / "ncu-kq256-threads-256-raw.csv"
    evidence_json = artifact_dir / "ncu-kq256-threads-256-fp4-evidence.json"
    evidence_txt = artifact_dir / "ncu-kq256-threads-256-fp4-evidence.txt"
    write_positive_raw_csv(raw_csv)
    run(
        [
            sys.executable,
            str(ANALYZER),
            "--csv",
            str(raw_csv),
            "--out",
            str(evidence_json),
            "--text",
            str(evidence_txt),
        ]
    )
    (artifact_dir / "ncu-kq256-threads-256.ncu-rep").write_text("synthetic ncu report\n", encoding="utf-8")

    (artifact_dir / "host-diagnostics.log").write_text(
        "\n".join(
            [
                "requested_arch=121a",
                "requested_device=0",
                "## nvidia-smi query",
                "0, NVIDIA GB10, 12.1, 580.00, 131072 MiB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "summary.txt").write_text(
        "\n".join(
            [
                "arch=121a",
                "device=0",
                f"host_diagnostics={artifact_dir / 'host-diagnostics.log'}",
                "exit_code=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run(
        [
            sys.executable,
            str(SUMMARIZER),
            "--dir",
            str(artifact_dir),
            "--csv",
            str(artifact_dir / "kq256-summary.csv"),
            "--json",
            str(artifact_dir / "kq256-summary.json"),
            "--text",
            str(artifact_dir / "kq256-summary.txt"),
            "--requested-device",
            "0",
            "--require-spark",
            "--require-ncu",
        ]
    )
    write_manifest(artifact_dir)
    return artifact_dir


def main() -> int:
    args = parse_args()
    temp: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        root = args.work_dir.resolve()
        if root.exists():
            raise SystemExit(f"work directory already exists: {root}")
        root.mkdir(parents=True)
    else:
        temp = tempfile.TemporaryDirectory(prefix="kq256-handoff-selftest-")
        root = Path(temp.name)

    try:
        print(f"selftest_root={root}")

        analyzer_dir = root / "analyzer"
        analyzer_dir.mkdir()
        positive_csv = analyzer_dir / "positive.csv"
        positive_json = analyzer_dir / "positive.json"
        positive_txt = analyzer_dir / "positive.txt"
        write_positive_raw_csv(positive_csv)
        run([sys.executable, str(ANALYZER), "--csv", str(positive_csv), "--out", str(positive_json), "--text", str(positive_txt)])

        tensor_csv = analyzer_dir / "tensor-only.csv"
        tensor_json = analyzer_dir / "tensor-only.json"
        tensor_txt = analyzer_dir / "tensor-only.txt"
        write_tensor_only_raw_csv(tensor_csv)
        run([sys.executable, str(ANALYZER), "--csv", str(tensor_csv), "--out", str(tensor_json), "--text", str(tensor_txt)], expect=2)

        artifact_dir = build_complete_artifact(root)
        strict_verify = [
            sys.executable,
            str(VERIFIER),
            "--dir",
            str(artifact_dir),
            "--require-manifest",
            "--require-go",
            "--require-ncu",
            "--require-ncu-threads",
            "256",
            "--require-host-diagnostics",
            "--require-host-arch",
            "121a",
            "--require-host-compute-cap",
            "12.1",
        ]
        run(strict_verify)

        missing_host = root / "synthetic-kq256-missing-hostdiag"
        shutil.copytree(artifact_dir, missing_host)
        (missing_host / "host-diagnostics.log").unlink()
        run(
            [
                sys.executable,
                str(VERIFIER),
                "--dir",
                str(missing_host),
                "--require-manifest",
                "--require-go",
                "--require-ncu",
                "--require-ncu-threads",
                "256",
                "--require-host-diagnostics",
                "--require-host-arch",
                "121a",
                "--require-host-compute-cap",
                "12.1",
            ],
            expect=2,
        )

        wrong_arch = root / "synthetic-kq256-wrong-arch"
        shutil.copytree(artifact_dir, wrong_arch)
        summary_lines = (wrong_arch / "summary.txt").read_text(encoding="utf-8").splitlines()
        (wrong_arch / "summary.txt").write_text(
            "\n".join("arch=120a" if line.startswith("arch=") else line for line in summary_lines) + "\n",
            encoding="utf-8",
        )
        write_manifest(wrong_arch)
        run(
            [
                sys.executable,
                str(VERIFIER),
                "--dir",
                str(wrong_arch),
                "--require-manifest",
                "--require-go",
                "--require-ncu",
                "--require-ncu-threads",
                "256",
                "--require-host-diagnostics",
                "--require-host-arch",
                "121a",
                "--require-host-compute-cap",
                "12.1",
            ],
            expect=2,
        )

        wrong_device = root / "synthetic-kq256-wrong-device"
        shutil.copytree(artifact_dir, wrong_device)
        summary_lines = (wrong_device / "summary.txt").read_text(encoding="utf-8").splitlines()
        (wrong_device / "summary.txt").write_text(
            "\n".join("device=1" if line.startswith("device=") else line for line in summary_lines) + "\n",
            encoding="utf-8",
        )
        write_manifest(wrong_device)
        run(
            [
                sys.executable,
                str(VERIFIER),
                "--dir",
                str(wrong_device),
                "--require-manifest",
                "--require-go",
                "--require-ncu",
                "--require-ncu-threads",
                "256",
                "--require-host-diagnostics",
                "--require-host-arch",
                "121a",
                "--require-host-compute-cap",
                "12.1",
            ],
            expect=2,
        )

        bad_manifest_path = root / "synthetic-kq256-bad-manifest-path"
        shutil.copytree(artifact_dir, bad_manifest_path)
        manifest_lines = (bad_manifest_path / "artifact-manifest.tsv").read_text(encoding="utf-8").splitlines()
        if len(manifest_lines) < 2:
            raise RuntimeError("synthetic manifest unexpectedly empty")
        cells = manifest_lines[1].split("\t")
        cells[-1] = "../escape.txt"
        manifest_lines[1] = "\t".join(cells)
        (bad_manifest_path / "artifact-manifest.tsv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        run(
            [
                sys.executable,
                str(VERIFIER),
                "--dir",
                str(bad_manifest_path),
                "--require-manifest",
                "--require-go",
                "--require-ncu",
                "--require-ncu-threads",
                "256",
                "--require-host-diagnostics",
                "--require-host-arch",
                "121a",
                "--require-host-compute-cap",
                "12.1",
            ],
            expect=2,
        )

        wrong_ncu_threads = strict_verify + ["--require-ncu-threads", "128"]
        run(wrong_ncu_threads, expect=2)

        bundle = root / "synthetic-kq256-complete.tar.gz"
        run([sys.executable, str(BUNDLER), "create", "--dir", str(artifact_dir), "--out", str(bundle), "--verify-first"])
        run(
            [
                sys.executable,
                str(BUNDLER),
                "verify",
                "--bundle",
                str(bundle),
                "--require-go",
                "--require-ncu",
                "--require-ncu-threads",
                "256",
                "--require-host-diagnostics",
                "--require-host-arch",
                "121a",
                "--require-host-compute-cap",
                "12.1",
            ]
        )

        bad_sha = root / "bad.sha256"
        bad_sha.write_text("0" * 64 + f"  {bundle.name}\n", encoding="utf-8")
        run([sys.executable, str(BUNDLER), "verify", "--bundle", str(bundle), "--sha256", str(bad_sha)], expect=2)

        bad_sha_name = root / "bad-name.sha256"
        bad_sha_name.write_text(f"{sha256_file(bundle)}  wrong-name.tar.gz\n", encoding="utf-8")
        run([sys.executable, str(BUNDLER), "verify", "--bundle", str(bundle), "--sha256", str(bad_sha_name)], expect=2)

        bad_sha_format = root / "bad-format.sha256"
        bad_sha_format.write_text(f"{sha256_file(bundle)}\nextra-line\n", encoding="utf-8")
        run([sys.executable, str(BUNDLER), "verify", "--bundle", str(bundle), "--sha256", str(bad_sha_format)], expect=2)

        unsafe_bundle = root / "unsafe.tar.gz"
        unsafe_payload = root / "unsafe-payload.txt"
        unsafe_payload.write_text("unsafe\n", encoding="utf-8")
        with tarfile.open(unsafe_bundle, "w:gz") as archive:
            archive.add(unsafe_payload, arcname="../escape.txt")
        unsafe_bundle.with_suffix(unsafe_bundle.suffix + ".sha256").write_text(
            f"{sha256_file(unsafe_bundle)}  {unsafe_bundle.name}\n",
            encoding="utf-8",
        )
        run([sys.executable, str(BUNDLER), "verify", "--bundle", str(unsafe_bundle)], expect=2)

        print("selftest_passed=true")
        return 0
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

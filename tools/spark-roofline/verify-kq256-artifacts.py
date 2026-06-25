#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


NCU_EVIDENCE_RE = re.compile(r"^ncu-kq256-threads-(\d+)-fp4-evidence\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify copied llama-spark-kq256 gate artifacts.")
    parser.add_argument("--dir", required=True, type=Path, help="Artifact directory from run-spark-kq256-gate.sh")
    parser.add_argument("--require-go", action="store_true", help="Require kq256-summary.json gate_decision=go")
    parser.add_argument("--require-ncu", action="store_true", help="Require at least one passing NCU evidence JSON")
    parser.add_argument("--require-ncu-threads", help="Require complete passing NCU evidence for these comma/space-separated thread counts")
    parser.add_argument("--require-manifest", action="store_true", help="Fail if artifact-manifest.tsv is missing")
    parser.add_argument("--require-host-diagnostics", action="store_true", help="Require host-diagnostics.log and summary.txt pointer")
    parser.add_argument("--require-host-arch", help="Require summary.txt/host-diagnostics.log to show this requested CUDA arch, e.g. 121a")
    parser.add_argument("--require-host-compute-cap", help="Require host-diagnostics.log to show this CUDA compute capability, e.g. 12.1")
    parser.add_argument("--require-build-arch", help="Require cmake-configure.log to show this CMAKE_CUDA_ARCHITECTURES value")
    parser.add_argument("--require-cuda-min", help="Require nvcc release at least this major.minor version, e.g. 12.8")
    parser.add_argument(
        "--reject-cuda-release",
        action="append",
        default=[],
        help="Reject this exact nvcc major.minor release; may be repeated, e.g. 13.1",
    )
    parser.add_argument("--text", type=Path, help="Optional verification report output path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_safe_relative_path(rel: str) -> bool:
    path = Path(rel)
    return bool(rel) and not path.is_absolute() and "\\" not in rel and ".." not in path.parts


def normalize_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.replace(",", " ").split() if item]


def parse_version_pair(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def version_lt(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[0] or (left[0] == right[0] and left[1] < right[1])


def version_text(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def verify_manifest(out_dir: Path, failures: list[str]) -> dict[str, object]:
    manifest = out_dir / "artifact-manifest.tsv"
    if not manifest.is_file():
        failures.append("missing artifact-manifest.tsv")
        return {"manifest_present": False, "manifest_rows": 0, "manifest_paths": set()}

    rows = 0
    manifest_paths: set[str] = set()
    with manifest.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sha256", "size_bytes", "path"}
        if not required.issubset(set(reader.fieldnames or [])):
            failures.append("artifact-manifest.tsv missing required columns")
            return {"manifest_present": True, "manifest_rows": 0, "manifest_paths": manifest_paths}

        for row in reader:
            rows += 1
            rel = row.get("path", "")
            if not is_safe_relative_path(rel):
                failures.append(f"manifest row has invalid relative path: {rel!r}")
                continue
            if rel in manifest_paths:
                failures.append(f"manifest has duplicate path entry: {rel}")
                continue
            manifest_paths.add(rel)

            path = out_dir / rel
            if not path.is_file():
                failures.append(f"manifest file missing: {rel}")
                continue

            expected_size = row.get("size_bytes", "")
            actual_size = path.stat().st_size
            if str(actual_size) != expected_size:
                failures.append(f"manifest size mismatch for {rel}: expected {expected_size}, got {actual_size}")

            expected_sha = row.get("sha256", "")
            if expected_sha and expected_sha != "sha256-unavailable":
                actual_sha = sha256_file(path)
                if actual_sha != expected_sha:
                    failures.append(f"manifest sha256 mismatch for {rel}")

    return {"manifest_present": True, "manifest_rows": rows, "manifest_paths": manifest_paths}


def require_manifest_path(manifest_info: dict[str, object], rel_path: str, failures: list[str]) -> None:
    paths = manifest_info.get("manifest_paths", set())
    if not isinstance(paths, set):
        failures.append("internal verifier error: manifest_paths is unavailable")
        return
    if rel_path not in paths:
        failures.append(f"required artifact is not listed in manifest: {rel_path}")


def load_summary(out_dir: Path, failures: list[str], required: bool) -> dict[str, object]:
    path = out_dir / "kq256-summary.json"
    if not path.is_file():
        if required:
            failures.append("missing kq256-summary.json")
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        failures.append(f"kq256-summary.json is invalid JSON: {exc}")
        return {}

    if not isinstance(data, dict):
        failures.append("kq256-summary.json root is not an object")
        return {}
    return data


def load_runner_summary(out_dir: Path, failures: list[str], required: bool) -> dict[str, str]:
    path = out_dir / "summary.txt"
    if not path.is_file():
        if required:
            failures.append("missing summary.txt")
        return {}

    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def inspect_ncu(out_dir: Path, failures: list[str], require_complete: bool, manifest_info: dict[str, object], require_manifest: bool) -> dict[str, object]:
    evidence_files = sorted(out_dir.glob("ncu-kq256-threads-*-fp4-evidence.json"))
    passed = 0
    complete_passed = 0
    complete_sets = 0
    nonzero_fp4_rows = 0
    complete_passed_threads: set[str] = set()
    for path in evidence_files:
        match = NCU_EVIDENCE_RE.match(path.name)
        if not match:
            continue
        threads = match.group(1)

        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            if require_complete:
                failures.append(f"invalid NCU evidence JSON: {path.name}")
            continue

        evidence_passed = data.get("passed") is True
        if evidence_passed:
            passed += 1
        value = data.get("nonzero_fp4_rows", 0)
        if isinstance(value, int):
            nonzero_fp4_rows += value

        expected_rel = [
            f"ncu-kq256-threads-{threads}-raw.csv",
            f"ncu-kq256-threads-{threads}-fp4-evidence.txt",
        ]
        expected = [out_dir / rel for rel in expected_rel]
        report_candidates = [
            out_dir / f"ncu-kq256-threads-{threads}.ncu-rep",
            out_dir / f"ncu-kq256-threads-{threads}",
        ]
        report_rel_candidates = [
            f"ncu-kq256-threads-{threads}.ncu-rep",
            f"ncu-kq256-threads-{threads}",
        ]
        missing = [item.name for item in expected if not item.is_file()]
        if not any(item.is_file() for item in report_candidates):
            missing.append(f"ncu-kq256-threads-{threads}.ncu-rep")

        if missing:
            if require_complete:
                failures.append(f"threads={threads}: incomplete NCU artifact set, missing {', '.join(missing)}")
        else:
            if require_complete and require_manifest:
                require_manifest_path(manifest_info, path.name, failures)
                for rel in expected_rel:
                    require_manifest_path(manifest_info, rel, failures)
                report_rel = next((rel for rel in report_rel_candidates if (out_dir / rel).is_file()), report_rel_candidates[0])
                require_manifest_path(manifest_info, report_rel, failures)
            complete_sets += 1
            if evidence_passed:
                complete_passed += 1
                complete_passed_threads.add(threads)

    return {
        "ncu_evidence_files": len(evidence_files),
        "ncu_passed_files": passed,
        "ncu_complete_sets": complete_sets,
        "ncu_complete_passed_sets": complete_passed,
        "ncu_complete_passed_threads": sorted(complete_passed_threads, key=int),
        "ncu_nonzero_fp4_rows": nonzero_fp4_rows,
    }


def inspect_build_logs(
    out_dir: Path,
    failures: list[str],
    manifest_info: dict[str, object],
    require_manifest: bool,
    required_build_arch: str | None,
) -> dict[str, object]:
    configure_log = out_dir / "cmake-configure.log"
    build_log = out_dir / "cmake-build.log"
    configure_present = configure_log.is_file()
    build_present = build_log.is_file()
    matched_arch = ""

    if required_build_arch:
        if not configure_present:
            failures.append("missing cmake-configure.log")
        else:
            text = configure_log.read_text(encoding="utf-8", errors="replace")
            allowed_needles = [
                f"CMAKE_CUDA_ARCHITECTURES={required_build_arch}",
                f"CMAKE_CUDA_ARCHITECTURES:STRING={required_build_arch}",
            ]
            if any(needle in text for needle in allowed_needles):
                matched_arch = required_build_arch
            else:
                failures.append(
                    f"cmake-configure.log does not show CMAKE_CUDA_ARCHITECTURES={required_build_arch}"
                )
        if not build_present:
            failures.append("missing cmake-build.log")
        if require_manifest:
            require_manifest_path(manifest_info, "cmake-configure.log", failures)
            require_manifest_path(manifest_info, "cmake-build.log", failures)

    return {
        "build_configure_log_present": configure_present,
        "build_log_present": build_present,
        "build_required_arch": required_build_arch or "",
        "build_required_arch_matched": matched_arch,
    }


def inspect_host_diagnostics(
    out_dir: Path,
    runner_summary: dict[str, str],
    failures: list[str],
    required: bool,
    manifest_info: dict[str, object],
    require_manifest: bool,
    required_arch: str | None,
    required_compute_cap: str | None,
    required_cuda_min: str | None,
    rejected_cuda_releases: list[str],
) -> dict[str, object]:
    path = out_dir / "host-diagnostics.log"
    present = path.is_file()
    summary_value = runner_summary.get("host_diagnostics", "")
    compute_caps: list[dict[str, str]] = []
    cuda_releases: set[str] = set()
    runner_requested_arch = runner_summary.get("arch", "")
    diag_requested_arch = ""

    if (required or required_compute_cap) and not present:
        failures.append("missing host-diagnostics.log")
    if required and not summary_value:
        failures.append("summary.txt missing host_diagnostics entry")
    if summary_value:
        summary_path = Path(summary_value)
        if summary_path.name != "host-diagnostics.log":
            failures.append(f"summary.txt host_diagnostics does not point to host-diagnostics.log: {summary_value!r}")
    if required and require_manifest:
        require_manifest_path(manifest_info, "host-diagnostics.log", failures)

    requested_device = runner_summary.get("device", "")
    if present:
        diag_requested_device = ""
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            for match in re.finditer(r"\brelease\s+(\d+\.\d+)\b", line):
                cuda_releases.add(match.group(1))
            if line.startswith("requested_arch="):
                diag_requested_arch = line.split("=", 1)[1].strip()
                continue
            if line.startswith("requested_device="):
                diag_requested_device = line.split("=", 1)[1].strip()
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5 and parts[0].isdigit() and re.fullmatch(r"\d+(?:\.\d+)?", parts[2]):
                compute_caps.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "compute_cap": parts[2],
                    }
                )
        if requested_device and diag_requested_device and requested_device != diag_requested_device:
            failures.append(
                f"summary.txt device={requested_device!r} does not match host-diagnostics requested_device={diag_requested_device!r}"
            )
        if not requested_device:
            requested_device = diag_requested_device

    if runner_summary.get("nvcc_release"):
        cuda_releases.add(runner_summary["nvcc_release"])

    parsed_cuda_releases: dict[str, tuple[int, int]] = {}
    for release in sorted(cuda_releases):
        parsed = parse_version_pair(release)
        if parsed is None:
            failures.append(f"could not parse CUDA release from host diagnostics: {release!r}")
            continue
        parsed_cuda_releases[release] = parsed

    if required_cuda_min:
        required_min = parse_version_pair(required_cuda_min)
        if required_min is None:
            failures.append(f"invalid --require-cuda-min value: {required_cuda_min!r}")
        elif not parsed_cuda_releases:
            failures.append(f"no nvcc release found; required CUDA >= {required_cuda_min}")
        elif all(version_lt(version, required_min) for version in parsed_cuda_releases.values()):
            found = ", ".join(sorted(parsed_cuda_releases)) or "none"
            failures.append(f"nvcc release does not satisfy CUDA >= {required_cuda_min}: found {found}")

    normalized_rejects: set[str] = set()
    for release in rejected_cuda_releases:
        parsed = parse_version_pair(release)
        if parsed is None:
            failures.append(f"invalid --reject-cuda-release value: {release!r}")
            continue
        normalized_rejects.add(version_text(parsed))
    rejected_found = normalized_rejects.intersection(parsed_cuda_releases)
    if rejected_found:
        failures.append(f"rejected CUDA release present: {', '.join(sorted(rejected_found))}")

    if required_arch:
        if runner_requested_arch and diag_requested_arch and runner_requested_arch != diag_requested_arch:
            failures.append(
                f"summary.txt arch={runner_requested_arch!r} does not match host-diagnostics requested_arch={diag_requested_arch!r}"
            )
        actual_arch = runner_requested_arch or diag_requested_arch
        if actual_arch != required_arch:
            found = actual_arch or "none"
            failures.append(f"requested arch is not {required_arch}: found {found}")

    matched_compute_cap = ""
    if required_compute_cap:
        for row in compute_caps:
            if row["compute_cap"] != required_compute_cap:
                continue
            if requested_device and row["index"] != requested_device:
                continue
            matched_compute_cap = row["compute_cap"]
            break
        if not matched_compute_cap:
            device_text = f" for requested device {requested_device}" if requested_device else ""
            found = ", ".join(f"{row['index']}:{row['compute_cap']}:{row['name']}" for row in compute_caps) or "none"
            failures.append(
                f"host-diagnostics.log does not show compute_cap={required_compute_cap}{device_text}; found {found}"
            )

    return {
        "host_diagnostics_present": present,
        "host_diagnostics_summary_entry": summary_value,
        "host_requested_arch": runner_requested_arch or diag_requested_arch,
        "host_required_arch": required_arch or "",
        "host_compute_caps": compute_caps,
        "host_required_compute_cap": required_compute_cap or "",
        "host_required_compute_cap_matched": matched_compute_cap,
        "cuda_releases": sorted(parsed_cuda_releases),
        "cuda_required_min": required_cuda_min or "",
        "cuda_rejected_releases": sorted(normalized_rejects),
    }


def main() -> int:
    args = parse_args()
    if not args.dir.is_dir():
        print(f"missing artifact directory: {args.dir}", file=sys.stderr)
        return 2

    failures: list[str] = []
    manifest_info: dict[str, object]
    if args.require_manifest or (args.dir / "artifact-manifest.tsv").exists():
        manifest_info = verify_manifest(args.dir, failures)
    else:
        manifest_info = {"manifest_present": False, "manifest_rows": 0}

    runner_summary = load_runner_summary(args.dir, failures, required=args.require_go)
    summary = load_summary(args.dir, failures, required=args.require_go)
    if args.require_manifest and summary:
        require_manifest_path(manifest_info, "summary.txt", failures)
        require_manifest_path(manifest_info, "kq256-summary.json", failures)
        if (args.dir / "kq256-summary.txt").is_file():
            require_manifest_path(manifest_info, "kq256-summary.txt", failures)
        if (args.dir / "kq256-summary.csv").is_file():
            require_manifest_path(manifest_info, "kq256-summary.csv", failures)

    ncu_info = inspect_ncu(args.dir, failures, require_complete=args.require_ncu, manifest_info=manifest_info, require_manifest=args.require_manifest)
    build_info = inspect_build_logs(
        args.dir,
        failures,
        manifest_info=manifest_info,
        require_manifest=args.require_manifest,
        required_build_arch=args.require_build_arch,
    )
    host_diag_info = inspect_host_diagnostics(
        args.dir,
        runner_summary,
        failures,
        required=args.require_host_diagnostics,
        manifest_info=manifest_info,
        require_manifest=args.require_manifest,
        required_arch=args.require_host_arch,
        required_compute_cap=args.require_host_compute_cap,
        required_cuda_min=args.require_cuda_min,
        rejected_cuda_releases=args.reject_cuda_release,
    )

    gate_decision = summary.get("gate_decision", "")
    if args.require_go and gate_decision != "go":
        failures.append(f"gate_decision is not go: {gate_decision!r}")
    runner_exit_code = runner_summary.get("exit_code", "")
    if args.require_go and runner_exit_code != "0":
        failures.append(f"runner exit_code is not 0: {runner_exit_code!r}")

    if args.require_ncu and ncu_info["ncu_passed_files"] < 1:
        failures.append("no passing NCU FP4 evidence JSON found")
    if args.require_ncu and ncu_info["ncu_complete_passed_sets"] < 1:
        failures.append("no complete passing NCU artifact set found")
    required_ncu_threads = normalize_list(args.require_ncu_threads)
    if required_ncu_threads:
        complete_passed_threads = set(ncu_info["ncu_complete_passed_threads"])
        for threads in required_ncu_threads:
            if threads not in complete_passed_threads:
                failures.append(f"missing complete passing NCU artifact set for required threads={threads}")

        summary_threads = {}
        rows = summary.get("rows", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                threads = row.get("threads")
                if threads is not None:
                    summary_threads[str(threads)] = row
        for threads in required_ncu_threads:
            row = summary_threads.get(threads)
            if row is None:
                failures.append(f"kq256-summary.json has no row for required NCU threads={threads}")
            elif row.get("ncu_passed") is not True:
                failures.append(f"kq256-summary.json row for required NCU threads={threads} is not ncu_passed=true")

    passed = len(failures) == 0
    lines = [
        f"artifact_dir={args.dir}",
        f"passed={str(passed).lower()}",
        f"gate_decision={gate_decision}",
        f"runner_exit_code={runner_exit_code}",
        f"manifest_present={str(manifest_info['manifest_present']).lower()}",
        f"manifest_rows={manifest_info['manifest_rows']}",
        f"build_configure_log_present={str(build_info['build_configure_log_present']).lower()}",
        f"build_log_present={str(build_info['build_log_present']).lower()}",
        f"build_required_arch={build_info['build_required_arch']}",
        f"build_required_arch_matched={build_info['build_required_arch_matched']}",
        f"host_diagnostics_present={str(host_diag_info['host_diagnostics_present']).lower()}",
        f"host_diagnostics_summary_entry={host_diag_info['host_diagnostics_summary_entry']}",
        f"host_requested_arch={host_diag_info['host_requested_arch']}",
        f"host_required_arch={host_diag_info['host_required_arch']}",
        f"host_required_compute_cap={host_diag_info['host_required_compute_cap']}",
        f"host_required_compute_cap_matched={host_diag_info['host_required_compute_cap_matched']}",
        f"host_compute_caps={json.dumps(host_diag_info['host_compute_caps'], sort_keys=True)}",
        f"cuda_releases={' '.join(host_diag_info['cuda_releases'])}",
        f"cuda_required_min={host_diag_info['cuda_required_min']}",
        f"cuda_rejected_releases={' '.join(host_diag_info['cuda_rejected_releases'])}",
        f"ncu_evidence_files={ncu_info['ncu_evidence_files']}",
        f"ncu_passed_files={ncu_info['ncu_passed_files']}",
        f"ncu_complete_sets={ncu_info['ncu_complete_sets']}",
        f"ncu_complete_passed_sets={ncu_info['ncu_complete_passed_sets']}",
        f"ncu_required_threads={' '.join(required_ncu_threads)}",
        f"ncu_complete_passed_threads={' '.join(ncu_info['ncu_complete_passed_threads'])}",
        f"ncu_nonzero_fp4_rows={ncu_info['ncu_nonzero_fp4_rows']}",
    ]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for failure in failures:
            lines.append(f"- {failure}")

    report = "\n".join(lines) + "\n"
    if args.text:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(report, encoding="utf-8")
    print(report, end="")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

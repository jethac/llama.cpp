#!/usr/bin/env python3
"""Bundle and verify llama-spark-kq256 gate artifacts.

This is a portability helper for copying Spark results off-host. It creates a
tar.gz archive with relative paths only, writes a SHA-256 sidecar, and can
extract the archive into a temporary directory before running the existing
copied-artifact verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    create = subparsers.add_parser("create", help="Create a tar.gz bundle from an artifact directory")
    create.add_argument("--dir", required=True, type=Path, help="Artifact directory to bundle")
    create.add_argument("--out", required=True, type=Path, help="Output .tar.gz path")
    create.add_argument("--verify-first", action="store_true", help="Run manifest-only verification before bundling")

    verify = subparsers.add_parser("verify", help="Verify a tar.gz bundle and its extracted artifact directory")
    verify.add_argument("--bundle", required=True, type=Path, help="Bundle .tar.gz path")
    verify.add_argument("--sha256", type=Path, help="SHA-256 sidecar path; defaults to <bundle>.sha256")
    verify.add_argument("--extract-dir", type=Path, help="Optional extraction directory to keep")
    verify.add_argument("--require-go", action="store_true", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-ncu", action="store_true", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-ncu-threads", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-host-diagnostics", action="store_true", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-host-arch", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-host-compute-cap", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-build-arch", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--require-cuda-min", help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--reject-cuda-release", action="append", default=[], help="Pass through to verify-kq256-artifacts.py")
    verify.add_argument("--keep-temp", action="store_true", help="Keep temporary extraction directory")

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


SHA256_SIDECAR_RE = re.compile(r"^([0-9a-fA-F]{64})[ \t]+([^ \t\r\n]+)$")


def read_sha256_sidecar(sha_path: Path, bundle_name: str) -> str:
    lines = [line.strip() for line in sha_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != 1:
        raise RuntimeError(f"SHA-256 sidecar must contain exactly one non-empty line: {sha_path}")
    match = SHA256_SIDECAR_RE.fullmatch(lines[0])
    if not match:
        raise RuntimeError(f"SHA-256 sidecar has invalid format: {sha_path}")
    digest, sidecar_name = match.groups()
    if Path(sidecar_name).name != sidecar_name or sidecar_name != bundle_name:
        raise RuntimeError(
            f"SHA-256 sidecar filename {sidecar_name!r} does not match bundle name {bundle_name!r}"
        )
    return digest.lower()


def verifier_path() -> Path:
    return Path(__file__).resolve().with_name("verify-kq256-artifacts.py")


def run_verifier(
    artifact_dir: Path,
    *,
    require_go: bool = False,
    require_ncu: bool = False,
    require_ncu_threads: str | None = None,
    require_host_diagnostics: bool = False,
    require_host_arch: str | None = None,
    require_host_compute_cap: str | None = None,
    require_build_arch: str | None = None,
    require_cuda_min: str | None = None,
    reject_cuda_releases: list[str] | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(verifier_path()),
        "--dir",
        str(artifact_dir),
        "--require-manifest",
    ]
    if require_go:
        cmd.append("--require-go")
    if require_ncu:
        cmd.append("--require-ncu")
    if require_ncu_threads:
        cmd.extend(["--require-ncu-threads", require_ncu_threads])
    if require_host_diagnostics:
        cmd.append("--require-host-diagnostics")
    if require_host_arch:
        cmd.extend(["--require-host-arch", require_host_arch])
    if require_host_compute_cap:
        cmd.extend(["--require-host-compute-cap", require_host_compute_cap])
    if require_build_arch:
        cmd.extend(["--require-build-arch", require_build_arch])
    if require_cuda_min:
        cmd.extend(["--require-cuda-min", require_cuda_min])
    for release in reject_cuda_releases or []:
        cmd.extend(["--reject-cuda-release", release])

    subprocess.run(cmd, check=True)


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def create_bundle(artifact_dir: Path, out_path: Path, verify_first: bool) -> int:
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        print(f"missing artifact directory: {artifact_dir}", file=sys.stderr)
        return 2
    if verify_first:
        run_verifier(artifact_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"refusing to overwrite existing bundle: {out_path}", file=sys.stderr)
        return 2

    root_name = artifact_dir.name
    with tarfile.open(out_path, "w:gz") as archive:
        for file_path in iter_files(artifact_dir):
            rel = file_path.relative_to(artifact_dir)
            archive.add(file_path, arcname=str(Path(root_name) / rel), recursive=False)

    digest = sha256_file(out_path)
    sidecar = out_path.with_suffix(out_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")

    print(f"bundle={out_path}")
    print(f"sha256={digest}")
    print(f"sha256_sidecar={sidecar}")
    print(f"files={len(iter_files(artifact_dir))}")
    return 0


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    top_level_dirs: set[str] = set()
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"unsafe archive member path: {member.name!r}")
        if len(member_path.parts) < 2:
            raise RuntimeError(f"archive member is not under a top-level artifact directory: {member.name!r}")
        top_level_dirs.add(member_path.parts[0])
        if member.issym() or member.islnk():
            raise RuntimeError(f"archive links are not allowed: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"archive special entries are not allowed: {member.name!r}")
    if len(top_level_dirs) != 1:
        raise RuntimeError(f"expected exactly one top-level artifact directory, found {len(top_level_dirs)}")
    return members


def extract_bundle(bundle: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    before = set(extract_dir.iterdir())
    with tarfile.open(bundle, "r:gz") as archive:
        members = safe_members(archive)
        archive.extractall(extract_dir, members=members)
    after = set(extract_dir.iterdir())
    created = sorted(after - before)
    dirs = [path for path in created if path.is_dir()]
    if len(dirs) != 1:
        raise RuntimeError(f"expected exactly one top-level artifact directory, found {len(dirs)}")
    return dirs[0]


def verify_bundle(
    bundle: Path,
    sha_path: Path,
    extract_dir: Path | None,
    require_go: bool,
    require_ncu: bool,
    require_ncu_threads: str | None,
    require_host_diagnostics: bool,
    require_host_arch: str | None,
    require_host_compute_cap: str | None,
    require_build_arch: str | None,
    require_cuda_min: str | None,
    reject_cuda_releases: list[str],
    keep_temp: bool,
) -> int:
    if not bundle.is_file():
        print(f"missing bundle: {bundle}", file=sys.stderr)
        return 2
    if not sha_path.is_file():
        print(f"missing SHA-256 sidecar: {sha_path}", file=sys.stderr)
        return 2

    try:
        expected = read_sha256_sidecar(sha_path, bundle.name)
    except RuntimeError as exc:
        print(f"bundle verification failed: {exc}", file=sys.stderr)
        return 2
    actual = sha256_file(bundle)
    if actual != expected:
        print(f"bundle sha256 mismatch: expected {expected}, got {actual}", file=sys.stderr)
        return 2

    if extract_dir is None:
        if keep_temp:
            extract_root = Path(tempfile.mkdtemp(prefix="kq256-bundle-"))
            cleanup_temp = False
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="kq256-bundle-")
            extract_root = Path(temp_dir.name)
            cleanup_temp = True
    else:
        extract_root = extract_dir
        temp_dir = None
        cleanup_temp = False
        if extract_root.exists() and any(extract_root.iterdir()):
            print(f"refusing to extract into non-empty directory: {extract_root}", file=sys.stderr)
            return 2

    try:
        try:
            artifact_dir = extract_bundle(bundle, extract_root)
            run_verifier(
                artifact_dir,
                require_go=require_go,
                require_ncu=require_ncu,
                require_ncu_threads=require_ncu_threads,
                require_host_diagnostics=require_host_diagnostics,
                require_host_arch=require_host_arch,
                require_host_compute_cap=require_host_compute_cap,
                require_build_arch=require_build_arch,
                require_cuda_min=require_cuda_min,
                reject_cuda_releases=reject_cuda_releases,
            )
        except (RuntimeError, tarfile.TarError, subprocess.CalledProcessError) as exc:
            print(f"bundle verification failed: {exc}", file=sys.stderr)
            return 2
        print(f"bundle={bundle}")
        print(f"sha256={actual}")
        print(f"extracted_artifacts={artifact_dir}")
        print("bundle_verified=true")
    finally:
        if extract_dir is None and keep_temp:
            print(f"kept_extract_dir={extract_root}")
        if cleanup_temp:
            temp_dir.cleanup()

    return 0


def main() -> int:
    args = parse_args()
    if args.cmd == "create":
        return create_bundle(args.dir, args.out, args.verify_first)
    if args.cmd == "verify":
        return verify_bundle(
            args.bundle,
            args.sha256 or args.bundle.with_suffix(args.bundle.suffix + ".sha256"),
            args.extract_dir,
            args.require_go,
            args.require_ncu,
            args.require_ncu_threads,
            args.require_host_diagnostics,
            args.require_host_arch,
            args.require_host_compute_cap,
            args.require_build_arch,
            args.require_cuda_min,
            args.reject_cuda_release,
            args.keep_temp,
        )
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

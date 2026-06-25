#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
usage: run-spark-kq256-handoff.sh [options]

Run the full D=256 KQ Spark handoff sequence:

  1. local synthetic tooling self-test
  2. Spark preflight
  3. full Spark KQ256 gate
  4. portable bundle creation
  5. strict bundle verification

This wrapper is for shipping/Spark evidence. It does not expose --allow-proxy.

Options:
  --build-dir DIR       CMake build directory (default: build-spark-sm121-kq256)
  --out-dir DIR         Full gate artifact directory (default: /tmp/llamacpp-spark-kq256)
  --preflight-dir DIR   Preflight artifact directory (default: <out-dir>-preflight)
  --bundle PATH         Output bundle path (default: <out-dir>.tar.gz)
  --summary PATH        Handoff summary path (default: <out-dir>-handoff-summary.txt)
  --device N            CUDA device id (default: 0)
  --threads LIST        Space/comma-separated thread counts (default: "128 256 512")
  --mtp-rows N          Useful MTP verification rows in m16 tile (default: 4)
  --iters N             Iterations for normal runs (default: 20000)
  --ncu-iters N         Iterations for Nsight runs (default: 2000)
  --ncu-threads LIST    Thread counts for Nsight runs (default: "512")
  --rebuild-full        Rebuild during the full gate instead of reusing preflight build
  --skip-self-test      Skip synthetic local tooling self-test
  --dry-run             Print commands without executing
  --help                Print this help

Environment overrides are passed through to run-spark-kq256-gate.sh:
  CMAKE, NCU, NCU_SET, NCU_METRICS, NCU_IMPORT_METRICS, NCU_KERNEL_REGEX, PYTHON
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-python3}"
build_dir="build-spark-sm121-kq256"
out_dir="/tmp/llamacpp-spark-kq256"
preflight_dir=""
bundle_path=""
summary_path=""
device="0"
threads="128 256 512"
mtp_rows="4"
iters="20000"
ncu_iters="2000"
ncu_threads="512"
rebuild_full=0
skip_self_test=0
dry_run=0
handoff_stage="init"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) build_dir="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --preflight-dir) preflight_dir="$2"; shift 2 ;;
        --bundle) bundle_path="$2"; shift 2 ;;
        --summary) summary_path="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --mtp-rows) mtp_rows="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --ncu-iters) ncu_iters="$2"; shift 2 ;;
        --ncu-threads) ncu_threads="$2"; shift 2 ;;
        --rebuild-full) rebuild_full=1; shift ;;
        --skip-self-test) skip_self_test=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$preflight_dir" ]]; then
    preflight_dir="${out_dir%/}-preflight"
fi
if [[ -z "$bundle_path" ]]; then
    bundle_path="${out_dir%/}.tar.gz"
fi
if [[ -z "$summary_path" ]]; then
    summary_path="${out_dir%/}-handoff-summary.txt"
fi

normalize_path() {
    local path="$1"
    path="${path//\\//}"
    path="${path//$'\xef\x80\xba'/:}"
    if [[ "$path" =~ ^([A-Za-z]):/(.*)$ ]]; then
        local drive="${BASH_REMATCH[1],,}"
        printf '/mnt/%s/%s\n' "$drive" "${BASH_REMATCH[2]}"
    else
        printf '%s\n' "$path"
    fi
}

out_dir="$(normalize_path "$out_dir")"
preflight_dir="$(normalize_path "$preflight_dir")"
bundle_path="$(normalize_path "$bundle_path")"
summary_path="$(normalize_path "$summary_path")"

quote_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

run_cmd() {
    echo "+ $(quote_cmd "$@")"
    if [[ "$dry_run" -eq 0 ]]; then
        "$@"
    fi
}

gate_script="$repo_root/tools/spark-roofline/run-spark-kq256-gate.sh"
self_test="$repo_root/tools/spark-roofline/test-kq256-handoff-tools.py"
bundler="$repo_root/tools/spark-roofline/bundle-kq256-artifacts.py"

write_handoff_summary() {
    local exit_code="$1"
    local handoff_status
    local handoff_complete

    if [[ "$dry_run" -ne 0 ]]; then
        handoff_status="planned"
        handoff_complete="false"
    elif [[ "$exit_code" -eq 0 ]]; then
        handoff_status="complete"
        handoff_complete="true"
    else
        handoff_status="failed"
        handoff_complete="false"
    fi

    local verify_cmd=(
        "$python_bin" "$bundler" verify
        --bundle "$bundle_path"
        --require-go
        --require-ncu
        --require-ncu-threads "$ncu_threads"
        --require-host-diagnostics
        --require-host-arch 121a
        --require-host-compute-cap 12.1
        --require-build-arch 121a
        --require-cuda-min 12.8
        --reject-cuda-release 13.1
    )
    local verify_copied_cmd=(
        "$python_bin" "$bundler" verify
        --bundle "<copied-bundle-path>"
        --require-go
        --require-ncu
        --require-ncu-threads "$ncu_threads"
        --require-host-diagnostics
        --require-host-arch 121a
        --require-host-compute-cap 12.1
        --require-build-arch 121a
        --require-cuda-min 12.8
        --reject-cuda-release 13.1
    )

    mkdir -p "$(dirname "$summary_path")"
    {
        echo "handoff_status=$handoff_status"
        echo "handoff_complete=$handoff_complete"
        echo "exit_code=$exit_code"
        echo "last_stage=$handoff_stage"
        echo "dry_run=$dry_run"
        echo "repo_root=$repo_root"
        echo "build_dir=$build_dir"
        echo "preflight_dir=$preflight_dir"
        echo "out_dir=$out_dir"
        echo "bundle=$bundle_path"
        echo "bundle_sha256=${bundle_path}.sha256"
        echo "summary=$summary_path"
        echo "device=$device"
        echo "threads=$threads"
        echo "mtp_rows=$mtp_rows"
        echo "iters=$iters"
        echo "ncu_threads=$ncu_threads"
        echo "ncu_iters=$ncu_iters"
        echo "rebuild_full=$rebuild_full"
        echo "verify_command=$(quote_cmd "${verify_cmd[@]}")"
        echo "verify_copied_bundle_command=$(quote_cmd "${verify_copied_cmd[@]}")"
        echo "copy_bundle=$bundle_path"
        echo "copy_sha256=${bundle_path}.sha256"
        echo "copy_summary=$summary_path"
    } > "$summary_path"
}

finish_handoff() {
    local exit_code="$?"
    write_handoff_summary "$exit_code"
    exit "$exit_code"
}

trap finish_handoff EXIT

echo "repo_root=$repo_root"
echo "build_dir=$build_dir"
echo "preflight_dir=$preflight_dir"
echo "out_dir=$out_dir"
echo "bundle=$bundle_path"
echo "summary=$summary_path"
echo "device=$device"
echo "threads=$threads"
echo "mtp_rows=$mtp_rows"
echo "iters=$iters"
echo "ncu_iters=$ncu_iters"
echo "ncu_threads=$ncu_threads"
echo "rebuild_full=$rebuild_full"
echo "dry_run=$dry_run"

if [[ "$skip_self_test" -eq 0 ]]; then
    handoff_stage="self_test"
    run_cmd "$python_bin" "$self_test"
fi

handoff_stage="preflight"
preflight_cmd=(
    bash "$gate_script"
    --preflight-only
    --device "$device"
    --build-dir "$build_dir"
    --out-dir "$preflight_dir"
)
if [[ "$dry_run" -ne 0 ]]; then
    preflight_cmd+=(--dry-run)
fi
run_cmd "${preflight_cmd[@]}"

handoff_stage="full_gate"
full_cmd=(
    bash "$gate_script"
    --device "$device"
    --build-dir "$build_dir"
    --out-dir "$out_dir"
    --threads "$threads"
    --mtp-rows "$mtp_rows"
    --iters "$iters"
    --ncu-threads "$ncu_threads"
    --ncu-iters "$ncu_iters"
)
if [[ "$rebuild_full" -eq 0 ]]; then
    full_cmd+=(--no-build)
fi
if [[ "$dry_run" -ne 0 ]]; then
    full_cmd+=(--dry-run)
fi
run_cmd "${full_cmd[@]}"

if [[ "$dry_run" -eq 0 ]]; then
    mkdir -p "$out_dir"
    for build_log in cmake-configure.log cmake-build.log; do
        if [[ ! -f "$out_dir/$build_log" && -f "$preflight_dir/$build_log" ]]; then
            cp "$preflight_dir/$build_log" "$out_dir/$build_log"
        fi
    done
fi

handoff_stage="bundle_create"
run_cmd "$python_bin" "$bundler" create \
    --dir "$out_dir" \
    --out "$bundle_path" \
    --verify-first

handoff_stage="bundle_verify"
run_cmd "$python_bin" "$bundler" verify \
    --bundle "$bundle_path" \
    --require-go \
    --require-ncu \
    --require-ncu-threads "$ncu_threads" \
    --require-host-diagnostics \
    --require-host-arch 121a \
    --require-host-compute-cap 12.1 \
    --require-build-arch 121a \
    --require-cuda-min 12.8 \
    --reject-cuda-release 13.1

handoff_stage="done"
write_handoff_summary 0
trap - EXIT

if [[ "$dry_run" -ne 0 ]]; then
    handoff_status="planned"
    handoff_complete="false"
else
    handoff_status="complete"
    handoff_complete="true"
fi

echo "handoff_status=$handoff_status"
echo "handoff_complete=$handoff_complete"
echo "copy_bundle=$bundle_path"
echo "copy_sha256=${bundle_path}.sha256"
echo "copy_summary=$summary_path"
echo "handoff_summary=$summary_path"

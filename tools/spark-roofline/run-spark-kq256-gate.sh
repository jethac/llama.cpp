#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
usage: run-spark-kq256-gate.sh [options]

Build and run the D=256 NVFP4 KQ/MTP Spark gate.

Options:
  --build-dir DIR       CMake build directory (default: build-spark-sm121-kq256)
  --out-dir DIR         Artifact directory (default: stage timestamp under /tmp)
  --arch ARCH           CUDA architecture (default: 121a)
  --device N            CUDA device id passed to llama-spark-kq256 (default: 0)
  --threads LIST        Space/comma-separated thread counts (default: "128 256 512")
  --mtp-rows N          Useful MTP verification rows in m16 tile (default: 4)
  --iters N             Iterations for normal runs (default: 20000)
  --ncu-iters N         Iterations for Nsight runs (default: 2000)
  --ncu-threads LIST    Thread counts for Nsight runs (default: "512")
  --allow-proxy         Allow non-sm_121 devices for local proxy runs
  --preflight-only      Build/find executable, run one tiny device smoke, then exit
  --no-build            Skip CMake configure/build
  --no-ncu              Skip Nsight Compute capture
  --dry-run             Print commands without executing
  --help                Print this help

Environment overrides:
  CMAKE                 CMake executable (default: cmake)
  NCU                   Nsight Compute CLI executable (default: ncu)
  NCU_SET               Nsight Compute section set (default: full)
  NCU_METRICS           Optional comma-separated explicit metrics
  NCU_IMPORT_METRICS    Optional comma-separated metrics for raw CSV import
  NCU_KERNEL_REGEX      Kernel filter (default: regex:.*kq256_operand_kernel.*)
  PYTHON                Python executable for CSV evidence analysis (default: python3)
  ALLOW_CUDA_13_1       Set to 1 only for investigation; CUDA 13.1 is rejected by default

The shipping gate is not just a successful run. By default this script requires
Spark hardware (`spark_target: yes`) and fails unless the imported Nsight Compute
CSV contains FP4-specific tensor-core evidence for kq256_operand_kernel. Use
--allow-proxy only for local non-shipping sm_120/sm_120a proxy runs, and pair it
with the matching proxy architecture, for example `--arch 120a`.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmake_bin="${CMAKE:-cmake}"
ncu_bin="${NCU:-ncu}"
python_bin="${PYTHON:-python3}"
build_dir="build-spark-sm121-kq256"
out_dir=""
arch="121a"
device="0"
threads="128 256 512"
mtp_rows="4"
iters="20000"
ncu_iters="2000"
ncu_threads="512"
do_build=1
do_ncu=1
dry_run=0
allow_proxy=0
preflight_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) build_dir="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --arch) arch="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --mtp-rows) mtp_rows="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --ncu-iters) ncu_iters="$2"; shift 2 ;;
        --ncu-threads) ncu_threads="$2"; shift 2 ;;
        --allow-proxy) allow_proxy=1; shift ;;
        --preflight-only) preflight_only=1; shift ;;
        --no-build) do_build=0; shift ;;
        --no-ncu) do_ncu=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$out_dir" ]]; then
    out_dir="/tmp/llamacpp-spark-kq256-$(date +%Y%m%d-%H%M%S)"
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

normalize_list() {
    echo "$1" | tr ',' ' '
}

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

run_logged() {
    local log_path="$1"
    shift
    echo "+ $(quote_cmd "$@")" | tee -a "$log_path"
    if [[ "$dry_run" -eq 0 ]]; then
        "$@" 2>&1 | tee -a "$log_path"
    fi
}

run_redirected() {
    local log_path="$1"
    local stdout_path="$2"
    shift 2
    echo "+ $(quote_cmd "$@") > $(printf '%q' "$stdout_path")" | tee -a "$log_path"
    if [[ "$dry_run" -eq 0 ]]; then
        "$@" > "$stdout_path" 2>> "$log_path"
    fi
}

assert_spark_log() {
    local log_path="$1"
    if [[ "$dry_run" -ne 0 || "$allow_proxy" -ne 0 ]]; then
        return
    fi

    if ! grep -q '^spark_target:[[:space:]]*yes$' "$log_path"; then
        {
            echo "Spark gate failed: expected spark_target: yes in $log_path"
            echo "Use --allow-proxy only for local non-shipping proxy runs."
        } | tee -a "$summary"
        exit 1
    fi
}

validate_arch() {
    if [[ "$allow_proxy" -eq 0 && "$arch" != "121a" ]]; then
        {
            echo "Spark gate failed: shipping Spark runs must use --arch 121a, got --arch $arch"
            echo "Use --allow-proxy only for local non-shipping proxy runs such as --arch 120a."
        } | tee -a "$summary"
        exit 1
    fi
}

validate_cuda_toolkit() {
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "nvcc not found; skipping CUDA toolkit version validation" | tee -a "$summary"
        return
    fi

    local version
    version="$(nvcc --version | sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1.\2/p' | head -n 1)"
    if [[ -z "$version" ]]; then
        echo "nvcc_version_parse=unavailable" | tee -a "$summary"
        return
    fi

    local major="${version%%.*}"
    local minor="${version#*.}"
    echo "nvcc_release=$version" | tee -a "$summary"

    if (( major < 12 || (major == 12 && minor < 8) )); then
        echo "Spark gate failed: CUDA >= 12.8 is required for Blackwell block-scaled FP4 MMA, got $version" | tee -a "$summary"
        exit 1
    fi

    if (( major == 13 && minor == 1 )) && [[ "${ALLOW_CUDA_13_1:-0}" != "1" ]]; then
        {
            echo "Spark gate failed: CUDA 13.1 is rejected by default due to the known MMQ field report."
            echo "Set ALLOW_CUDA_13_1=1 only for explicit investigation, not for shipping evidence."
        } | tee -a "$summary"
        exit 1
    fi
}

validate_ncu_tool() {
    if [[ "$do_ncu" -eq 0 ]]; then
        return
    fi
    if [[ "$dry_run" -ne 0 ]]; then
        echo "ncu_validation=dry-run" | tee -a "$summary"
        return
    fi
    if ! command -v "$ncu_bin" >/dev/null 2>&1; then
        {
            echo "Spark gate failed: Nsight Compute CLI not found: $ncu_bin"
            echo "Set NCU=/path/to/ncu or use --no-ncu only for non-shipping investigation."
        } | tee -a "$summary"
        exit 1
    fi
    echo "ncu_validation=found" | tee -a "$summary"
}

record_host_diagnostics() {
    local diag="$out_dir/host-diagnostics.log"
    {
        echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "repo_root=$repo_root"
        echo "build_dir=$build_dir"
        echo "requested_arch=$arch"
        echo "requested_device=$device"
        echo
        echo "## uname"
        if command -v uname >/dev/null 2>&1; then
            uname -a
        else
            echo "unavailable"
        fi
        echo
        echo "## os-release"
        if [[ -r /etc/os-release ]]; then
            sed -n '1,80p' /etc/os-release
        else
            echo "unavailable"
        fi
        echo
        echo "## cmake"
        if command -v "$cmake_bin" >/dev/null 2>&1; then
            "$cmake_bin" --version 2>&1 | sed -n '1,20p'
        else
            echo "missing: $cmake_bin"
        fi
        echo
        echo "## nvcc"
        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version 2>&1 | sed -n '1,20p'
        else
            echo "missing"
        fi
        echo
        echo "## ncu"
        if command -v "$ncu_bin" >/dev/null 2>&1; then
            "$ncu_bin" --version 2>&1 | sed -n '1,40p'
        else
            echo "missing: $ncu_bin"
        fi
        echo
        echo "## nvidia-smi -L"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi -L 2>&1
        else
            echo "missing"
        fi
        echo
        echo "## nvidia-smi query"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv,noheader 2>&1 || true
        else
            echo "missing"
        fi
        echo
        echo "## disk"
        df -h "$out_dir" "$repo_root" 2>&1 || true
    } > "$diag"
    echo "host_diagnostics=$diag" | tee -a "$summary"
}

write_manifest() {
    local manifest_path="$out_dir/artifact-manifest.tsv"
    local tmp_path="$out_dir/artifact-manifest.tmp"
    {
        printf 'sha256\tsize_bytes\tpath\n'
        while IFS= read -r -d '' file_path; do
            local rel_path="${file_path#$out_dir/}"
            local size_bytes
            size_bytes="$(wc -c < "$file_path" | tr -d '[:space:]')"
            local sha
            if command -v sha256sum >/dev/null 2>&1; then
                sha="$(sha256sum "$file_path" | awk '{print $1}')"
            elif command -v shasum >/dev/null 2>&1; then
                sha="$(shasum -a 256 "$file_path" | awk '{print $1}')"
            else
                sha="sha256-unavailable"
            fi
            printf '%s\t%s\t%s\n' "$sha" "$size_bytes" "$rel_path"
        done < <(find "$out_dir" -maxdepth 1 -type f ! -name 'artifact-manifest.tsv' ! -name 'artifact-manifest.tmp' -print0 | sort -z)
    } > "$tmp_path"
    mv "$tmp_path" "$manifest_path"
}

finalize_artifacts() {
    local exit_code="$?"
    if [[ "${finalizing_artifacts:-0}" -eq 1 ]]; then
        return "$exit_code"
    fi
    finalizing_artifacts=1

    if [[ -n "${out_dir:-}" && -d "$out_dir" && -n "${summary:-}" && -f "$summary" ]]; then
        grep -q '^artifact_manifest=' "$summary" || echo "artifact_manifest=$out_dir/artifact-manifest.tsv" >> "$summary"
        grep -q '^artifacts=' "$summary" || echo "artifacts=$out_dir" >> "$summary"
        grep -q '^exit_code=' "$summary" || echo "exit_code=$exit_code" >> "$summary"
        write_manifest || true
    fi

    return "$exit_code"
}

out_dir="$(normalize_path "$out_dir")"
mkdir -p "$out_dir"

summary="$out_dir/summary.txt"
{
    echo "repo_root=$repo_root"
    echo "build_dir=$build_dir"
    echo "out_dir=$out_dir"
    echo "arch=$arch"
    echo "device=$device"
    echo "threads=$threads"
    echo "mtp_rows=$mtp_rows"
    echo "iters=$iters"
    echo "ncu_iters=$ncu_iters"
    echo "ncu_threads=$ncu_threads"
    echo "allow_proxy=$allow_proxy"
    echo "preflight_only=$preflight_only"
    echo "python=$python_bin"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git -C "$repo_root" rev-parse HEAD 2>/dev/null | sed 's/^/git_head=/'
    status_tmp="$out_dir/git-status.tmp"
    git -C "$repo_root" status --short --untracked-files=no > "$status_tmp" 2>/dev/null || true
    echo "git_status_count=$(wc -l < "$status_tmp")"
    grep -E '^( M|M |A | D|D |R |C |UU|AA|DD) (tools/spark-roofline|tools/CMakeLists.txt)' "$status_tmp" \
        | sed -n '1,200s/^/git_status_relevant=/p' || true
    rm -f "$status_tmp"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | sed 's/^/nvidia_smi=/'
    fi
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version | sed 's/^/nvcc=/'
    fi
    if command -v "$ncu_bin" >/dev/null 2>&1; then
        "$ncu_bin" --version | sed 's/^/ncu=/'
    fi
} | tee "$summary"

trap finalize_artifacts EXIT

record_host_diagnostics
validate_arch
if [[ "$do_build" -eq 1 ]]; then
    validate_cuda_toolkit
fi
validate_ncu_tool

if [[ "$do_build" -eq 1 ]]; then
    run_logged "$out_dir/cmake-configure.log" \
        "$cmake_bin" -S "$repo_root" -B "$repo_root/$build_dir" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF

    run_logged "$out_dir/cmake-build.log" \
        "$cmake_bin" --build "$repo_root/$build_dir" --config Release --target llama-spark-kq256 -j "$(nproc)"
fi

exe=""
for candidate in \
    "$repo_root/$build_dir/bin/llama-spark-kq256" \
    "$repo_root/$build_dir/bin/llama-spark-kq256.exe" \
    "$repo_root/$build_dir/bin/Release/llama-spark-kq256" \
    "$repo_root/$build_dir/bin/Release/llama-spark-kq256.exe"; do
    if [[ -x "$candidate" ]]; then
        exe="$candidate"
        break
    fi
done
if [[ "$dry_run" -eq 0 && ! -x "$exe" ]]; then
    echo "missing executable under $repo_root/$build_dir/bin" >&2
    exit 1
fi
if [[ -z "$exe" ]]; then
    exe="$repo_root/$build_dir/bin/llama-spark-kq256"
fi

if [[ "$preflight_only" -eq 1 ]]; then
    preflight_log="$out_dir/kq256-preflight.log"
    run_logged "$preflight_log" \
        "$exe" --device "$device" --threads 32 --mtp-rows 1 --iters 1
    assert_spark_log "$preflight_log"
    if [[ "$dry_run" -ne 0 ]]; then
        echo "preflight=dry-run" | tee -a "$summary"
    else
        echo "preflight=pass" | tee -a "$summary"
    fi
    echo "artifacts=$out_dir" | tee -a "$summary"
    exit 0
fi

for t in $(normalize_list "$threads"); do
    run_log="$out_dir/kq256-threads-${t}.log"
    run_logged "$run_log" \
        "$exe" --device "$device" --threads "$t" --mtp-rows "$mtp_rows" --iters "$iters"
    assert_spark_log "$run_log"
done

if [[ "$do_ncu" -eq 1 ]]; then
    if [[ "$dry_run" -eq 0 ]] && ! command -v "$ncu_bin" >/dev/null 2>&1; then
        echo "Nsight Compute CLI not found: $ncu_bin" | tee -a "$summary"
        exit 1
    fi

    for t in $(normalize_list "$ncu_threads"); do
        report="$out_dir/ncu-kq256-threads-${t}"
        ncu_cmd=(
            "$ncu_bin"
            --target-processes all
            --kernel-name-base function
            --kernel-name "${NCU_KERNEL_REGEX:-regex:.*kq256_operand_kernel.*}"
            --set "${NCU_SET:-full}"
            --export "$report"
            --force-overwrite
        )
        if [[ -n "${NCU_METRICS:-}" ]]; then
            ncu_cmd+=(--metrics "$NCU_METRICS")
        fi
        ncu_cmd+=("$exe" --device "$device" --threads "$t" --mtp-rows "$mtp_rows" --iters "$ncu_iters")

        run_logged "$out_dir/ncu-kq256-threads-${t}.log" "${ncu_cmd[@]}"

        report_file="${report}.ncu-rep"
        if [[ "$dry_run" -eq 0 && ! -f "$report_file" && -f "$report" ]]; then
            report_file="$report"
        fi

        raw_csv="$out_dir/ncu-kq256-threads-${t}-raw.csv"
        import_log="$out_dir/ncu-kq256-threads-${t}-import.log"
        import_cmd=("$ncu_bin" --import "$report_file" --page raw --csv)
        if [[ -n "${NCU_IMPORT_METRICS:-}" ]]; then
            import_cmd+=(--metrics "$NCU_IMPORT_METRICS")
        fi
        if [[ "$dry_run" -eq 0 && ! -f "$report_file" ]]; then
            echo "missing Nsight Compute report for raw CSV import: $report_file" | tee -a "$import_log"
            exit 1
        fi
        run_redirected "$import_log" "$raw_csv" "${import_cmd[@]}"

        evidence_txt="$out_dir/ncu-kq256-threads-${t}-fp4-evidence.txt"
        evidence_json="$out_dir/ncu-kq256-threads-${t}-fp4-evidence.json"
        analyzer="$repo_root/tools/spark-roofline/analyze-ncu-fp4-evidence.py"
        analyzer_cmd=(
            "$python_bin" "$analyzer"
            --csv "$raw_csv"
            --out "$evidence_json"
            --text "$evidence_txt"
            --kernel-regex "kq256_operand_kernel"
        )
        if [[ "$dry_run" -eq 0 ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
            echo "Python not found for Nsight CSV evidence analysis: $python_bin" | tee -a "$summary"
            exit 1
        fi
        run_logged "$out_dir/ncu-kq256-threads-${t}-analyze.log" "${analyzer_cmd[@]}"
        if [[ "$dry_run" -ne 0 ]]; then
            {
                echo "source_csv=$raw_csv"
                echo "dry_run=true"
                echo "analyzer=$analyzer"
                echo "output_json=$evidence_json"
            } > "$evidence_txt"
        fi
    done
fi

summary_csv="$out_dir/kq256-summary.csv"
summary_json="$out_dir/kq256-summary.json"
summary_txt="$out_dir/kq256-summary.txt"
summarizer="$repo_root/tools/spark-roofline/summarize-kq256-gate.py"
summarizer_cmd=(
    "$python_bin" "$summarizer"
    --dir "$out_dir"
    --csv "$summary_csv"
    --json "$summary_json"
    --text "$summary_txt"
    --requested-device "$device"
)
if [[ "$allow_proxy" -eq 0 ]]; then
    summarizer_cmd+=(--require-spark)
fi
if [[ "$do_ncu" -eq 1 ]]; then
    summarizer_cmd+=(--require-ncu)
fi
run_logged "$out_dir/kq256-summary.log" "${summarizer_cmd[@]}"
if [[ "$dry_run" -ne 0 ]]; then
    {
        echo "dry_run=true"
        echo "summary_csv=$summary_csv"
        echo "summary_json=$summary_json"
        echo "summary_txt=$summary_txt"
    } > "$summary_txt"
fi

echo "artifacts=$out_dir" | tee -a "$summary"

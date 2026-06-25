# D=256 KQ Spark Gate Runbook

This runbook is for the real DGX Spark / GB10 `sm_121a` gate for
`llama-spark-kq256`. It is not a proxy benchmark recipe.

## Required Result

The D=256 KQ gate is green only when all are true:

- the run is on the Spark `sm_121a` device;
- `cmake-configure.log` proves the build used `CMAKE_CUDA_ARCHITECTURES=121a`;
- `summary.txt` records the expected `git_head`;
- `summary.txt` records `git_status_relevant_count=0` for `.gitattributes`,
  `tools/CMakeLists.txt`, and `tools/spark-roofline`;
- `kq256-summary.json` reports `gate_decision=go`;
- at least one Nsight Compute evidence JSON reports passing FP4-specific evidence;
- the required Nsight thread shape, default `512`, has a complete passing artifact
  set and `kq256-summary.json` marks that row `ncu_passed=true`;
- `host-diagnostics.log` is present and manifest-hashed;
- `summary.txt` / `host-diagnostics.log` agree that the requested arch is `121a`;
- the selected host device reports `compute_cap=12.1`;
- `summary.txt` / `host-diagnostics.log` prove an acceptable CUDA toolkit:
  `nvcc` release is at least 12.8 and is not 13.1;
- the copied bundle SHA-256 sidecar verifies;
- `artifact-manifest.tsv` verifies after copying or extracting the bundle.

## Local Tooling Self-Test

Before changing the handoff scripts or running a Spark handoff, run the local
synthetic self-test:

```bash
python tools/spark-roofline/test-kq256-handoff-tools.py
```

This does not use a GPU and does not produce Spark evidence. It checks that the
Nsight CSV analyzer, KQ256 summarizer, copied-artifact verifier, host diagnostics
requirement, and bundle verifier still agree on the same pass/fail contract.

## Preflight

The preferred Spark command is the end-to-end handoff wrapper:

```bash
bash tools/spark-roofline/run-spark-kq256-handoff.sh \
  --device 0 \
  --build-dir build-spark-sm121-kq256 \
  --out-dir /tmp/llamacpp-spark-kq256 \
  --threads "128 256 512" \
  --mtp-rows 4 \
  --iters 20000 \
  --ncu-threads "512" \
  --ncu-iters 2000
```

The wrapper runs the local tooling self-test, Spark preflight, full gate, bundle
creation, and strict bundle verification in order. The full gate reuses the
preflight build by default; in that mode the wrapper copies the preflight
`cmake-configure.log` and `cmake-build.log` into the full artifact directory so
the copied bundle can prove the build architecture. Pass `--rebuild-full` only
if you explicitly want a second build. The wrapper prints the bundle and
`.sha256` paths to copy off the Spark host, and writes a small handoff summary at
`/tmp/llamacpp-spark-kq256-handoff-summary.txt` by default.

The handoff summary is written on success, dry-run, and failure. It records
`handoff_status`, `handoff_complete`, `exit_code`, `last_stage`, copy paths for
the bundle, `.sha256` sidecar, and summary file, plus a strict copied-bundle
verification command.
That command includes the exact `git_head` from the checkout used on the Spark
host, so copied results cannot accidentally verify against a different commit.

A wrapper `--dry-run` is only a command plan. Its summary reports
`handoff_status=planned` and `handoff_complete=false`; do not treat dry-run
output as Spark evidence.

The manual steps below are the same sequence broken out for debugging.

Run this first on the Spark host:

```bash
bash tools/spark-roofline/run-spark-kq256-gate.sh \
  --preflight-only \
  --device 0 \
  --build-dir build-spark-sm121-kq256 \
  --out-dir /tmp/llamacpp-spark-kq256-preflight
```

Use a different `--device` only if `nvidia-smi -L` shows the Spark target is not
device 0. Do not use `--allow-proxy` for a shipping/Spark claim.

The preflight should end with:

```text
preflight=pass
host_diagnostics=/tmp/llamacpp-spark-kq256-preflight/host-diagnostics.log
artifact_manifest=/tmp/llamacpp-spark-kq256-preflight/artifact-manifest.tsv
```

The `kq256-preflight.log` must contain:

```text
spark_target:         yes
blackwell_fp4_target: yes
```

## Vast.ai Discovery Check

If using Vast.ai to look for a temporary host, run the read-only discovery
helper before renting anything:

```bash
python tools/spark-roofline/check-vast-spark-offers.py --limit 50
```

This helper only searches offers. It does not create, launch, start, stop, or
modify instances.

The Spark gate still requires a real `sm_121a` / GB10 target. The helper treats
common nearby Blackwell offers as near misses:

- `compute_cap=1200`: RTX 50 / RTX PRO 6000-class `sm_120`, useful only as a
  proxy.
- `compute_cap=1000`: B200 / GB200-class `sm_100`, not Spark.

Exit codes:

- `0`: at least one Spark-looking candidate was found.
- `2`: the search succeeded, but no Spark-looking candidate was found.
- other nonzero: the search itself failed.

## Upstream Work Discovery Check

Before spending time on a new Spark run or before packaging a branch, refresh
the read-only upstream search:

```bash
python tools/spark-roofline/check-upstream-nvfp4-work.py --limit 50
```

This helper shells out to `gh search` only. It does not create or update GitHub
issues, PRs, comments, branches, or labels.

The default query set is intentionally small to reduce GitHub search-rate-limit
pressure. Before packaging a branch, run the broader sweep too:

```bash
python tools/spark-roofline/check-upstream-nvfp4-work.py --limit 50 --exhaustive
```

Exit codes:

- `0`: search completed and no directly competing NVFP4/FP4 KV FlashAttention
  work was found.
- `1`: search completed and at least one direct match was found; inspect it
  before proceeding.
- `2`: search was incomplete, usually because `gh` is missing or GitHub search
  rate-limited the request.

## GitHub Runner Discovery Check

Before assuming the GitHub runner fleet can provide Spark evidence, run the
read-only runner audit:

```bash
python tools/spark-roofline/check-github-spark-runners.py --repo jethac/llama.cpp
```

This scans local workflow `runs-on` labels and queries the selected repository's
self-hosted runner inventory through `gh api` when permitted. It does not
dispatch workflows, register runners, or change repository settings.

For the public upstream repository, the runner inventory may be hidden from
non-admin tokens:

```bash
python tools/spark-roofline/check-github-spark-runners.py --repo ggml-org/llama.cpp
```

Exit codes:

- `0`: a Spark-looking self-hosted runner was found.
- `2`: runner inventory was queried successfully and no Spark-looking runner was
  found.
- `3`: runner inventory could not be queried, usually because GitHub returned
  `403` for the Actions runner endpoint.

## Full Gate

Run the full gate after preflight passes:

```bash
bash tools/spark-roofline/run-spark-kq256-gate.sh \
  --device 0 \
  --build-dir build-spark-sm121-kq256 \
  --out-dir /tmp/llamacpp-spark-kq256 \
  --threads "128 256 512" \
  --mtp-rows 4 \
  --iters 20000 \
  --ncu-threads "512" \
  --ncu-iters 2000
```

The runner enforces `--arch 121a` by default. It also rejects CUDA versions
below 12.8 and rejects CUDA 13.1 unless `ALLOW_CUDA_13_1=1` is explicitly set
for investigation. Do not set that override for shipping evidence.

## Expected Artifacts

The artifact directory should contain at least:

- `summary.txt`
- `host-diagnostics.log`
- `cmake-configure.log`
- `cmake-build.log`
- `kq256-threads-128.log`
- `kq256-threads-256.log`
- `kq256-threads-512.log`
- `ncu-kq256-threads-512.ncu-rep`
- `ncu-kq256-threads-512-raw.csv`
- `ncu-kq256-threads-512-fp4-evidence.json`
- `ncu-kq256-threads-512-fp4-evidence.txt`
- `kq256-summary.csv`
- `kq256-summary.json`
- `kq256-summary.txt`
- `artifact-manifest.tsv`

If `--ncu-threads` changes, replace `512` in the Nsight artifact names with
the selected thread count.

## Pass Criteria

Inspect the text summary:

```bash
cat /tmp/llamacpp-spark-kq256/kq256-summary.txt
```

Required lines:

```text
passed=true
gate_decision=go
```

The best row is selected by useful-MTP KQ TOPS:

```text
best_threads=...
best_useful_mtp_kq_tops=...
best_kq_tops=...
best_k_read_gbps=...
best_q_quant_gbps=...
```

The Nsight evidence text must show nonzero FP4-specific evidence:

```bash
cat /tmp/llamacpp-spark-kq256/ncu-kq256-threads-512-fp4-evidence.txt
```

Generic HMMA/tensor-only evidence is not enough for the default gate.

## Copy And Verify

If using the wrapper, the bundle is already created and verified. If running
manual steps, create a portable bundle before copying the artifact directory off
the Spark host:

```bash
python tools/spark-roofline/bundle-kq256-artifacts.py create \
  --dir /tmp/llamacpp-spark-kq256 \
  --out /tmp/llamacpp-spark-kq256.tar.gz \
  --verify-first
```

Copy these files off the Spark host:

```text
/tmp/llamacpp-spark-kq256.tar.gz
/tmp/llamacpp-spark-kq256.tar.gz.sha256
/tmp/llamacpp-spark-kq256-handoff-summary.txt
```

After copying, verify the bundle. This checks the bundle SHA-256, safely
extracts it, verifies `artifact-manifest.tsv`, and runs the strict copied
artifact gate:

```bash
python tools/spark-roofline/bundle-kq256-artifacts.py verify \
  --bundle /path/to/copied/llamacpp-spark-kq256.tar.gz \
  --require-go \
  --require-ncu \
  --require-ncu-threads 512 \
  --require-git-head <expected-git-head> \
  --require-clean-relevant-git \
  --require-host-diagnostics \
  --require-host-arch 121a \
  --require-host-compute-cap 12.1 \
  --require-build-arch 121a \
  --require-cuda-min 12.8 \
  --reject-cuda-release 13.1
```

The bundle verifier should report:

```text
passed=true
gate_decision=go
bundle_verified=true
git_required_head=<expected-git-head>
git_status_relevant_count=0
git_clean_relevant_required=true
build_required_arch_matched=121a
host_required_arch=121a
host_required_compute_cap_matched=12.1
cuda_required_min=12.8
cuda_rejected_releases=13.1
ncu_required_threads=512
ncu_complete_passed_threads=512
```

The verifier rejects unsafe archive members and manifest paths, including
absolute paths, `..` traversal, links, special tar entries, duplicate manifest
entries, and bundles with multiple top-level artifact directories.
It also requires the `.sha256` sidecar to contain exactly one valid SHA-256
line whose filename matches the copied bundle name.

Only after this verification should the D=256 KQ gate be treated as green and
the next work item move to D=256 mixed PV / SWA / Gemma 3 quality.

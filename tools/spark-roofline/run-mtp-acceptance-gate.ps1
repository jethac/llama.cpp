[CmdletBinding()]
param(
    [string] $Model = "",

    [string] $BuildDir = "",

    [string] $ServerExe = "",

    [string] $PythonExe = "python",

    [string] $OutDir = "",

    [string[]] $CacheTypes = @("f16", "nvfp4"),

    [int] $GpuLayers = 999,

    [int] $ContextSize = 8192,

    [int] $BatchSize = 512,

    [int] $UbatchSize = 512,

    [int] $Parallel = 1,

    [int] $Port = 18080,

    [string] $Bench = "qualitative",

    [string] $Category = "coding",

    [int] $OutputTokens = 128,

    [int] $Limit = 2,

    [string] $SpecType = "draft-mtp",

    [int] $SpecDraftNMax = 3,

    [string] $DraftModel = "",

    [string[]] $ServerExtraArgs = @(),

    [string[]] $SpecExtraArgs = @(),

    [string[]] $SpeedBenchExtraArgs = @(),

    [string] $SummarizeRunDir = "",

    [switch] $SkipPythonDependencyCheck,

    [switch] $NoBuild,

    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

$CacheTypes = @($CacheTypes | ForEach-Object {
    $_ -split ","
} | ForEach-Object {
    $_.Trim()
} | Where-Object {
    $_ -ne ""
})

function Resolve-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $Path))
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Exe,

        [Parameter(Mandatory = $true)]
        [string[]] $Args
    )

    $parts = @($Exe) + $Args
    return ($parts | ForEach-Object {
        if ($_ -match '\s') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
        [string] $Exe,

        [Parameter(Mandatory = $true)]
        [string[]] $Args,

        [Parameter(Mandatory = $true)]
        [string] $LogPath
    )

    $cmdLine = Format-CommandLine -Exe $Exe -Args $Args
    Write-Host ""
    Write-Host "[$Name]"
    Write-Host $cmdLine

    if ($DryRun) {
        return
    }

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Args 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. See $LogPath"
    }
}

function Test-PythonDependency {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Python,

        [Parameter(Mandatory = $true)]
        [string[]] $Modules
    )

    foreach ($module in $Modules) {
        $code = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module') is not None else 1)"
        & $Python -c $code | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Python module '$module' is missing. Install SPEED-Bench dependencies with: $Python -m pip install -r tools/server/bench/speed-bench/requirements.txt"
        }
    }
}

function Wait-ServerReady {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Url,

        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process] $Process,

        [string] $ErrorLog = "",

        [int] $TimeoutSeconds = 180
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            $tail = ""
            if ($ErrorLog -ne "" -and (Test-Path -LiteralPath $ErrorLog -PathType Leaf)) {
                $tail = (Get-Content -LiteralPath $ErrorLog -Tail 20) -join "`n"
            }
            throw "llama-server exited before becoming ready. Exit code: $($Process.ExitCode)`n$tail"
        }

        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "$Url/health" -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        } catch {
        }

        Start-Sleep -Seconds 1
    }

    throw "Timed out waiting for llama-server at $Url"
}

function Read-SpeedBenchSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunDir,

        [Parameter(Mandatory = $true)]
        [string] $CacheType
    )

    $jsonPath = Join-Path $RunDir "$CacheType-speed-bench.json"
    if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) {
        return $null
    }

    $data = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
    $overall = $null
    foreach ($row in @($data.summary)) {
        if ($row.category -eq "overall") {
            $overall = $row
            break
        }
    }
    if ($null -eq $overall -and @($data.summary).Count -gt 0) {
        $overall = @($data.summary)[0]
    }
    if ($null -eq $overall) {
        return $null
    }

    return [pscustomobject]@{
        cache_type       = $CacheType
        json_path        = $jsonPath
        samples          = $overall.requests
        avg_prompt_tps   = $overall.avg_prompt_t_s
        avg_pred_tps     = $overall.avg_pred_t_s
        avg_latency_s    = $overall.avg_latency
        draft_n          = $overall.draft_n
        draft_n_accepted = $overall.accepted
        accept_rate      = $overall.accept_rate
    }
}

function Write-AcceptanceSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunDir,

        [Parameter(Mandatory = $true)]
        [string[]] $Types
    )

    $rows = @()
    foreach ($type in $Types) {
        $row = Read-SpeedBenchSummary -RunDir $RunDir -CacheType $type
        if ($null -ne $row) {
            $rows += $row
        }
    }

    if ($rows.Count -eq 0) {
        Write-Warning "No SPEED-Bench JSON files found to summarize in $RunDir"
        return
    }

    $csvPath = Join-Path $RunDir "acceptance-summary.csv"
    $jsonPath = Join-Path $RunDir "acceptance-summary.json"
    $rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding ASCII
    $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $jsonPath -Encoding ASCII

    Write-Host ""
    Write-Host "Acceptance summary written:"
    Write-Host "  $csvPath"
    Write-Host "  $jsonPath"
    $rows | Format-Table cache_type, samples, avg_pred_tps, draft_n, draft_n_accepted, accept_rate
}

function Get-SafeFileStem {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Value
    )

    return ($Value -replace '[^A-Za-z0-9_.-]', '_')
}

function Invoke-SpeedBenchComparisons {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunDir,

        [Parameter(Mandatory = $true)]
        [string[]] $Types,

        [Parameter(Mandatory = $true)]
        [string] $CompareScript,

        [Parameter(Mandatory = $true)]
        [string] $Python
    )

    $baselineType = "f16"
    $baselineJson = Join-Path $RunDir "$baselineType-speed-bench.json"
    if (-not (Test-Path -LiteralPath $baselineJson -PathType Leaf)) {
        Write-Warning "Skipping SPEED-Bench comparisons because $baselineJson was not found"
        return
    }

    foreach ($type in @($Types | Select-Object -Unique)) {
        if ($type -eq $baselineType) {
            continue
        }

        $specJson = Join-Path $RunDir "$type-speed-bench.json"
        if (-not (Test-Path -LiteralPath $specJson -PathType Leaf)) {
            Write-Warning "Skipping f16 vs $type comparison because $specJson was not found"
            continue
        }

        $safeType = Get-SafeFileStem -Value $type
        $compareArgs = @(
            $CompareScript,
            "--baseline", $baselineJson,
            "--speculative", $specJson
        )
        Invoke-LoggedCommand -Name "compare f16 vs $type" -Exe $Python -Args $compareArgs -LogPath (Join-Path $RunDir "f16-vs-$safeType-compare.log")
    }
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$compare = Join-Path $repoRoot "tools\server\bench\speed-bench\speed_bench_compare.py"

if ($SummarizeRunDir -ne "") {
    $SummarizeRunDir = Resolve-FullPath $SummarizeRunDir
    if (-not (Test-Path -LiteralPath $SummarizeRunDir -PathType Container)) {
        throw "Run directory not found: $SummarizeRunDir"
    }
    Write-AcceptanceSummary -RunDir $SummarizeRunDir -Types $CacheTypes
    Invoke-SpeedBenchComparisons -RunDir $SummarizeRunDir -Types $CacheTypes -CompareScript $compare -Python $PythonExe
    exit 0
}

if ($Model -eq "") {
    throw "Model is required unless -SummarizeRunDir is used."
}

if ($BuildDir -eq "") {
    $BuildDir = Join-Path $repoRoot "build-nvfp4-static-sm120-tests"
}
$BuildDir = Resolve-FullPath $BuildDir

if ($OutDir -eq "") {
    $OutDir = Join-Path $repoRoot "quality-runs\mtp-acceptance"
}
$OutDir = Resolve-FullPath $OutDir
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Model = Resolve-FullPath $Model
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model file not found: $Model"
}

if ($ServerExe -eq "") {
    $ServerExe = Join-Path $BuildDir "bin\Release\llama-server.exe"
}
$ServerExe = Resolve-FullPath $ServerExe

if ($DraftModel -ne "") {
    $DraftModel = Resolve-FullPath $DraftModel
    if (-not (Test-Path -LiteralPath $DraftModel -PathType Leaf)) {
        throw "Draft model file not found: $DraftModel"
    }
}

if (-not $NoBuild) {
    $cmake = "cmake"
    $cmakeArgs = @("--build", $BuildDir, "--config", "Release", "--target", "llama-server", "-j", "8")
    Invoke-LoggedCommand -Name "build llama-server" -Exe $cmake -Args $cmakeArgs -LogPath (Join-Path $OutDir "build-llama-server.log")
}

if (-not $DryRun -and -not (Test-Path -LiteralPath $ServerExe -PathType Leaf)) {
    throw "llama-server not found: $ServerExe"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$modelStem = [System.IO.Path]::GetFileNameWithoutExtension($Model)
$runDir = Join-Path $OutDir "$timestamp-$modelStem"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$speedBench = Join-Path $repoRoot "tools\server\bench\speed-bench\speed_bench.py"
$manifest = Join-Path $runDir "manifest.txt"

@(
    "model=$Model",
    "build_dir=$BuildDir",
    "server_exe=$ServerExe",
    "python=$PythonExe",
    "cache_types=$($CacheTypes -join ',')",
    "gpu_layers=$GpuLayers",
    "ctx_size=$ContextSize",
    "batch_size=$BatchSize",
    "ubatch_size=$UbatchSize",
    "parallel=$Parallel",
    "base_port=$Port",
    "bench=$Bench",
    "category=$Category",
    "output_tokens=$OutputTokens",
    "limit=$Limit",
    "spec_type=$SpecType",
    "spec_draft_n_max=$SpecDraftNMax",
    "draft_model=$DraftModel",
    "server_extra_args=$($ServerExtraArgs -join ' ')",
    "spec_extra_args=$($SpecExtraArgs -join ' ')",
    "speed_bench_extra_args=$($SpeedBenchExtraArgs -join ' ')",
    "skip_python_dependency_check=$SkipPythonDependencyCheck",
    "dry_run=$DryRun"
) | Set-Content -LiteralPath $manifest -Encoding ASCII

Write-Host "Run directory: $runDir"

if (-not $DryRun -and -not $SkipPythonDependencyCheck) {
    Test-PythonDependency -Python $PythonExe -Modules @("requests", "datasets", "tqdm")
}

$speedOutputs = @{}
for ($i = 0; $i -lt $CacheTypes.Count; ++$i) {
    $cacheType = $CacheTypes[$i]
    $thisPort = $Port + $i
    $url = "http://127.0.0.1:$thisPort"
    $serverLog = Join-Path $runDir "$cacheType-server.log"
    $serverErr = Join-Path $runDir "$cacheType-server.err.log"
    $speedLog = Join-Path $runDir "$cacheType-speed-bench.log"
    $speedJson = Join-Path $runDir "$cacheType-speed-bench.json"
    $speedOutputs[$cacheType] = $speedJson

    $serverArgs = @(
        "-m", $Model,
        "-c", "$ContextSize",
        "-b", "$BatchSize",
        "-ub", "$UbatchSize",
        "-ngl", "$GpuLayers",
        "-fa", "on",
        "-ctk", $cacheType,
        "-ctv", $cacheType,
        "--port", "$thisPort",
        "--host", "127.0.0.1",
        "-np", "$Parallel",
        "--jinja",
        "--spec-type", $SpecType,
        "--spec-draft-n-max", "$SpecDraftNMax"
    ) + $ServerExtraArgs + $SpecExtraArgs

    if ($DraftModel -ne "") {
        $serverArgs += @("--spec-draft-model", $DraftModel)
    }

    $speedArgs = @(
        $speedBench,
        "--url", "127.0.0.1:$thisPort",
        "--bench", $Bench,
        "--category", $Category,
        "--osl", "$OutputTokens",
        "--concurrency", "$Parallel",
        "--limit", "$Limit",
        "--output", $speedJson
    ) + $SpeedBenchExtraArgs

    Write-Host ""
    Write-Host "[$cacheType server]"
    Write-Host (Format-CommandLine -Exe $ServerExe -Args $serverArgs)
    Write-Host "[$cacheType speed-bench]"
    Write-Host (Format-CommandLine -Exe $PythonExe -Args $speedArgs)

    if ($DryRun) {
        continue
    }

    $server = $null
    try {
        $server = Start-Process -FilePath $ServerExe -ArgumentList $serverArgs -RedirectStandardOutput $serverLog -RedirectStandardError $serverErr -PassThru -WindowStyle Hidden
        Wait-ServerReady -Url $url -Process $server -ErrorLog $serverErr
        Invoke-LoggedCommand -Name "$cacheType speed-bench" -Exe $PythonExe -Args $speedArgs -LogPath $speedLog
    } finally {
        if ($null -ne $server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
            $server.WaitForExit()
        }
    }
}

if (-not $DryRun) {
    Write-AcceptanceSummary -RunDir $runDir -Types $CacheTypes
    Invoke-SpeedBenchComparisons -RunDir $runDir -Types $CacheTypes -CompareScript $compare -Python $PythonExe
}

Write-Host ""
Write-Host "MTP acceptance gate artifacts written to $runDir"

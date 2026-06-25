[CmdletBinding()]
param(
    [string] $Model = "",

    [string] $Corpus = "",

    [string] $BuildDir = "",

    [string] $PerplexityExe = "",

    [string] $OutDir = "",

    [int] $GpuLayers = 999,

    [int] $ContextSize = 4096,

    [int] $BatchSize = 512,

    [int] $Threads = 0,

    [string[]] $CommonExtraArgs = @(),

    [string[]] $Nvfp4ExtraArgs = @(),

    [string] $SummarizeRunDir = "",

    [switch] $NoBuild,

    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

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

function Invoke-GateCommand {
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

function Get-RegexPair {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Pattern
    )

    $match = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        return $null
    }

    return @{
        value = [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        error = [double]::Parse($match.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

function Get-RegexValue {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,

        [Parameter(Mandatory = $true)]
        [string] $Pattern
    )

    $match = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        return $null
    }

    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-KlSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunDir,

        [Parameter(Mandatory = $true)]
        [string] $KvType,

        [Parameter(Mandatory = $true)]
        [string] $LogName
    )

    $logPath = Join-Path $RunDir $LogName
    if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) {
        return $null
    }

    $text = Get-Content -LiteralPath $logPath -Raw
    $pplQ = Get-RegexPair -Text $text -Pattern '^Mean PPL\(Q\)\s*:\s*([-+0-9.]+)\s*(?:\+/-|\u00B1)\s*([-+0-9.]+)'
    $pplBase = Get-RegexPair -Text $text -Pattern '^Mean PPL\(base\)\s*:\s*([-+0-9.]+)\s*(?:\+/-|\u00B1)\s*([-+0-9.]+)'
    $pplRatio = Get-RegexPair -Text $text -Pattern '^Mean PPL\(Q\)/PPL\(base\)\s*:\s*([-+0-9.]+)\s*(?:\+/-|\u00B1)\s*([-+0-9.]+)'
    $kld = Get-RegexPair -Text $text -Pattern '^Mean\s+KLD:\s*([-+0-9.]+)\s*(?:\+/-|\u00B1)\s*([-+0-9.]+)'
    $rmsDp = Get-RegexPair -Text $text -Pattern '^RMS \u0394p\s*:\s*([-+0-9.]+)\s*(?:\+/-|\u00B1)\s*([-+0-9.]+)\s*%'
    $sameTopP = Get-RegexPair -Text $text -Pattern '^Same top p:\s*([-+0-9.]+)\s*(?:\+/-|\u00B1)\s*([-+0-9.]+)\s*%'

    return [pscustomobject]@{
        kv_type            = $KvType
        log_path           = $logPath
        mean_ppl_q         = if ($null -ne $pplQ) { $pplQ.value } else { $null }
        mean_ppl_q_err     = if ($null -ne $pplQ) { $pplQ.error } else { $null }
        mean_ppl_base      = if ($null -ne $pplBase) { $pplBase.value } else { $null }
        mean_ppl_base_err  = if ($null -ne $pplBase) { $pplBase.error } else { $null }
        ppl_ratio          = if ($null -ne $pplRatio) { $pplRatio.value } else { $null }
        ppl_ratio_err      = if ($null -ne $pplRatio) { $pplRatio.error } else { $null }
        mean_kld           = if ($null -ne $kld) { $kld.value } else { $null }
        mean_kld_err       = if ($null -ne $kld) { $kld.error } else { $null }
        rms_delta_p_pct    = if ($null -ne $rmsDp) { $rmsDp.value } else { $null }
        rms_delta_p_pct_err = if ($null -ne $rmsDp) { $rmsDp.error } else { $null }
        same_top_p_pct     = if ($null -ne $sameTopP) { $sameTopP.value } else { $null }
        same_top_p_pct_err = if ($null -ne $sameTopP) { $sameTopP.error } else { $null }
    }
}

function Write-RunSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunDir
    )

    $rows = @()
    $rows += Get-KlSummary -RunDir $RunDir -KvType "q4_0" -LogName "q4_0-kv-kl.log"
    $rows += Get-KlSummary -RunDir $RunDir -KvType "nvfp4" -LogName "nvfp4-kv-kl.log"
    $rows = @($rows | Where-Object { $null -ne $_ })

    if ($rows.Count -eq 0) {
        Write-Warning "No KL logs found to summarize in $RunDir"
        return
    }

    $csvPath = Join-Path $RunDir "summary.csv"
    $jsonPath = Join-Path $RunDir "summary.json"
    $rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding ASCII
    $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $jsonPath -Encoding ASCII

    Write-Host ""
    Write-Host "Summary written:"
    Write-Host "  $csvPath"
    Write-Host "  $jsonPath"
    $rows | Format-Table kv_type, mean_kld, mean_kld_err, ppl_ratio, ppl_ratio_err, rms_delta_p_pct, same_top_p_pct
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")

if ($SummarizeRunDir -ne "") {
    $SummarizeRunDir = Resolve-FullPath $SummarizeRunDir
    if (-not (Test-Path -LiteralPath $SummarizeRunDir -PathType Container)) {
        throw "Run directory not found: $SummarizeRunDir"
    }
    Write-RunSummary -RunDir $SummarizeRunDir
    exit 0
}

if ($Model -eq "") {
    throw "Model is required unless -SummarizeRunDir is used."
}
if ($Corpus -eq "") {
    throw "Corpus is required unless -SummarizeRunDir is used."
}

if ($BuildDir -eq "") {
    $BuildDir = Join-Path $repoRoot "build-nvfp4-static-sm120-tests"
}
$BuildDir = Resolve-FullPath $BuildDir

if ($OutDir -eq "") {
    $OutDir = Join-Path $repoRoot "quality-runs\nvfp4-kv"
}
$OutDir = Resolve-FullPath $OutDir
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Model = Resolve-FullPath $Model
$Corpus = Resolve-FullPath $Corpus

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model file not found: $Model"
}
if (-not (Test-Path -LiteralPath $Corpus -PathType Leaf)) {
    throw "Corpus file not found: $Corpus"
}

if ($PerplexityExe -eq "") {
    $PerplexityExe = Join-Path $BuildDir "bin\Release\llama-perplexity.exe"
}
$PerplexityExe = Resolve-FullPath $PerplexityExe

if (-not $NoBuild) {
    $cmake = "cmake"
    $cmakeArgs = @("--build", $BuildDir, "--config", "Release", "--target", "llama-perplexity", "-j", "8")
    Invoke-GateCommand -Name "build llama-perplexity" -Exe $cmake -Args $cmakeArgs -LogPath (Join-Path $OutDir "build-llama-perplexity.log")
}

if (-not $DryRun -and -not (Test-Path -LiteralPath $PerplexityExe -PathType Leaf)) {
    throw "llama-perplexity not found: $PerplexityExe"
}

$modelStem = [System.IO.Path]::GetFileNameWithoutExtension($Model)
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $OutDir "$timestamp-$modelStem"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$baseLogits = Join-Path $runDir "f16-kv-base.kld"
$manifest = Join-Path $runDir "manifest.txt"

$commonArgs = @(
    "-m", $Model,
    "-f", $Corpus,
    "-ngl", "$GpuLayers",
    "-c", "$ContextSize",
    "-b", "$BatchSize",
    "-fa", "on"
) + $CommonExtraArgs

if ($Threads -gt 0) {
    $commonArgs += @("-t", "$Threads")
}

$baseArgs = $commonArgs + @(
    "-ctk", "f16",
    "-ctv", "f16",
    "--save-all-logits", $baseLogits
)

$q40Args = $commonArgs + @(
    "-ctk", "q4_0",
    "-ctv", "q4_0",
    "--kl-divergence-base", $baseLogits,
    "--kl-divergence"
)

$nvfp4Args = $commonArgs + @(
    "-ctk", "nvfp4",
    "-ctv", "nvfp4",
    "--kl-divergence-base", $baseLogits,
    "--kl-divergence"
) + $Nvfp4ExtraArgs

$manifestLines = @(
    "model=$Model",
    "corpus=$Corpus",
    "build_dir=$BuildDir",
    "perplexity_exe=$PerplexityExe",
    "gpu_layers=$GpuLayers",
    "ctx_size=$ContextSize",
    "batch_size=$BatchSize",
    "threads=$Threads",
    "common_extra_args=$($CommonExtraArgs -join ' ')",
    "nvfp4_extra_args=$($Nvfp4ExtraArgs -join ' ')",
    "base_logits=$baseLogits",
    "dry_run=$DryRun"
)
$manifestLines | Set-Content -LiteralPath $manifest -Encoding ASCII

Write-Host "Run directory: $runDir"
Write-Host "Warning: --save-all-logits can create a very large .kld file. Use a short smoke corpus before full Wikitext/Gemma runs."

Invoke-GateCommand -Name "F16 KV baseline logits" -Exe $PerplexityExe -Args $baseArgs -LogPath (Join-Path $runDir "f16-kv.log")
Invoke-GateCommand -Name "q4_0 KV KL reference" -Exe $PerplexityExe -Args $q40Args -LogPath (Join-Path $runDir "q4_0-kv-kl.log")
Invoke-GateCommand -Name "NVFP4 KV KL candidate" -Exe $PerplexityExe -Args $nvfp4Args -LogPath (Join-Path $runDir "nvfp4-kv-kl.log")

if (-not $DryRun) {
    Write-RunSummary -RunDir $runDir
}

Write-Host ""
Write-Host "Quality gate logs written to $runDir"

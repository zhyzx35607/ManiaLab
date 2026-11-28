[CmdletBinding()]
param(
    [switch]$Smoke,
    [switch]$Resume,
    [ValidatePattern("^dataset_provisional(?:_[A-Za-z0-9_-]+)?$")]
    [string]$SmokeDatasetName = "dataset_provisional",
    [ValidateSet("v32-mini", "v32")]
    [string]$Model = "v32-mini",
    [ValidateRange(1, 1000000)]
    [int]$Epochs = 3,
    [ValidateRange(0, 100000000)]
    [int]$Steps = 0,
    [ValidateRange(0, 16)]
    [int]$Workers = 2
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$kitRoot = Join-Path $repoRoot "workspace_exp\lora_6_10k_kit"
$runtimeRoot = Join-Path $kitRoot "runtime"
$cacheRoot = Join-Path $runtimeRoot "cache"
$pythonCandidates = @(
    (Join-Path $runtimeRoot "venv\Scripts\python.exe"),
    (Join-Path $repoRoot "workspace_exp\phase06\v32-runtime\venv\Scripts\python.exe"),
    (Join-Path $repoRoot ".venv\Scripts\python.exe")
)
$python = $pythonCandidates | Where-Object {
    Test-Path -LiteralPath $_ -PathType Leaf
} | Select-Object -First 1
$sourceRoot = Join-Path $repoRoot "workspace_exp\phase06\v32-runtime\source"
$torosuCli = Join-Path $repoRoot "engines\torosu_cli\torosu-cli.exe"
$datasetName = if ($Smoke) { $SmokeDatasetName } else { "dataset" }
$datasetDir = Join-Path $kitRoot $datasetName
$datasetManifest = Join-Path $datasetDir "manifest.json"
$minimumFreeGiB = 4.0
$minimumCFreeGiB = 5.0

if (-not $repoRoot.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The overnight CPU runner requires the ManiaLab workspace on D:."
}
if (-not $python) {
    throw "No installed training Python was found. Expected an existing kit, Phase 06 V32, or project virtual environment; this runner does not install tools."
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "inference.py") -PathType Leaf)) {
    throw "The pinned local Mapperatorinator source is missing at $sourceRoot."
}
if (-not (Test-Path -LiteralPath $torosuCli -PathType Leaf)) {
    throw "The prebuilt Torosu CLI is missing at $torosuCli."
}
if (-not (Test-Path -LiteralPath $datasetManifest -PathType Leaf)) {
    throw "Dataset manifest is missing at $datasetManifest."
}

$env:TMP = Join-Path $runtimeRoot "tmp"
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:HF_HOME = Join-Path $cacheRoot "huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
Remove-Item Env:TRANSFORMERS_CACHE -ErrorAction SilentlyContinue
$env:TORCH_HOME = Join-Path $cacheRoot "torch"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:TORCH_HOME "inductor"
$env:XDG_CACHE_HOME = Join-Path $cacheRoot "xdg"
$env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:CARGO_HOME = Join-Path $runtimeRoot "cargo-home"
$env:RUSTUP_HOME = Join-Path $runtimeRoot "rustup-home"
$env:MANIALAB_CACHE_ROOT = $cacheRoot
if (-not $env:MANIALAB_V32_MODEL_CACHE) {
    $env:MANIALAB_V32_MODEL_CACHE = Join-Path $env:HF_HOME "manialab"
}
$env:MANIALAB_MAPPER_REPO = $sourceRoot
$env:MANIALAB_TOROSU_CLI = $torosuCli
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONNOUSERSITE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "5"
$env:MKL_NUM_THREADS = "5"
$env:OPENBLAS_NUM_THREADS = "1"

@(
    $env:TMP,
    $env:HF_HOME,
    $env:TORCH_HOME,
    $env:XDG_CACHE_HOME,
    $env:PIP_CACHE_DIR,
    $env:UV_CACHE_DIR,
    $env:CARGO_HOME,
    $env:RUSTUP_HOME,
    $env:MANIALAB_V32_MODEL_CACHE
) | ForEach-Object {
    $path = [System.IO.Path]::GetFullPath($_)
    if (-not $path.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cache path must stay on D:: $path"
    }
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

$cDrive = Get-PSDrive -Name C -ErrorAction Stop
$cFreeGiB = $cDrive.Free / 1GB
if ($cFreeGiB -le $minimumCFreeGiB) {
    throw ("C: has {0:N2} GiB free, at or below the {1:N2} GiB stop floor." -f $cFreeGiB, $minimumCFreeGiB)
}
$dDrive = Get-PSDrive -Name D -ErrorAction Stop
$dFreeGiB = $dDrive.Free / 1GB
if ($dFreeGiB -lt $minimumFreeGiB) {
    throw ("D: has {0:N2} GiB free, below the {1:N2} GiB training reserve." -f $dFreeGiB, $minimumFreeGiB)
}

$manifestReviewStatus = & $python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('review_status', ''))" $datasetManifest
if ($LASTEXITCODE -ne 0) {
    throw "Could not parse dataset review_status from $datasetManifest."
}
$manifestReviewStatus = "$manifestReviewStatus".Trim()
if ($manifestReviewStatus -notin @("provisional", "human_reviewed")) {
    throw "Dataset review_status is missing or invalid."
}
if (-not $Smoke -and $manifestReviewStatus -ne "human_reviewed") {
    throw "The dataset is provisional. Finish zhyzx35607_review.csv and rebuild the final dataset before normal training."
}
if ($Smoke -and $manifestReviewStatus -ne "provisional") {
    throw "-Smoke is reserved for the provisional dataset_provisional."
}

$probe = & $python -c "import torch, accelerate, transformers, peft, torchaudio, pandas, pyarrow; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python is missing training dependencies; no packages were installed. $probe"
}

Push-Location $repoRoot
try {
    if ($Smoke) {
        $smokeRoot = Join-Path $kitRoot ("smoke\run-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        $common = @(
            "-m", "training.train_10k_lora",
            "--dataset-dir", $datasetDir,
            "--output-dir", $smokeRoot,
            "--model", $Model,
            "--device", "cpu",
            "--cpu-threads", "5",
            "--num-workers", "0",
            "--epochs", "3",
            "--gradient-accumulation-steps", "1",
            "--checkpoint-every-steps", "1",
            "--validation-batches", "1",
            "--validation-samples", "2",
            "--generation-window-seconds", "20",
            "--smoke"
        )
        & $python @common --max-steps 1
        if ($LASTEXITCODE -ne 0) {
            throw "CPU smoke cold-start step failed with exit code $LASTEXITCODE."
        }
        & $python @common --max-steps 2 --resume-latest
        if ($LASTEXITCODE -ne 0) {
            throw "CPU smoke resume step failed with exit code $LASTEXITCODE."
        }
    } else {
        $trainingRoot = Join-Path $kitRoot "training"
        $arguments = @(
            "-m", "training.train_10k_lora",
            "--dataset-dir", $datasetDir,
            "--output-dir", $trainingRoot,
            "--model", $Model,
            "--device", "cpu",
            "--cpu-threads", "5",
            "--num-workers", "$Workers",
            "--epochs", "$Epochs"
        )
        if ($Steps -gt 0) {
            $arguments += @("--max-steps", "$Steps")
        }
        if ($Resume) {
            $arguments += "--resume-latest"
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "CPU LoRA training failed with exit code $LASTEXITCODE."
        }
    }
} finally {
    Pop-Location
}

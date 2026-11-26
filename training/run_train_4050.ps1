[CmdletBinding()]
param(
    [ValidateSet("v32-mini", "v32")]
    [string]$Model = "v32-mini",
    [ValidateRange(1, 1000000)]
    [int]$Epochs = 3,
    [ValidateRange(0, 100000000)]
    [int]$Steps = 0,
    [switch]$Resume,
    [string]$TorosuCli
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$kitRoot = Join-Path $repoRoot "workspace_exp\lora_10k_kit"
$runtimeRoot = Join-Path $kitRoot "runtime"
$cacheRoot = Join-Path $runtimeRoot "cache"
$venvRoot = Join-Path $runtimeRoot "venv"
$pythonRoot = Join-Path $runtimeRoot "python"
$toolsRoot = Join-Path $runtimeRoot "tools"
$python = Join-Path $venvRoot "Scripts\python.exe"
$uv = Join-Path $toolsRoot "uv.exe"

$env:TMP = Join-Path $runtimeRoot "tmp"
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:HF_HOME = Join-Path $cacheRoot "huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:TORCH_HOME = Join-Path $cacheRoot "torch"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:TORCH_HOME "inductor"
$env:XDG_CACHE_HOME = Join-Path $cacheRoot "xdg"
$env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:UV_PYTHON_INSTALL_DIR = $pythonRoot
$env:MANIALAB_CACHE_ROOT = $cacheRoot
$env:MANIALAB_V32_MODEL_CACHE = Join-Path $cacheRoot "huggingface\manialab"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

@(
    $env:TMP,
    $env:HF_HOME,
    $env:TORCH_HOME,
    $env:XDG_CACHE_HOME,
    $env:PIP_CACHE_DIR,
    $env:UV_CACHE_DIR,
    $env:UV_PYTHON_INSTALL_DIR,
    $toolsRoot
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python310 = $false
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $installedPython = & $pyLauncher.Source -0p 2>$null
        if ($LASTEXITCODE -eq 0 -and $installedPython -match '3\.10') {
            & $pyLauncher.Source -3.10 -m venv $venvRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the training virtual environment with Python 3.10."
            }
            $python310 = Test-Path -LiteralPath $python -PathType Leaf
        }
    }
    if (-not $python310) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $version = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.10") {
                & $pythonCommand.Source -m venv $venvRoot
                if ($LASTEXITCODE -ne 0) {
                    throw "Could not create the training virtual environment."
                }
                $python310 = Test-Path -LiteralPath $python -PathType Leaf
            }
        }
    }
    if (-not $python310) {
        if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
            $downloadName = "uv-win-x64-$([guid]::NewGuid().ToString('N'))"
            $archive = Join-Path $env:TMP "$downloadName.zip"
            $staging = Join-Path $env:TMP $downloadName
            New-Item -ItemType Directory -Force -Path $staging | Out-Null
            try {
                Invoke-WebRequest `
                    -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" `
                    -OutFile $archive `
                    -UseBasicParsing
                Expand-Archive -LiteralPath $archive -DestinationPath $staging
                $uvFiles = @(Get-ChildItem -LiteralPath $staging -Filter uv.exe -File -Recurse)
                if ($uvFiles.Count -ne 1) {
                    throw "The official uv package did not contain exactly one uv.exe."
                }
                Copy-Item -LiteralPath $uvFiles[0].FullName -Destination $uv
            } finally {
                $stageFull = [System.IO.Path]::GetFullPath($staging)
                $tempFull = [System.IO.Path]::GetFullPath($env:TMP).TrimEnd('\') + '\'
                if ($stageFull.StartsWith($tempFull, [System.StringComparison]::OrdinalIgnoreCase)) {
                    Remove-Item -LiteralPath $stageFull -Recurse -Force -ErrorAction SilentlyContinue
                }
                Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
            }
        }
        & $uv python install 3.10 --install-dir $env:UV_PYTHON_INSTALL_DIR --cache-dir $env:UV_CACHE_DIR
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install the managed Python 3.10 runtime."
        }
        & $uv venv --managed-python --seed --python 3.10 $venvRoot
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw "Could not create the portable Python 3.10 training environment."
        }
    }
}

$needsCudaTorch = $true
if (Test-Path -LiteralPath $python -PathType Leaf) {
    $torchProbe = & $python -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $torchProbe -match "^2\.9\.1\+cu128\|True$") {
        $needsCudaTorch = $false
    }
}

if ($needsCudaTorch) {
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchaudio==2.9.1
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
}

& $python -m pip install -r (Join-Path $repoRoot "assets\mapperatorinator-requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Mapperatorinator training dependencies could not be installed."
}

$cudaProbe = & $python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0 -or $cudaProbe.Trim() -ne "1") {
    throw "CUDA is unavailable in this Python environment. Confirm the NVIDIA driver is installed."
}

$datasetManifest = Join-Path $kitRoot "dataset\manifest.json"
if (-not (Test-Path -LiteralPath $datasetManifest -PathType Leaf)) {
    throw "Dataset missing at $datasetManifest. Run python -m training.prepare_10k_dataset first."
}

if ($TorosuCli) {
    if (-not (Test-Path -LiteralPath $TorosuCli -PathType Leaf)) {
        throw "Torosu CLI does not exist: $TorosuCli"
    }
    $env:MANIALAB_TOROSU_CLI = (Resolve-Path -LiteralPath $TorosuCli).Path
} elseif (Test-Path -LiteralPath (Join-Path $repoRoot "engines\torosu_cli\torosu-cli.exe") -PathType Leaf) {
    $env:MANIALAB_TOROSU_CLI = Join-Path $repoRoot "engines\torosu_cli\torosu-cli.exe"
}

$arguments = @(
    "-m", "training.train_10k_lora",
    "--model", $Model,
    "--epochs", "$Epochs"
)
if ($Steps -gt 0) {
    $arguments += @("--max-steps", "$Steps")
}
if ($Resume) {
    $arguments += "--resume-latest"
}

Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "10K LoRA training failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

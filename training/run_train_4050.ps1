[CmdletBinding()]
param(
    [ValidateSet("v32-mini", "v32")]
    [string]$Model = "v32-mini",
    [ValidateRange(1, 1000000)]
    [int]$Epochs = 3,
    [ValidateRange(0, 100000000)]
    [int]$Steps = 0,
    [switch]$Resume,
    [string]$TorosuCli,
    [string]$PythonPath,
    [switch]$NoMirrors
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$kitRoot = if (Test-Path -LiteralPath (Join-Path $repoRoot "dataset\manifest.json") -PathType Leaf) {
    $repoRoot
} else {
    Join-Path $repoRoot "workspace_exp\lora_6_10k_kit"
}
$runtimeRoot = Join-Path $kitRoot "runtime"
$cacheRoot = Join-Path $runtimeRoot "cache"
$venvRoot = Join-Path $runtimeRoot "venv"
$pythonRoot = Join-Path $runtimeRoot "python"
$toolsRoot = Join-Path $runtimeRoot "tools"
$datasetManifest = Join-Path $kitRoot "dataset\manifest.json"
$python = Join-Path $venvRoot "Scripts\python.exe"
$uv = Join-Path $toolsRoot "uv.exe"
$minimumCFreeGiB = 5.0
$minimumKitFreeGiB = 4.0
$cDrive = Get-PSDrive -Name C -ErrorAction Stop
$cFreeGiB = $cDrive.Free / 1GB
if ($cFreeGiB -le $minimumCFreeGiB) {
    throw ("C: has {0:N2} GiB free, at or below the {1:N2} GiB stop floor." -f $cFreeGiB, $minimumCFreeGiB)
}
$kitDriveName = ([System.IO.Path]::GetPathRoot($kitRoot)).TrimEnd(':\')
$kitDrive = Get-PSDrive -Name $kitDriveName -ErrorAction Stop
$kitFreeGiB = $kitDrive.Free / 1GB
if ($kitFreeGiB -lt $minimumKitFreeGiB) {
    throw ("Training drive {0}: has {1:N2} GiB free, below the {2:N2} GiB training reserve." -f $kitDriveName, $kitFreeGiB, $minimumKitFreeGiB)
}

if (-not (Test-Path -LiteralPath $datasetManifest -PathType Leaf)) {
    throw "Reviewed 6–10K dataset manifest is missing at $datasetManifest."
}
Add-Type -AssemblyName System.Web.Extensions
$manifestSerializer = New-Object System.Web.Script.Serialization.JavaScriptSerializer
$manifestSerializer.MaxJsonLength = 100MB
$manifest = $manifestSerializer.DeserializeObject(
    [System.IO.File]::ReadAllText($datasetManifest, [System.Text.Encoding]::UTF8)
)
if ($manifest["review_status"] -ne "human_reviewed") {
    throw "The dataset is provisional. Complete zhyzx35607_review.csv and rebuild before GPU training."
}

$env:TMP = Join-Path $runtimeRoot "tmp"
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:HF_HOME = Join-Path $cacheRoot "huggingface"
$bundledWhisperCache = Join-Path $repoRoot "assets\huggingface"
$env:HF_HUB_CACHE = if (Test-Path -LiteralPath $bundledWhisperCache -PathType Container) { $bundledWhisperCache } else { Join-Path $env:HF_HOME "hub" }
$env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE
$env:TORCH_HOME = Join-Path $cacheRoot "torch"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:TORCH_HOME "inductor"
$env:XDG_CACHE_HOME = Join-Path $cacheRoot "xdg"
$env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
$env:UV_CACHE_DIR = Join-Path $cacheRoot "uv"
$env:UV_PYTHON_INSTALL_DIR = $pythonRoot
$env:MANIALAB_CACHE_ROOT = $cacheRoot
$localAppRoot = Join-Path $env:LOCALAPPDATA "ManiaLab"
$bundledModels = Join-Path $kitRoot "models"
$localModels = Join-Path $localAppRoot "models\mapperatorinator"
$env:MANIALAB_V32_MODEL_CACHE = if (Test-Path -LiteralPath $bundledModels -PathType Container) {
    $bundledModels
} elseif (Test-Path -LiteralPath $localModels -PathType Container) {
    $localModels
} else {
    Join-Path $cacheRoot "huggingface\manialab"
}
$env:MANIALAB_KIT_ROOT = $kitRoot
$bundledSource = Join-Path $kitRoot "source"
$localSource = Join-Path $localAppRoot "mapperatorinator-v32\source"
if (Test-Path -LiteralPath (Join-Path $bundledSource "inference.py") -PathType Leaf) {
    $env:MANIALAB_MAPPER_REPO = $bundledSource
} elseif (Test-Path -LiteralPath (Join-Path $localSource "inference.py") -PathType Leaf) {
    $env:MANIALAB_MAPPER_REPO = $localSource
}
$modelName = if ($Model -eq "v32-mini") { "Mapperatorinator-v32-mini-7807f0d" } else { "Mapperatorinator-v32" }
$modelCheckpoint = Join-Path $env:MANIALAB_V32_MODEL_CACHE "$modelName\gamemode=3"
$modelFiles = @("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")
$modelReady = @($modelFiles | ForEach-Object { Test-Path -LiteralPath (Join-Path $modelCheckpoint $_) -PathType Leaf } | Where-Object { -not $_ }).Count -eq 0
$whisperCandidates = @(
    $bundledWhisperCache,
    (Join-Path $localAppRoot "assets\huggingface"),
    (Join-Path $localAppRoot "cache\huggingface"),
    (Join-Path $localAppRoot "mapperatorinator-v32\cache\huggingface")
)
$whisperFiles = @(
    "models--openai--whisper-base\refs\main",
    "models--openai--whisper-base\snapshots\e37978b90ca9030d5170a5c07aadb050351a65bb\config.json",
    "models--openai--whisper-small\refs\main",
    "models--openai--whisper-small\snapshots\973afd24965f72e36ca33b3055d56a652f456b4d\config.json"
)
$whisperCache = $null
foreach ($candidate in $whisperCandidates) {
    if ((@($whisperFiles | ForEach-Object { Test-Path -LiteralPath (Join-Path $candidate $_) -PathType Leaf } | Where-Object { -not $_ }).Count -eq 0)) {
        $whisperCache = $candidate
        break
    }
}
if ($whisperCache) {
    $env:HF_HUB_CACHE = $whisperCache
    $env:TRANSFORMERS_CACHE = $whisperCache
    $env:MANIALAB_WHISPER_CONFIG_CACHE = $whisperCache
    $whisperReady = $true
} else {
    $whisperReady = $false
    Remove-Item Env:MANIALAB_WHISPER_CONFIG_CACHE -ErrorAction SilentlyContinue
}
if ($modelReady -and $whisperReady) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
} else {
    Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
    Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
}
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

$usingExistingCudaPython = $false
$knownCudaPython = Join-Path $env:LOCALAPPDATA "ManiaLab\mapperatorinator-v32\venv\Scripts\python.exe"
if ($PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "PythonPath does not exist: $PythonPath" }
    $python = (Resolve-Path -LiteralPath $PythonPath).Path
} elseif (Test-Path -LiteralPath $knownCudaPython -PathType Leaf) {
    $knownTorch = & $knownCudaPython -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $knownTorch -match "^2\.9\.1\+cu128\|True$") {
        $python = $knownCudaPython
        $usingExistingCudaPython = $true
    }
}

if (-not $usingExistingCudaPython -and -not (Test-Path -LiteralPath $python -PathType Leaf)) {
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

$needsCudaTorch = -not $usingExistingCudaPython
if (-not $usingExistingCudaPython -and (Test-Path -LiteralPath $python -PathType Leaf)) {
    $torchProbe = & $python -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $torchProbe -match "^2\.9\.1\+cu128\|True$") {
        $needsCudaTorch = $false
    }
}

if ($needsCudaTorch) {
    $torchIndexUrl = if ($NoMirrors) { "https://download.pytorch.org/whl/cu128" } elseif ($env:MANIALAB_TORCH_INDEX_URL) { $env:MANIALAB_TORCH_INDEX_URL } else { "https://mirror.nju.edu.cn/pytorch/whl/cu128/" }
    $env:MANIALAB_PYPI_INDEX_URL = if ($NoMirrors) { "https://pypi.org/simple" } elseif ($env:MANIALAB_PYPI_INDEX_URL) { $env:MANIALAB_PYPI_INDEX_URL } else { "https://mirrors.aliyun.com/pypi/simple/" }
    & $python -m pip install --index-url $env:MANIALAB_PYPI_INDEX_URL --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $python -m pip install --index-url $torchIndexUrl torch==2.9.1 torchaudio==2.9.1
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
}

$requirements = Join-Path $repoRoot "assets\mapperatorinator-requirements.txt"
if ($usingExistingCudaPython) {
    $lightDependencies = @("accelerate==1.12.0", "pandas==2.3.3", "pyarrow==22.0.0", "peft==0.18.1", "datasets==4.8.3", "safetensors")
    if (-not $NoMirrors) {
        $env:MANIALAB_PYPI_INDEX_URL = if ($env:MANIALAB_PYPI_INDEX_URL) { $env:MANIALAB_PYPI_INDEX_URL } else { "https://mirrors.aliyun.com/pypi/simple/" }
        & $python -m pip install --index-url $env:MANIALAB_PYPI_INDEX_URL @lightDependencies
    } else {
        & $python -m pip install @lightDependencies
    }
} else {
    if (-not $NoMirrors) {
        $env:MANIALAB_PYPI_INDEX_URL = if ($env:MANIALAB_PYPI_INDEX_URL) { $env:MANIALAB_PYPI_INDEX_URL } else { "https://mirrors.aliyun.com/pypi/simple/" }
        & $python -m pip install --index-url $env:MANIALAB_PYPI_INDEX_URL -r $requirements
    } else {
        & $python -m pip install -r $requirements
    }
}
if ($LASTEXITCODE -ne 0) {
    throw "Mapperatorinator training dependencies could not be installed."
}

$cudaProbe = & $python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($LASTEXITCODE -ne 0 -or $cudaProbe.Trim() -ne "1") {
    throw "CUDA is unavailable in this Python environment. Confirm the NVIDIA driver is installed."
}

if ($TorosuCli) {
    if (-not (Test-Path -LiteralPath $TorosuCli -PathType Leaf)) {
        throw "Torosu CLI does not exist: $TorosuCli"
    }
    $env:MANIALAB_TOROSU_CLI = (Resolve-Path -LiteralPath $TorosuCli).Path
} elseif (Test-Path -LiteralPath (Join-Path $kitRoot "engines\torosu_cli\torosu-cli.exe") -PathType Leaf) {
    $env:MANIALAB_TOROSU_CLI = Join-Path $kitRoot "engines\torosu_cli\torosu-cli.exe"
} elseif (Test-Path -LiteralPath (Join-Path $localAppRoot "engines\torosu_cli\torosu-cli.exe") -PathType Leaf) {
    $env:MANIALAB_TOROSU_CLI = Join-Path $localAppRoot "engines\torosu_cli\torosu-cli.exe"
} elseif (Test-Path -LiteralPath (Join-Path $repoRoot "engines\torosu_cli\torosu-cli.exe") -PathType Leaf) {
    $env:MANIALAB_TOROSU_CLI = Join-Path $repoRoot "engines\torosu_cli\torosu-cli.exe"
}

$arguments = @(
    "-m", "training.train_10k_lora",
    "--model", $Model,
    "--epochs", "$Epochs",
    "--device", "cuda",
    "--gradient-accumulation-steps", "16"
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

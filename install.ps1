$ErrorActionPreference = "Stop"

function Test-IsWindowsProcess {
    return [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
}

function Normalize-ProcessPathEnvironment {
    $PathEntries = @(Get-ChildItem Env: | Where-Object { $_.Name -ieq "Path" })
    if ($PathEntries.Count -eq 0) {
        return
    }

    $Preferred = $PathEntries | Where-Object { $_.Name -ceq "Path" } | Select-Object -First 1
    if (-not $Preferred) {
        $Preferred = $PathEntries[0]
    }
    $PathValue = $Preferred.Value

    foreach ($Entry in $PathEntries) {
        Remove-Item -LiteralPath ("Env:" + $Entry.Name) -ErrorAction SilentlyContinue
    }
    Remove-Item Env:PATH -ErrorAction SilentlyContinue
    $env:Path = $PathValue
}

function Invoke-SelectedPython {
    param(
        [string]$PythonCommand,
        [string[]]$PythonArgs,
        [string[]]$Arguments
    )
    & $PythonCommand @PythonArgs @Arguments
}

function Test-CompatiblePython {
    param(
        [string]$PythonCommand,
        [string[]]$PythonArgs = @()
    )
    Invoke-SelectedPython $PythonCommand $PythonArgs @("-c", "import sys, sysconfig; text = ' '.join(str(x) for x in (sys.version, sys.executable, sysconfig.get_platform())).lower(); sys.exit(1 if ('msys' in text or 'mingw' in text or 'ucrt64' in text) else 0)") 2>$null
    return $LASTEXITCODE -eq 0
}

function Select-CompatiblePython {
    if ($env:PYTHON_BIN) {
        return @{ Command = $env:PYTHON_BIN; Args = @(); Display = $env:PYTHON_BIN }
    }

    if (Test-IsWindowsProcess) {
        $Candidates = @(
            @{ Command = "py"; Args = @("-3.12"); Display = "py -3.12" },
            @{ Command = "py"; Args = @("-3.11"); Display = "py -3.11" },
            @{ Command = "python"; Args = @(); Display = "python" }
        )
        foreach ($Candidate in $Candidates) {
            if (-not (Get-Command $Candidate.Command -ErrorAction SilentlyContinue)) {
                continue
            }
            if (Test-CompatiblePython $Candidate.Command $Candidate.Args) {
                return $Candidate
            }
        }
        return @{ Command = "python"; Args = @(); Display = "python" }
    }

    return @{ Command = "python"; Args = @(); Display = "python" }
}

function Assert-CompatiblePython {
    param(
        [string]$PythonCommand,
        [string[]]$PythonArgs = @()
    )
    Invoke-SelectedPython $PythonCommand $PythonArgs @("-c", "import sys, sysconfig; print(sys.executable); print(sys.version.replace('\n', ' ')); print(sysconfig.get_platform())")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if (-not (Test-CompatiblePython $PythonCommand $PythonArgs)) {
        Write-Error "MSYS2/Mingw Python is not supported for Fusion Memory's local Qwen runtime because PyTorch wheels are not available for that Python ABI. Install official Windows CPython or conda Python 3.11/3.12, then rerun with `$env:PYTHON_BIN set to that python.exe."
        exit 1
    }
}

if (Test-IsWindowsProcess) {
    Normalize-ProcessPathEnvironment
}

$PythonSelection = Select-CompatiblePython
$Python = $PythonSelection.Command
$PythonArgs = $PythonSelection.Args
Write-Host "Using Python: $($PythonSelection.Display)"
Assert-CompatiblePython $Python $PythonArgs

Invoke-SelectedPython $Python $PythonArgs @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'Python 3.11+ is required.')")

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Invoke-SelectedPython $Python $PythonArgs @("-m", "pip", "install", "--upgrade", "pip")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Invoke-SelectedPython $Python $PythonArgs @("-m", "pip", "install", "-e", "$ScriptDir")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Invoke-SelectedPython $Python $PythonArgs @("-m", "pip", "install", "-e", "$ScriptDir[postgres,qwen]")
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Optional Postgres/Qwen dependencies could not be installed. Continuing with install-check; installation will fail until required Qwen dependencies are available."
}
if ($env:FUSION_MEMORY_USE_WIZARD -eq "1") {
    Invoke-SelectedPython $Python $PythonArgs @("-m", "fusion_memory.cli", "init", "--wizard")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} elseif ($env:FUSION_MEMORY_SKIP_WIZARD -eq "1") {
    Invoke-SelectedPython $Python $PythonArgs @("-m", "fusion_memory.cli", "install-check", "--force")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    Invoke-SelectedPython $Python $PythonArgs @("-m", "fusion_memory.cli", "install-check", "--force")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
Invoke-SelectedPython $Python $PythonArgs @("-m", "fusion_memory.cli", "doctor")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Fusion Memory is installed."
Write-Host "Bundled model paths: $ScriptDir\models\Qwen3-Embedding-0.6B and $ScriptDir\models\Qwen3-Reranker-0.6B"
Write-Host "The installer tries to install full runtime dependencies including Postgres and local Qwen model support."
Write-Host "If the installer reported compromised mode, this machine could not run the bundled models; set DASHSCOPE_API_KEY for the recommended Aliyun API path."
Write-Host "Start it with: fusion-memory start"
Write-Host "Check it with: fusion-memory status"

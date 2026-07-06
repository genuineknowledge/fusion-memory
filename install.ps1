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
    Invoke-SelectedPython $PythonCommand $PythonArgs @("-c", "import sys, sysconfig; text = ' '.join(str(x) for x in (sys.version, sys.executable, sysconfig.get_platform())).lower(); compatible = sys.version_info[:2] in ((3, 11), (3, 12)) and not any(token in text for token in ('msys', 'mingw', 'ucrt64')); sys.exit(0 if compatible else 1)") 2>$null
    return $LASTEXITCODE -eq 0
}

function Select-CompatiblePython {
    if ($env:PYTHON_BIN) {
        if (Test-CompatiblePython $env:PYTHON_BIN @()) {
            return @{ Command = $env:PYTHON_BIN; Args = @(); Display = $env:PYTHON_BIN }
        }
        Write-Warning "Ignoring incompatible PYTHON_BIN; looking for CPython 3.11/3.12."
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
    Invoke-SelectedPython $PythonCommand $PythonArgs @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if (-not (Test-CompatiblePython $PythonCommand $PythonArgs)) {
        Write-Error "Current Python is not compatible with Fusion Memory local Qwen on Windows. Install official CPython or conda Python 3.11/3.12, then rerun .\install.ps1 or set `$env:PYTHON_BIN to that python.exe."
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
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir ".fusion-memory-venv"
$LogDir = Join-Path $ScriptDir ".fusion-memory-logs"

# Qwen and native runtime dependencies are installed in the dedicated memory venv with --only-binary=:all:.
$OldPythonPath = $env:PYTHONPATH
try {
    if ($OldPythonPath) {
        $env:PYTHONPATH = "$ScriptDir$([System.IO.Path]::PathSeparator)$OldPythonPath"
    } else {
        $env:PYTHONPATH = $ScriptDir
    }

    $InstallerArgs = @(
        "-m",
        "fusion_memory.windows_installer",
        "--python-command",
        $Python,
        "--script-dir",
        $ScriptDir,
        "--venv-dir",
        $VenvDir,
        "--log-dir",
        $LogDir
    )
    foreach ($Arg in $PythonArgs) {
        $InstallerArgs += @("--python-arg", $Arg)
    }
    Invoke-SelectedPython $Python $PythonArgs $InstallerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    if ($null -eq $OldPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $OldPythonPath
    }
}

Write-Host ""
Write-Host "Fusion Memory is installed."
Write-Host "Bundled model paths: $ScriptDir\models\Qwen3-Embedding-0.6B and $ScriptDir\models\Qwen3-Reranker-0.6B"
Write-Host "Start it with: $VenvDir\Scripts\fusion-memory.exe start"
Write-Host "Check it with: $VenvDir\Scripts\fusion-memory.exe status"

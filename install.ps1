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

if (Test-IsWindowsProcess) {
    Normalize-ProcessPathEnvironment
}

$Python = $env:PYTHON_BIN
if (-not $Python) {
    $Python = "python"
}

& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'Python 3.11+ is required.')"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $Python -m pip install -e "$ScriptDir"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $Python -m pip install -e "$ScriptDir[postgres,qwen]"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Optional Postgres/Qwen dependencies could not be installed. Continuing with install-check; installation will fail until required Qwen dependencies are available."
}
if ($env:FUSION_MEMORY_USE_WIZARD -eq "1") {
    & $Python -m fusion_memory.cli init --wizard
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} elseif ($env:FUSION_MEMORY_SKIP_WIZARD -eq "1") {
    & $Python -m fusion_memory.cli install-check --force
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    & $Python -m fusion_memory.cli install-check --force
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
& $Python -m fusion_memory.cli doctor
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

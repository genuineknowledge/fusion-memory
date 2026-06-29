$ErrorActionPreference = "Stop"

$Python = $env:PYTHON_BIN
if (-not $Python) {
    $Python = "python"
}

& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'Python 3.11+ is required.')"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python -m pip install --upgrade pip
& $Python -m pip install -e "$ScriptDir[postgres,qwen]"
if ($env:FUSION_MEMORY_USE_WIZARD -eq "1") {
    & $Python -m fusion_memory.cli init --wizard
} elseif ($env:FUSION_MEMORY_SKIP_WIZARD -eq "1") {
    & $Python -m fusion_memory.cli install-check --force
} else {
    & $Python -m fusion_memory.cli install-check --force
}
& $Python -m fusion_memory.cli doctor

Write-Host ""
Write-Host "Fusion Memory is installed."
Write-Host "Bundled model paths: $ScriptDir\models\Qwen3-Embedding-0.6B and $ScriptDir\models\Qwen3-Reranker-0.6B"
Write-Host "The installer installs full runtime dependencies including Postgres and local Qwen model support."
Write-Host "If the installer reported compromised mode, this machine could not run the bundled models; set DASHSCOPE_API_KEY for the recommended Aliyun API path."
Write-Host "Start it with: fusion-memory start"
Write-Host "Check it with: fusion-memory status"

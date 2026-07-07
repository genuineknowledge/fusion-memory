$ErrorActionPreference = "Stop"

function Normalize-ProcessPathEnvironment {
    $ProcessEnvironment = [Environment]::GetEnvironmentVariables("Process")
    $PathEntries = @($ProcessEnvironment.Keys | Where-Object { $_ -ieq "Path" })
    if ($PathEntries.Count -eq 0) {
        return
    }

    $Preferred = $PathEntries | Where-Object { $_ -ceq "Path" } | Select-Object -First 1
    if (-not $Preferred) {
        $Preferred = $PathEntries[0]
    }
    $PathValue = [string]$ProcessEnvironment[$Preferred]

    foreach ($Entry in $PathEntries) {
        [Environment]::SetEnvironmentVariable([string]$Entry, $null, "Process")
        Remove-Item -LiteralPath ("Env:" + $Entry) -ErrorAction SilentlyContinue
    }
    Remove-Item Env:PATH -ErrorAction SilentlyContinue
    [Environment]::SetEnvironmentVariable("Path", $PathValue, "Process")
}

function Get-UvDownloadUrl {
    $Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
    if ($Arch -eq "arm64") {
        return "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-pc-windows-msvc.zip"
    }
    return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
}

function Resolve-Uv {
    param(
        [string]$ScriptDir,
        [string]$ToolsDir,
        [string]$LogFile
    )

    if ($env:FUSION_MEMORY_UV_BIN -and (Test-Path -LiteralPath $env:FUSION_MEMORY_UV_BIN)) {
        return $env:FUSION_MEMORY_UV_BIN
    }

    $Found = Get-Command uv -ErrorAction SilentlyContinue
    if ($Found) {
        return [string]$Found.Source
    }

    $LocalCandidates = @(
        (Join-Path $ToolsDir "uv.exe"),
        (Join-Path $ScriptDir ".venv\Scripts\uv.exe"),
        (Join-Path $ScriptDir "venv\Scripts\uv.exe"),
        (Join-Path (Split-Path -Parent $ScriptDir) ".venv\Scripts\uv.exe"),
        (Join-Path (Split-Path -Parent $ScriptDir) "venv\Scripts\uv.exe"),
        (Join-Path (Get-Location).Path ".venv\Scripts\uv.exe"),
        (Join-Path (Get-Location).Path "venv\Scripts\uv.exe")
    )
    if ($env:VIRTUAL_ENV) {
        $LocalCandidates += (Join-Path $env:VIRTUAL_ENV "Scripts\uv.exe")
    }
    foreach ($Candidate in $LocalCandidates) {
        if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
            Add-Content -Path $LogFile -Value "Using local uv at $Candidate"
            return $Candidate
        }
    }

    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $Archive = Join-Path $ToolsDir "uv.zip"
    $Uv = Join-Path $ToolsDir "uv.exe"
    $Url = Get-UvDownloadUrl
    Add-Content -Path $LogFile -Value "Downloading uv from $Url"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Archive -UseBasicParsing
        Expand-Archive -LiteralPath $Archive -DestinationPath $ToolsDir -Force
        $Candidate = Get-ChildItem -Path $ToolsDir -Recurse -Filter "uv.exe" | Select-Object -First 1
        if (-not $Candidate) {
            throw "uv.exe was not found in the downloaded archive."
        }
        Copy-Item -LiteralPath $Candidate.FullName -Destination $Uv -Force
        return $Uv
    } catch {
        Add-Content -Path $LogFile -Value "uv bootstrap failed: $_"
        throw
    }
}

function Test-CompatiblePython {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string]$LogFile
    )

    $Probe = "import sys, sysconfig; text=' '.join(str(x) for x in (sys.version, sys.executable, sysconfig.get_platform())).lower(); compatible=sys.version_info[:2] in ((3, 11), (3, 12)) and not any(token in text for token in ('msys', 'mingw', 'ucrt64')); print(sys.executable if compatible else ''); sys.exit(0 if compatible else 1)"
    try {
        $Output = & $Command @Arguments "-c" $Probe 2>> $LogFile
        if ($LASTEXITCODE -eq 0 -and $Output) {
            $Python = [string]($Output | Select-Object -Last 1)
            if ($Python -and (Test-Path -LiteralPath $Python)) {
                Add-Content -Path $LogFile -Value "Using compatible Python at $Python"
                return $Python
            }
        }
    } catch {
        Add-Content -Path $LogFile -Value "Python compatibility probe failed for $Command $($Arguments -join ' '): $_"
    }
    return $null
}

function Resolve-CompatiblePython {
    param(
        [string]$LogFile
    )

    $Candidates = @()
    if ($env:FUSION_MEMORY_PYTHON_BIN) {
        $Candidates += [pscustomobject]@{ Command = $env:FUSION_MEMORY_PYTHON_BIN; Arguments = @() }
    }
    $Candidates += [pscustomobject]@{ Command = "py"; Arguments = @("-3.12") }
    $Candidates += [pscustomobject]@{ Command = "py"; Arguments = @("-3.11") }
    $LocalRoots = @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    foreach ($Root in $LocalRoots) {
        foreach ($Version in @("Python312", "Python311")) {
            $Candidates += [pscustomobject]@{ Command = (Join-Path $Root "Programs\Python\$Version\python.exe"); Arguments = @() }
            $Candidates += [pscustomobject]@{ Command = (Join-Path $Root "$Version\python.exe"); Arguments = @() }
        }
    }
    foreach ($Name in @("python.exe", "python3.exe")) {
        $Candidates += [pscustomobject]@{ Command = $Name; Arguments = @() }
    }

    foreach ($Candidate in $Candidates) {
        $Command = [string]$Candidate.Command
        $Arguments = [string[]]$Candidate.Arguments
        if (-not $Command) {
            continue
        }
        if ($Command -match '[\\/]' -and -not (Test-Path -LiteralPath $Command)) {
            continue
        }
        if ($Command -notmatch '[\\/]') {
            $Resolved = Get-Command $Command -ErrorAction SilentlyContinue
            if (-not $Resolved) {
                continue
            }
            if ([string]$Resolved.Source -match "\\WindowsApps\\") {
                continue
            }
            $Command = [string]$Resolved.Source
        }
        $Python = Test-CompatiblePython -Command $Command -Arguments $Arguments -LogFile $LogFile
        if ($Python) {
            return $Python
        }
    }
    return $null
}

function New-ToolInstallArgs {
    param(
        [string]$Python,
        [bool]$ManagedPython
    )

    $Args = @(
        "tool", "install",
        "--force",
        "--python", $Python
    )
    if ($ManagedPython) {
        $Args += "--managed-python"
    }
    $Args += @(
        "--no-progress",
        "--with", "modelscope-hub>=0.1.6",
        "--with", "psycopg2-binary>=2.9",
        "--with", "torch>=2.5",
        "--with", "transformers>=4.51",
        "--with", "sentence-transformers>=3.4",
        "--with", "safetensors",
        "--with", "tokenizers",
        "--with", "hf-xet",
        "--with", "click",
        "--with", "typer",
        "--no-build-package", "psycopg2-binary",
        "--no-build-package", "torch",
        "--no-build-package", "transformers",
        "--no-build-package", "sentence-transformers",
        "--no-build-package", "safetensors",
        "--no-build-package", "tokenizers",
        "--no-build-package", "hf-xet",
        $Package
    )
    return $Args
}

function Invoke-StepResult {
    param(
        [string]$Name,
        [string]$LogFile,
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host "$Name..."
    Add-Content -Path $LogFile -Value ""
    Add-Content -Path $LogFile -Value "=== $Name ==="
    Add-Content -Path $LogFile -Value ($Command + " " + ($Arguments -join " "))
    & $Command @Arguments *>> $LogFile
    return ($LASTEXITCODE -eq 0)
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$LogFile,
        [string]$Command,
        [string[]]$Arguments
    )

    $Ok = Invoke-StepResult -Name $Name -LogFile $LogFile -Command $Command -Arguments $Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Fusion Memory installation needs attention. Step: $Name. Log: $LogFile"
        exit $LASTEXITCODE
    }
    if (-not $Ok) {
        Write-Error "Fusion Memory installation needs attention. Step: $Name. Log: $LogFile"
        exit 1
    }
}

Normalize-ProcessPathEnvironment

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir ".fusion-memory-logs"
$LogFile = Join-Path $LogDir "install.log"
$ToolsDir = Join-Path $ScriptDir ".fusion-memory-tools"
$Package = if ($env:FUSION_MEMORY_PACKAGE) { $env:FUSION_MEMORY_PACKAGE } else { $ScriptDir }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Content -Path $LogFile -Value ""

$Uv = Resolve-Uv -ScriptDir $ScriptDir -ToolsDir $ToolsDir -LogFile $LogFile
& $Uv --version *>> $LogFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Fusion Memory installation needs attention. Step: uv bootstrap. Log: $LogFile"
    exit $LASTEXITCODE
}

$CompatiblePython = Resolve-CompatiblePython -LogFile $LogFile
if ($CompatiblePython) {
    $ToolInstallArgs = New-ToolInstallArgs -Python $CompatiblePython -ManagedPython $false
    $ToolInstallOk = Invoke-StepResult -Name "fusion memory tool install" -LogFile $LogFile -Command $Uv -Arguments $ToolInstallArgs
    if (-not $ToolInstallOk) {
        Add-Content -Path $LogFile -Value "Compatible Windows CPython tool install failed; retrying with uv-managed Python 3.12."
        $ToolInstallArgs = New-ToolInstallArgs -Python "3.12" -ManagedPython $true
        Invoke-Step -Name "fusion memory tool install fallback" -LogFile $LogFile -Command $Uv -Arguments $ToolInstallArgs
    }
} else {
    Add-Content -Path $LogFile -Value "No compatible Windows CPython runtime found; using uv-managed Python 3.12."
    $ToolInstallArgs = New-ToolInstallArgs -Python "3.12" -ManagedPython $true
    Invoke-Step -Name "fusion memory tool install fallback" -LogFile $LogFile -Command $Uv -Arguments $ToolInstallArgs
}

$ToolBinDir = & $Uv tool dir --bin
if ($LASTEXITCODE -ne 0) {
    Write-Error "Fusion Memory installation needs attention. Step: uv tool dir --bin. Log: $LogFile"
    exit $LASTEXITCODE
}
$FusionMemory = Join-Path $ToolBinDir "fusion-memory.exe"

Invoke-Step -Name "local qwen models" -LogFile $LogFile -Command $FusionMemory -Arguments @("download-models", "--json")
if ($env:FUSION_MEMORY_USE_WIZARD -eq "1") {
    Invoke-Step -Name "wizard" -LogFile $LogFile -Command $FusionMemory -Arguments @("init", "--wizard")
} else {
    Invoke-Step -Name "install readiness" -LogFile $LogFile -Command $FusionMemory -Arguments @("install-check", "--force")
}
Invoke-Step -Name "doctor" -LogFile $LogFile -Command $FusionMemory -Arguments @("doctor")

Write-Host ""
Write-Host "Fusion Memory is installed."
Write-Host "Log: $LogFile"
Write-Host "Start it with: fusion-memory start"
Write-Host "Check it with: fusion-memory status"

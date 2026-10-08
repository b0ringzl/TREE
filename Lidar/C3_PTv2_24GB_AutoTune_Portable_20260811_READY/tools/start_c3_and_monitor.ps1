$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "runtime\env\python.exe"
$Summary = Join-Path $Root "output\c3_summary.json"
$Runner = Join-Path $PSScriptRoot "run_c3.ps1"
$KeepAwake = Join-Path $PSScriptRoot "keep_awake.ps1"
$PauseRequest = Join-Path $Root "output\pause_request.json"

function Start-KeepAwake {
    param(
        [int]$OwnerPid,
        [string]$Mode
    )
    $ExistingGuard = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.CommandLine -and
        $_.CommandLine.Contains("keep_awake.ps1") -and
        $_.CommandLine.Contains(("-OwnerPid {0}" -f $OwnerPid))
    }
    if (-not $ExistingGuard) {
        Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList @(
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", "`"$KeepAwake`"",
                "-OwnerPid", [string]$OwnerPid,
                "-Root", "`"$Root`"",
                "-Mode", $Mode
            ) `
            -WindowStyle Hidden | Out-Null
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "output") | Out-Null
Start-KeepAwake -OwnerPid $PID -Mode "setup"
Write-Host "Keep-awake guard enabled for setup, tuning, and device checks."

& (Join-Path $PSScriptRoot "prepare_runtime.ps1")

$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.CommandLine -and
    $_.CommandLine.Contains($Root) -and
    (
        $_.CommandLine.Contains("run_c3.ps1") -or
        $_.CommandLine.Contains("train_b5_ptv2.py")
    )
}

$AllComplete = $false
if (Test-Path -LiteralPath $Summary) {
    try {
        $SummaryState = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
        $AllComplete = $SummaryState.status -eq "complete"
    } catch {
        $AllComplete = $false
    }
}

$RunnerPid = 0
if (-not $Existing -and -not $AllComplete) {
    if (Test-Path -LiteralPath $PauseRequest) {
        try {
            $PauseState = Get-Content -LiteralPath $PauseRequest -Raw | ConvertFrom-Json
            if ($PauseState.status -in @("requested", "complete")) {
                $PauseState.status = "cleared"
                $PauseState | Add-Member `
                    -NotePropertyName cleared_at `
                    -NotePropertyValue ([DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")) `
                    -Force
                $PauseState.message = "Pause request cleared for resume"
                $PauseState | ConvertTo-Json | Set-Content -LiteralPath $PauseRequest -Encoding UTF8
            }
        } catch {
            throw "Pause request could not be cleared for resume: $($_.Exception.Message)"
        }
    }
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$Runner`""
        ) `
        -WindowStyle Hidden `
        -PassThru
    $RunnerPid = $Process.Id
    Write-Host "C3 runner launched. Wrapper PID: $RunnerPid"
} elseif ($Existing) {
    $RunnerProcess = $Existing | Where-Object {
        $_.CommandLine.Contains("run_c3.ps1")
    } | Select-Object -First 1
    if ($RunnerProcess) {
        $RunnerPid = [int]$RunnerProcess.ProcessId
    }
    Write-Host "C3 training is already running; opening the monitor."
} else {
    Write-Host "All C3 seeds are complete; opening the final result."
}

if ($RunnerPid -gt 0) {
    Start-KeepAwake -OwnerPid $RunnerPid -Mode "training"
    Write-Host "Keep-awake guard attached to C3 runner PID $RunnerPid."
}

& $Python (Join-Path $Root "app\scripts\monitor_c3.py") --root $Root

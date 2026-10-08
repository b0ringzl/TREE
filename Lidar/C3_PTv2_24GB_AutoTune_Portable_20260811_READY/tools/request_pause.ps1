$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $Root "output"
$LauncherPath = Join-Path $Output "c3_launcher_state.json"
$RequestPath = Join-Path $Output "pause_request.json"

$Processes = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.CommandLine -and
    $_.CommandLine.Contains($Root) -and
    (
        $_.CommandLine.Contains("run_c3.ps1") -or
        $_.CommandLine.Contains("train_b5_ptv2.py")
    )
})
if (-not $Processes) {
    Write-Host "C3 training is not running. Nothing needs to be paused."
    exit 0
}
if (-not (Test-Path -LiteralPath $LauncherPath)) {
    throw "C3 launcher state is missing."
}

$Launcher = Get-Content -LiteralPath $LauncherPath -Raw | ConvertFrom-Json
$Seed = [int]$Launcher.active_seed
$Run = Join-Path $Output ("runs\seed_{0}" -f $Seed)
$LivePath = Join-Path $Run "live_progress.json"
$CurrentEpoch = 0
if (Test-Path -LiteralPath $LivePath) {
    try {
        $Live = Get-Content -LiteralPath $LivePath -Raw | ConvertFrom-Json
        $CurrentEpoch = [int]$Live.epoch
    } catch {
        $CurrentEpoch = 0
    }
}

$ExistingRequest = $null
if (Test-Path -LiteralPath $RequestPath) {
    try {
        $ExistingRequest = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json
    } catch {
        $ExistingRequest = $null
    }
}
if (-not $ExistingRequest -or $ExistingRequest.status -ne "requested") {
    @{
        status = "requested"
        requested_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
        seed = $Seed
        target_epoch = $CurrentEpoch
        message = "Pause after the current epoch, validation, and checkpoint save"
    } | ConvertTo-Json | Set-Content -LiteralPath $RequestPath -Encoding UTF8
}

Write-Host "Safe pause requested for seed $Seed after epoch $CurrentEpoch."
Write-Host "The current epoch, validation, and last.pt save will finish first."
Write-Host "This window may remain open; closing it does not cancel the request."

while ($true) {
    Start-Sleep -Seconds 3
    try {
        $Request = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json
        if ($Request.status -eq "complete") {
            Write-Host "Pause complete after epoch $($Request.completed_epoch)."
            exit 0
        }
    } catch {
    }
    $StillRunning = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine.Contains($Root) -and
        (
            $_.CommandLine.Contains("run_c3.ps1") -or
            $_.CommandLine.Contains("train_b5_ptv2.py")
        )
    })
    if (-not $StillRunning) {
        throw "Training stopped before the safe pause request was confirmed."
    }
}

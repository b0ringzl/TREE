$ErrorActionPreference = "Stop"

$Sandbox = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $Sandbox)
$Runner = Join-Path $PSScriptRoot "run_c2_ptv2.ps1"
$RunDir = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\c2_ptv2\run_batch16_seed20260731"
$FinalMetrics = Join-Path $RunDir "final_metrics.json"
$LastCheckpoint = Join-Path $RunDir "last.pt"

$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @("powershell.exe", "python.exe") -and
    $_.CommandLine -and
    (
        $_.CommandLine.Contains("run_c2_ptv2.ps1") -or
        (
            $_.CommandLine.Contains("train_b5_ptv2.py") -and
            $_.CommandLine.Contains("run_batch16_seed20260731")
        )
    )
}
if ($Existing) {
    throw "C2 training is already running (PID: $($Existing.ProcessId -join ', '))"
}
if (Test-Path -LiteralPath $FinalMetrics) {
    $Final = Get-Content -LiteralPath $FinalMetrics -Raw | ConvertFrom-Json
    if ($Final.status -eq "complete") {
        throw "C2 training is already complete: $FinalMetrics"
    }
}

$Resume = Test-Path -LiteralPath $LastCheckpoint
$Arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Runner`""
)
if ($Resume) {
    $Arguments += "-Resume"
}
$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $Arguments `
    -WindowStyle Hidden `
    -PassThru

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
@{
    status = "launched"
    launched_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    wrapper_pid = $Process.Id
    resume = $Resume
    run_dir = $RunDir
} | ConvertTo-Json -Depth 3 | Set-Content `
    -LiteralPath (Join-Path $RunDir "launch_request.json") `
    -Encoding UTF8

[pscustomobject]@{
    Status = "launched"
    PID = $Process.Id
    Resume = $Resume
    RunDir = $RunDir
}

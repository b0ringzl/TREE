$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $Root "output"
$Active = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.CommandLine -and
    $_.CommandLine.Contains($Root) -and
    (
        $_.CommandLine.Contains("run_c3.ps1") -or
        $_.CommandLine.Contains("train_b5_ptv2.py")
    )
}
if ($Active) {
    throw "C3 training is still running. Package results after it stops."
}
if (-not (Test-Path -LiteralPath $Output)) {
    throw "No C3 output folder was found."
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Destination = Join-Path $Root ("C3_RESULTS_{0}.zip" -f $Timestamp)
$Items = @(
    $Output,
    (Join-Path $Root "diagnostics"),
    (Join-Path $Root "c3_config.json"),
    (Join-Path $Root "c3_config.template.json"),
    (Join-Path $Root "transfer_manifest.json"),
    (Join-Path $Root "使用说明.md")
) | Where-Object { Test-Path -LiteralPath $_ }

Compress-Archive -LiteralPath $Items -DestinationPath $Destination -CompressionLevel Optimal
Write-Host "Result package created:"
Write-Host $Destination

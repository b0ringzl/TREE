$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$SandboxRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$PythonPath = Join-Path $ProjectRoot 'envs\whu-tscmdl\python.exe'
$ServerPath = Join-Path $SandboxRoot 'scripts\d2f_point_quality_review_server.py'
$ReviewRoot = Join-Path $ProjectRoot 'lidar data\whu\derived\tscmdl\d2_exposure_stratified_evaluation\20260821_real_point_quality_review_v1'
$Port = 8770
$Url = "http://127.0.0.1:$Port/"
$RuntimePath = Join-Path $ReviewRoot 'server_runtime.json'
$StdoutPath = Join-Path $ReviewRoot 'server_stdout.log'
$StderrPath = Join-Path $ReviewRoot 'server_stderr.log'

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment not found: $PythonPath"
}

$Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $Listener) {
    $Arguments = @(
        $ServerPath,
        '--port', [string]$Port,
        '--no-browser'
    )
    $Process = Start-Process -FilePath $PythonPath -ArgumentList $Arguments -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath -PassThru
    $Deadline = (Get-Date).AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 250
        $Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Process.HasExited) {
            $ErrorText = if (Test-Path -LiteralPath $StderrPath) { Get-Content -LiteralPath $StderrPath -Raw } else { '' }
            throw "D2f server exited during startup. $ErrorText"
        }
    } while ($null -eq $Listener -and (Get-Date) -lt $Deadline)
    if ($null -eq $Listener) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw 'D2f server did not start within 15 seconds.'
    }
    [ordered]@{
        started_at = (Get-Date).ToString('o')
        pid = $Process.Id
        port = $Port
        url = $Url
        review_root = $ReviewRoot
    } | ConvertTo-Json | Set-Content -LiteralPath $RuntimePath -Encoding utf8
}

Start-Process $Url
Write-Host "D2f point-quality review is ready: $Url"

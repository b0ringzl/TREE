param(
    [int]$BackendPort = 8010,
    [int]$FrontendPort = 5175
)

$ErrorActionPreference = "Stop"

$Root = "D:\TREE"
$Project = Join-Path $Root "hk-tree-labeler-labelpaw-hover-sandbox"
$Backend = Join-Path $Project "backend"
$Frontend = Join-Path $Project "frontend"
$Runtime = Join-Path $Project ".runtime"
$SessionDir = Join-Path $Runtime ("session_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$BrowserProfile = Join-Path $Runtime "browser-profile"
$ApiKeyFile = Join-Path $SessionDir "google_maps_api_key.txt"
$DatasetDir = Join-Path $Root "dataset"
$TempDir = Join-Path $Root "temp"
$BackendUrl = "http://127.0.0.1:$BackendPort"
$FrontendUrl = "http://127.0.0.1:$FrontendPort"

New-Item -ItemType Directory -Force -Path $Runtime, $SessionDir, $BrowserProfile, $DatasetDir, $TempDir | Out-Null

function Get-RelativeFileSet([string]$Path) {
    if (-not (Test-Path $Path)) {
        return @{}
    }
    $rootPath = (Resolve-Path $Path).Path.TrimEnd("\")
    $set = @{}
    Get-ChildItem -LiteralPath $rootPath -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        $relative = $_.FullName.Substring($rootPath.Length).TrimStart("\")
        $set[$relative] = $true
    }
    return $set
}

function Get-RelativeDirectorySet([string]$Path) {
    if (-not (Test-Path $Path)) {
        return @{}
    }
    $rootPath = (Resolve-Path $Path).Path.TrimEnd("\")
    $set = @{}
    Get-ChildItem -LiteralPath $rootPath -Recurse -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $relative = $_.FullName.Substring($rootPath.Length).TrimStart("\")
        if ($relative) {
            $set[$relative] = $true
        }
    }
    return $set
}

function Get-NewKeys($Before, $After) {
    $items = New-Object System.Collections.Generic.List[string]
    foreach ($key in $After.Keys) {
        if (-not $Before.ContainsKey($key)) {
            $items.Add($key)
        }
    }
    return $items | Sort-Object
}

function Wait-HttpOk([string]$Url, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Milliseconds 600
        }
    }
    return $false
}

function Start-LocalProcess([string]$FileName, [string]$Arguments, [string]$WorkingDirectory, [hashtable]$Environment = @{}) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FileName
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    foreach ($key in $Environment.Keys) {
        $psi.EnvironmentVariables[$key] = [string]$Environment[$key]
    }
    return [System.Diagnostics.Process]::Start($psi)
}

function Find-Browser {
    $candidates = @(
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:LocalAppData\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    return ""
}

function Stop-IfRunning($Process) {
    if ($null -ne $Process) {
        try {
            if (-not $Process.HasExited) {
                $Process.Kill()
                $Process.WaitForExit(5000) | Out-Null
            }
        }
        catch {
        }
    }
}

function Stop-ProjectProcessOnPort([int]$Port, [string]$Label) {
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $connection) {
        return
    }
    $processId = [int]$connection.OwningProcess
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    $commandLine = if ($processInfo) { [string]$processInfo.CommandLine } else { "" }
    if ($commandLine -like "*$Project*" -or $commandLine -like "*uvicorn*app.main:app*" -or $commandLine -like "*vite*") {
        Write-Host "Stopping previous $Label on port $Port (PID $processId)."
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 800
        return
    }
    throw "Port $Port is already in use by PID $processId. Close it first or choose another port."
}

$python = Join-Path $Backend ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "D:\TREE\hk-tree-labeler\backend\.venv\Scripts\python.exe"
}
$vite = Join-Path $Frontend "node_modules\vite\bin\vite.js"
if (-not (Test-Path $vite)) {
    $vite = "D:\TREE\hk-tree-labeler\frontend\node_modules\vite\bin\vite.js"
}
$nodeCommand = Get-Command "node.exe" -ErrorAction SilentlyContinue
if (-not (Test-Path $python)) {
    throw "Backend virtual environment not found: $python. Install backend dependencies first."
}
if (-not (Test-Path $vite)) {
    throw "Vite not found: $vite. Run npm install first."
}
if (-not $nodeCommand) {
    throw "node.exe not found. Install Node.js or add it to PATH."
}
$node = $nodeCommand.Source

$datasetFilesBefore = Get-RelativeFileSet $DatasetDir
$datasetDirsBefore = Get-RelativeDirectorySet $DatasetDir
$tempDirsBefore = Get-RelativeDirectorySet $TempDir
$sessionStartedAt = Get-Date
$backendProcess = $null
$frontendProcess = $null
$browserProcess = $null
$status = $null

try {
    Stop-ProjectProcessOnPort $BackendPort "backend"
    Stop-ProjectProcessOnPort $FrontendPort "frontend"

    Write-Host ""
    Write-Host "Starting backend: $BackendUrl ..."
    Write-Host "Google Maps API Key will be entered and validated in the browser UI."
    $backendArgs = "-m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort"
    $backendProcess = Start-LocalProcess $python $backendArgs $Backend @{
        GOOGLE_MAPS_API_KEY = ""
        GOOGLE_MAPS_API_KEY_FILE = $ApiKeyFile
    }

    if (-not (Wait-HttpOk "$BackendUrl/api/species/list" 45)) {
        throw "Backend startup timed out. Check whether port $BackendPort is already in use."
    }

    Write-Host "Starting frontend: $FrontendUrl ..."
    $frontendArgs = "`"$vite`" --host 127.0.0.1 --port $FrontendPort"
    $frontendProcess = Start-LocalProcess $node $frontendArgs $Frontend @{
        VITE_API_BASE = $BackendUrl
    }

    if (-not (Wait-HttpOk $FrontendUrl 45)) {
        throw "Frontend startup timed out. Check whether port $FrontendPort is already in use."
    }

    $browser = Find-Browser
    Write-Host ""
    Write-Host "Services are ready:"
    Write-Host "  Frontend: $FrontendUrl"
    Write-Host "  Backend:  $BackendUrl"
    Write-Host ""

    if ($browser) {
        Write-Host "Opening a dedicated browser window."
        Write-Host "Close that browser window to stop all services and print the session report."
        $browserArgs = "--user-data-dir=`"$BrowserProfile`" --no-first-run --new-window --app=$FrontendUrl"
        $browserProcess = Start-LocalProcess $browser $browserArgs $Project
        $browserProcess.WaitForExit()
    }
    else {
        Write-Host "Edge/Chrome was not found. Open this URL manually: $FrontendUrl"
        Write-Host "Press Enter here when finished; services will then stop and a report will be printed."
        Read-Host | Out-Null
    }

    try {
        $status = Invoke-RestMethod -Uri "$BackendUrl/api/task/status" -TimeoutSec 5
    }
    catch {
        $status = $null
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping session processes..."
    Stop-IfRunning $browserProcess
    Stop-IfRunning $frontendProcess
    Stop-IfRunning $backendProcess

    $sessionEndedAt = Get-Date
    $datasetFilesAfter = Get-RelativeFileSet $DatasetDir
    $datasetDirsAfter = Get-RelativeDirectorySet $DatasetDir
    $tempDirsAfter = Get-RelativeDirectorySet $TempDir
    $newDatasetFiles = @(Get-NewKeys $datasetFilesBefore $datasetFilesAfter)
    $newDatasetDirs = @(Get-NewKeys $datasetDirsBefore $datasetDirsAfter)
    $newTempDirs = @(Get-NewKeys $tempDirsBefore $tempDirsAfter)

    $savedSampleDirs = @(
        $newDatasetDirs |
            Where-Object { ($_ -split "\\").Count -eq 2 } |
            Sort-Object
    )
    $newLabelFiles = @(
        $newDatasetFiles |
            Where-Object { $_.ToLowerInvariant().EndsWith(".txt") } |
            Sort-Object
    )
    $newImageFiles = @(
        $newDatasetFiles |
            Where-Object { $_.ToLowerInvariant().EndsWith(".jpg") -or $_.ToLowerInvariant().EndsWith(".jpeg") -or $_.ToLowerInvariant().EndsWith(".png") } |
            Sort-Object
    )

    $report = New-Object System.Collections.Generic.List[string]
    $report.Add("HK Tree Street View Labeler - Session Report")
    $report.Add("Started:  $($sessionStartedAt.ToString('yyyy-MM-dd HH:mm:ss'))")
    $report.Add("Ended:    $($sessionEndedAt.ToString('yyyy-MM-dd HH:mm:ss'))")
    $report.Add("Duration: $([Math]::Round(($sessionEndedAt - $sessionStartedAt).TotalMinutes, 2)) minutes")
    $report.Add("")

    if ($null -ne $status) {
        $report.Add("Task status: $($status.status)")
        $report.Add("Species: $($status.species)")
        $report.Add("Target count: $($status.target_count)")
        $report.Add("Queued for review: $($status.queued_count)")
        $report.Add("Saved samples: $($status.reviewed_count)")
        $report.Add("Rejected samples: $($status.rejected_count)")
        $report.Add("Skipped failed samples: $($status.failed_count)")
        $report.Add("Pending queue: $($status.pending)")
        if ($status.message) {
            $report.Add("Message: $($status.message)")
        }
        $report.Add("")
        $report.Add("API calls:")
        if ($status.api_counts) {
            foreach ($entry in ($status.api_counts.PSObject.Properties | Sort-Object Name)) {
                $report.Add("  - $($entry.Name): $($entry.Value)")
            }
        }
        else {
            $report.Add("  - unavailable")
        }
        $report.Add("")
        $report.Add("API errors/retries:")
        if ($status.api_error_counts) {
            foreach ($entry in ($status.api_error_counts.PSObject.Properties | Sort-Object Name)) {
                $report.Add("  - $($entry.Name): $($entry.Value)")
            }
        }
        else {
            $report.Add("  - none recorded")
        }
    }
    else {
        $report.Add("Task status: unavailable; backend may have exited early.")
    }

    $report.Add("")
    $report.Add("New dataset sample dirs: $($savedSampleDirs.Count)")
    foreach ($item in ($savedSampleDirs | Select-Object -First 30)) {
        $report.Add("  - dataset\$item")
    }
    if ($savedSampleDirs.Count -gt 30) {
        $report.Add("  ... $($savedSampleDirs.Count - 30) more omitted")
    }

    $report.Add("")
    $report.Add("New image files: $($newImageFiles.Count)")
    $report.Add("New YOLO label files: $($newLabelFiles.Count)")
    $report.Add("New temp review dirs: $($newTempDirs.Count)")

    $reportPath = Join-Path $SessionDir "report.txt"
    $report | Out-File -FilePath $reportPath -Encoding UTF8

    Write-Host ""
    Write-Host ($report -join [Environment]::NewLine)
    Write-Host ""
    Write-Host "Report saved to: $reportPath"
    Write-Host "Session ended."
}

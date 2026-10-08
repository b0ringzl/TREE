param(
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Continue"
$Project = $PSScriptRoot
$Backend = Join-Path $Project "backend"
$Runtime = Join-Path $Project ".runtime"
$ApiKeyFile = Join-Path $Runtime "google_maps_api_key.txt"
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
if (-not $ApiKey -and (Test-Path $ApiKeyFile)) {
    $ApiKey = [System.IO.File]::ReadAllText($ApiKeyFile).Trim()
}
if (-not $ApiKey) {
    $secure = Read-Host "Enter Google Maps API Key" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $ApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}
if (-not $ApiKey) {
    throw "API Key cannot be empty."
}
$ApiKey = $ApiKey.Trim()
[System.IO.File]::WriteAllText($ApiKeyFile, $ApiKey, $Utf8NoBom)
$env:GOOGLE_MAPS_API_KEY = $ApiKey
$env:GOOGLE_MAPS_API_KEY_FILE = $ApiKeyFile
Set-Location $Backend
& (Join-Path $Backend ".venv\Scripts\python.exe") -m uvicorn app.main:app --host 127.0.0.1 --port 8000 *>> (Join-Path $Backend "server.log")

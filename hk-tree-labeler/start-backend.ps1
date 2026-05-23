param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey
)

$ErrorActionPreference = "Continue"
$env:GOOGLE_MAPS_API_KEY = $ApiKey
Set-Location "D:\TREE\hk-tree-labeler\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 *>> "D:\TREE\hk-tree-labeler\backend\server.log"

$ErrorActionPreference = "Stop"
$env:VITE_API_BASE = "http://127.0.0.1:8010"
Set-Location "D:\TREE\hk-tree-labeler-labelpaw-hover-sandbox\frontend"
try {
    $Vite = "D:\TREE\hk-tree-labeler-labelpaw-hover-sandbox\frontend\node_modules\vite\bin\vite.js"
    if (-not (Test-Path $Vite)) {
        $Vite = "D:\TREE\hk-tree-labeler\frontend\node_modules\vite\bin\vite.js"
    }
    node $Vite --host 127.0.0.1 --port 5175 *>> "D:\TREE\hk-tree-labeler-labelpaw-hover-sandbox\frontend\vite.log"
}
catch {
    $_ | Out-String | Out-File "D:\TREE\hk-tree-labeler-labelpaw-hover-sandbox\frontend\vite.err.log" -Append
    throw
}

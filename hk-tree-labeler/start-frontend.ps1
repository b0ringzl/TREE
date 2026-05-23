$ErrorActionPreference = "Stop"
Set-Location "D:\TREE\hk-tree-labeler\frontend"
try {
    npm run dev -- --host 127.0.0.1 --port 5173 *>> "D:\TREE\hk-tree-labeler\frontend\vite.log"
}
catch {
    $_ | Out-String | Out-File "D:\TREE\hk-tree-labeler\frontend\vite.err.log" -Append
    throw
}

param([ValidateSet('image-only','multimodal')][string]$Edition='multimodal',[switch]$NoBrowser,[switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$taskPython=Join-Path $taskRoot 'runtime\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Extract the complete ZIP before starting.' }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
$env:PATH="$taskRoot\runtime;$taskRoot\runtime\Library\bin;$taskRoot\runtime\DLLs;$taskRoot\runtime\Scripts;$env:PATH"
$taskArguments=@('-s', (Join-Path $taskRoot 'launch.py'), '--edition', $Edition)
if ($NoBrowser) { $taskArguments+='--no-browser' }
if ($CheckOnly) { $taskArguments+='--check-only' }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { Write-Host 'Startup failed. Read the message above and user_data logs.'; exit $LASTEXITCODE }

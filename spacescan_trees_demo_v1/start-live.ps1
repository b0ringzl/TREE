param([switch]$OpenBrowser, [ValidateSet('image-only','multimodal')][string]$Edition = 'multimodal')
$ErrorActionPreference = 'Stop'
$demoProject = Split-Path -Parent $PSScriptRoot
$demoPython = $env:SPACESCAN_PYTHON
if (-not $demoPython) {
    $demoCondaPython = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'
    if (Test-Path -LiteralPath $demoCondaPython -PathType Leaf) { $demoPython = $demoCondaPython }
    else {
        $demoPythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($demoPythonCommand) { $demoPython = $demoPythonCommand.Source }
    }
}
if (-not $demoPython -or -not (Test-Path -LiteralPath $demoPython -PathType Leaf)) {
    throw 'Set SPACESCAN_PYTHON to your Python executable. See README.md.'
}
if (-not (Get-NetTCPConnection -State Listen -LocalPort 8039 -ErrorAction SilentlyContinue)) {
    foreach ($demoRequired in @(
        'external_datasets\distillation_models\bioclip25\open_clip_model.safetensors',
        'outputs\distillation\bioclip_probe_20260908_0227\classes.json',
        'outputs\distillation\bioclip_probe_20260908_0227\protocol.json',
        'outputs\distillation\20260908_0045\aliases.json',
        'tree_species_teacher_demo\models\SSTv1_Aquilaria_UserCrops_20260919_H120\member_1.pt',
        'tree_species_teacher_demo\models\SSTv1_Aquilaria_UserCrops_20260919_H120\member_2.pt',
        'tree_species_teacher_demo\models\SSTv1_Aquilaria_UserCrops_20260919_H120\member_3.pt'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $demoProject $demoRequired) -PathType Leaf)) {
            throw "Missing local model asset: $demoRequired. Restore authorized assets using ASSETS_REQUIRED.md; this Git checkout is source-only."
        }
    }
}
if (-not $env:SAM_PYTHON) { $env:SAM_PYTHON = $demoPython }
$demoLogs = Join-Path $PSScriptRoot ('logs\live_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $demoLogs | Out-Null
foreach ($demoService in @(@{port=8039;module='tree_species_teacher_demo.server:app'},@{port=8043;module='spacescan_trees_demo_v1.server:app'})) {
    if (-not (Get-NetTCPConnection -State Listen -LocalPort $demoService.port -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $demoPython -ArgumentList @('-m','uvicorn',$demoService.module,'--host','127.0.0.1','--port',[string]$demoService.port) -WorkingDirectory $demoProject -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoLogs ($demoService.port.ToString()+'.out.log')) -RedirectStandardError (Join-Path $demoLogs ($demoService.port.ToString()+'.err.log'))
    }
}
$demoRelease = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'release.json') -Raw | ConvertFrom-Json
$demoDeadline = (Get-Date).AddSeconds(180)
$demoHealth = $null
do {
    try { $demoHealth = Invoke-RestMethod 'http://127.0.0.1:8043/api/health' -TimeoutSec 5 } catch { $demoHealth = $null }
    if ($demoHealth.single_image_version -and $demoHealth.single_image_version -ne $demoRelease.classifier_version) {
        throw 'The running model differs from the pinned release. No service was stopped.'
    }
    if ($demoHealth.ready -and $demoHealth.release_id -eq $demoRelease.release_id) { break }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $demoDeadline)
if (-not $demoHealth.ready -or $demoHealth.release_id -ne $demoRelease.release_id) {
    throw "Demo is not ready. Inspect $demoLogs. No model was replaced."
}
$demoUrl = 'http://127.0.0.1:8043/' + $Edition
Write-Host "SpaceScan Trees: $demoUrl"
Write-Host 'No training, model download or weight replacement was performed.'
if ($OpenBrowser) { Start-Process $demoUrl }

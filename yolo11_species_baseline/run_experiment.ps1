param(
    [string]$RunName = "yolo11s_10class_v2",
    [string]$Model = "yolo11s.pt",
    [int]$Epochs = 80,
    [int]$BatchSize = 48,
    [int]$ImageSize = 640,
    [int]$Seed = 20260824,
    [double]$LearningRate = 0.001
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "envs\yolo11-tree\Scripts\python.exe"
$RunDir = Join-Path $PSScriptRoot "experiments\$RunName"
$DatasetDir = Join-Path $RunDir "dataset"
$Manifest = Join-Path $ProjectRoot "image_species_baseline\experiments\hk_web_resnet50_10class_v2\dataset\manifest.csv"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "YOLO11 environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Source manifest not found: $Manifest"
}
if (Test-Path -LiteralPath (Join-Path $RunDir "final_test_metrics.json") -PathType Leaf) {
    throw "Completed run already exists. Choose a new -RunName: $RunDir"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

& $Python (Join-Path $PSScriptRoot "scripts\build_yolo_dataset.py") `
    --manifest $Manifest `
    --output-dir $DatasetDir
if ($LASTEXITCODE -ne 0) { throw "YOLO dataset build failed" }

& $Python (Join-Path $PSScriptRoot "scripts\train_yolo11.py") `
    --dataset-yaml (Join-Path $DatasetDir "dataset.yaml") `
    --run-dir $RunDir `
    --model $Model `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --image-size $ImageSize `
    --seed $Seed `
    --learning-rate $LearningRate
if ($LASTEXITCODE -ne 0) { throw "YOLO11 training failed" }

& $Python (Join-Path $PSScriptRoot "scripts\evaluate_yolo11.py") `
    --dataset-yaml (Join-Path $DatasetDir "dataset.yaml") `
    --run-dir $RunDir `
    --image-size $ImageSize
if ($LASTEXITCODE -ne 0) { throw "YOLO11 test/inference failed" }

& $Python (Join-Path $PSScriptRoot "scripts\validate_run.py") `
    --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "YOLO11 run acceptance failed" }

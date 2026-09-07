param(
    [string]$RunName = "streetview_yolo11m_seg_20260906_seed1",
    [int]$Epochs = 60,
    [int]$BatchSize = 16,
    [int]$ImageSize = 640,
    [int]$Seed = 20260906
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "envs\yolo11-tree\Scripts\python.exe"
$Manifest = Join-Path $PSScriptRoot "experiments\full_web_resnet50_20260906_seed1\dataset\crop_manifest.csv"
$RunRoot = Join-Path $PSScriptRoot "experiments\$RunName"
$DatasetRoot = Join-Path $RunRoot "dataset"
$TrainRun = Join-Path $RunRoot "train"
$InitialModel = Join-Path $ProjectRoot "yolo11m-seg.pt"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $InitialModel -PathType Leaf)) { throw "Model not found: $InitialModel" }
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot "dataset_summary.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\materialize_yolo11_seg.py") --manifest $Manifest --output-dir $DatasetRoot
    if ($LASTEXITCODE -ne 0) { throw "YOLO segmentation dataset materialization failed" }
}

if (-not (Test-Path -LiteralPath (Join-Path $TrainRun "training_complete.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\train_yolo11_seg.py") --data (Join-Path $DatasetRoot "data.yaml") --run-dir $TrainRun --model $InitialModel --epochs $Epochs --batch-size $BatchSize --image-size $ImageSize --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "YOLO segmentation training failed" }
}

if (-not (Test-Path -LiteralPath (Join-Path $TrainRun "final_test_metrics.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\evaluate_yolo11_seg.py") --data (Join-Path $DatasetRoot "test.yaml") --dataset-summary (Join-Path $DatasetRoot "dataset_summary.json") --run-dir $TrainRun --batch-size $BatchSize --image-size $ImageSize
    if ($LASTEXITCODE -ne 0) { throw "YOLO segmentation test evaluation failed" }
}

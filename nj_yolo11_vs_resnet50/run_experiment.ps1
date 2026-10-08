param(
    [string]$RunName = "nj_yolo11s_cls_seed_20260728",
    [string]$Model = "yolo11s-cls.pt",
    [int]$Epochs = 100,
    [int]$BatchSize = 128,
    [int]$ImageSize = 224,
    [int]$Seed = 20260728,
    [double]$LearningRate = 0.001
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "envs\yolo11-tree\Scripts\python.exe"
$DatasetPackage = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\d1_four_class_training_package\20260817_165127"
$ResNetSuite = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\d1_resnet50_clean_repeats\20260817_protocol_v1"
$RunDir = Join-Path $PSScriptRoot "experiments\$RunName"
$DatasetDir = Join-Path $RunDir "dataset"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $DatasetPackage -PathType Container)) { throw "NJ package not found: $DatasetPackage" }
if (Test-Path -LiteralPath (Join-Path $RunDir "comparison.json") -PathType Leaf) {
    throw "Completed run already exists. Choose a new -RunName: $RunDir"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

& $Python (Join-Path $PSScriptRoot "scripts\build_classification_dataset.py") `
    --source-package $DatasetPackage `
    --output-dir $DatasetDir
if ($LASTEXITCODE -ne 0) { throw "Dataset build failed" }

& $Python (Join-Path $PSScriptRoot "scripts\train_yolo11_classifier.py") `
    --dataset-root $DatasetDir `
    --run-dir $RunDir `
    --model $Model `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --image-size $ImageSize `
    --seed $Seed `
    --learning-rate $LearningRate
if ($LASTEXITCODE -ne 0) { throw "YOLO11 classification training failed" }

& $Python (Join-Path $PSScriptRoot "scripts\select_and_evaluate.py") `
    --dataset-root $DatasetDir `
    --run-dir $RunDir `
    --image-size $ImageSize `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Checkpoint selection or test evaluation failed" }

& $Python (Join-Path $PSScriptRoot "scripts\compare_with_resnet50.py") `
    --yolo-run-dir $RunDir `
    --resnet-suite $ResNetSuite `
    --resnet-seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Model comparison failed" }

& $Python (Join-Path $PSScriptRoot "scripts\validate_comparison.py") `
    --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Comparison acceptance failed" }

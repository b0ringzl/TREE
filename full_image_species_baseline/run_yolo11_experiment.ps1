param(
    [string]$RunName = "full_web_yolo11s_cls_20260906_seed1",
    [int]$PretrainEpochs = 15,
    [int]$FinetuneEpochs = 30,
    [int]$BatchSize = 128,
    [int]$ImageSize = 224,
    [int]$Seed = 20260906
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "envs\yolo11-tree\Scripts\python.exe"
$ManifestRoot = Join-Path $PSScriptRoot "experiments\full_web_resnet50_20260906_seed1\dataset"
$RunRoot = Join-Path $PSScriptRoot "experiments\$RunName"
$FullDataset = Join-Path $RunRoot "datasets\full"
$CropDataset = Join-Path $RunRoot "datasets\crop"
$PretrainRun = Join-Path $RunRoot "pretrain"
$FinetuneRun = Join-Path $RunRoot "finetune"
$InitialModel = Join-Path $ProjectRoot "yolo11s-cls.pt"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $InitialModel -PathType Leaf)) { throw "Model not found: $InitialModel" }
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $FullDataset "dataset_summary.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\materialize_yolo_cls.py") --manifest (Join-Path $ManifestRoot "full_manifest.csv") --output-dir $FullDataset
    if ($LASTEXITCODE -ne 0) { throw "Full YOLO dataset materialization failed" }
}
if (-not (Test-Path -LiteralPath (Join-Path $CropDataset "dataset_summary.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\materialize_yolo_cls.py") --manifest (Join-Path $ManifestRoot "crop_manifest.csv") --output-dir $CropDataset
    if ($LASTEXITCODE -ne 0) { throw "Crop YOLO dataset materialization failed" }
}

if (-not (Test-Path -LiteralPath (Join-Path $PretrainRun "final_metrics.json"))) {
    if (-not (Test-Path -LiteralPath (Join-Path $PretrainRun "training_complete.json"))) {
        & $Python (Join-Path $PSScriptRoot "scripts\train_yolo11_cls.py") --dataset-root $FullDataset --run-dir $PretrainRun --model $InitialModel --epochs $PretrainEpochs --batch-size $BatchSize --image-size $ImageSize --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "YOLO full-image pretraining failed" }
    }
    & $Python (Join-Path $PSScriptRoot "scripts\evaluate_yolo11_cls.py") --dataset-root $FullDataset --run-dir $PretrainRun --image-size $ImageSize
    if ($LASTEXITCODE -ne 0) { throw "YOLO pretraining evaluation failed" }
}

if (-not (Test-Path -LiteralPath (Join-Path $FinetuneRun "final_metrics.json"))) {
    if (-not (Test-Path -LiteralPath (Join-Path $FinetuneRun "training_complete.json"))) {
        & $Python (Join-Path $PSScriptRoot "scripts\train_yolo11_cls.py") --dataset-root $CropDataset --run-dir $FinetuneRun --model (Join-Path $PretrainRun "weights\selected_by_val_group_macro_f1.pt") --epochs $FinetuneEpochs --batch-size ([Math]::Min($BatchSize, 96)) --image-size $ImageSize --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "YOLO crop fine-tuning failed" }
    }
    & $Python (Join-Path $PSScriptRoot "scripts\evaluate_yolo11_cls.py") --dataset-root $CropDataset --run-dir $FinetuneRun --image-size $ImageSize
    if ($LASTEXITCODE -ne 0) { throw "YOLO fine-tuning evaluation failed" }
}

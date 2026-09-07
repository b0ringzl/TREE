param(
    [string]$RunName = "full_web_resnet50_20260906_seed1",
    [int]$PretrainEpochs = 15,
    [int]$FinetuneEpochs = 30,
    [int]$BatchSize = 128,
    [int]$ImageSize = 224,
    [int]$Seed = 20260906,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$RunDir = Join-Path $PSScriptRoot "experiments\$RunName"
$DatasetDir = Join-Path $RunDir "dataset"
$PretrainDir = Join-Path $RunDir "pretrain"
$FinetuneDir = Join-Path $RunDir "finetune"
$SourceCheckpoint = Join-Path $ProjectRoot "image_species_baseline\experiments\hk_web_resnet50_10class_v2\best.pt"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $SourceCheckpoint -PathType Leaf)) { throw "Source checkpoint not found: $SourceCheckpoint" }
if ((Test-Path -LiteralPath (Join-Path $RunDir "acceptance.json")) -and -not $Resume) {
    throw "Completed run exists. Choose a new RunName or use -Resume: $RunDir"
}
New-Item -ItemType Directory -Force -Path $DatasetDir, $PretrainDir, $FinetuneDir | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $DatasetDir "dataset_summary.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\build_manifests.py") --output-dir $DatasetDir --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "Manifest build failed" }
}

$resumeArg = @()
if ($Resume) { $resumeArg = @("--resume") }
if (-not (Test-Path -LiteralPath (Join-Path $PretrainDir "final_metrics.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\train_resnet50.py") `
        --manifest (Join-Path $DatasetDir "full_manifest.csv") `
        --source-checkpoint $SourceCheckpoint `
        --output-dir $PretrainDir `
        --stage-name "full-web-image-pretrain" `
        --epochs $PretrainEpochs --warmup-epochs 2 --batch-size $BatchSize `
        --image-size $ImageSize --seed $Seed @resumeArg
    if ($LASTEXITCODE -ne 0) { throw "Full-image pretraining failed" }
}

if (-not (Test-Path -LiteralPath (Join-Path $FinetuneDir "final_metrics.json"))) {
    & $Python (Join-Path $PSScriptRoot "scripts\train_resnet50.py") `
        --manifest (Join-Path $DatasetDir "crop_manifest.csv") `
        --source-checkpoint (Join-Path $PretrainDir "best.pt") `
        --output-dir $FinetuneDir `
        --stage-name "human-annotated-crop-finetune" `
        --epochs $FinetuneEpochs --warmup-epochs 2 --batch-size ([Math]::Min($BatchSize, 96)) `
        --image-size $ImageSize --seed $Seed @resumeArg
    if ($LASTEXITCODE -ne 0) { throw "Annotated-crop fine-tuning failed" }
}

& $Python (Join-Path $PSScriptRoot "scripts\summarize_run.py") --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Run validation failed" }

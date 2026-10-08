param(
    [string]$RunName = "hk_web_resnet50_10class_v2",
    [int]$Epochs = 60,
    [int]$BatchSize = 64,
    [int]$ImageSize = 320,
    [int]$Seed = 20260824
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$RunDir = Join-Path $PSScriptRoot "experiments\$RunName"
$DatasetDir = Join-Path $RunDir "dataset"
$CopiedCheckpoint = Join-Path $RunDir "source_model_copy\d1_seed_20260728_best_4class.pt"
$SourceCheckpoint = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\d1_resnet50_clean_repeats\20260817_protocol_v1\runs\seed_20260728\best.pt"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $SourceCheckpoint -PathType Leaf)) {
    throw "Source checkpoint not found: $SourceCheckpoint"
}
if (Test-Path -LiteralPath (Join-Path $RunDir "final_metrics.json") -PathType Leaf) {
    throw "Completed run already exists. Choose a new -RunName to avoid overwriting it: $RunDir"
}
New-Item -ItemType Directory -Force -Path $DatasetDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CopiedCheckpoint) | Out-Null
if (-not (Test-Path -LiteralPath $CopiedCheckpoint -PathType Leaf)) {
    Copy-Item -LiteralPath $SourceCheckpoint -Destination $CopiedCheckpoint
}

& $Python (Join-Path $PSScriptRoot "scripts\build_ten_class_manifest.py") `
    --output-dir $DatasetDir `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Dataset manifest build failed" }

& $Python (Join-Path $PSScriptRoot "scripts\train_resnet50_transfer.py") `
    --manifest (Join-Path $DatasetDir "manifest.csv") `
    --source-checkpoint $CopiedCheckpoint `
    --output-dir $RunDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --image-size $ImageSize `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Training failed" }

& $Python (Join-Path $PSScriptRoot "scripts\plot_training_history.py") `
    --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Training-curve export failed" }

& $Python (Join-Path $PSScriptRoot "scripts\validate_run.py") `
    --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Run validation failed" }

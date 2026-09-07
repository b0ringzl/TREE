"""Build the epoch-60 recovery hotfix for the portable C2 run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "portable_c2_template"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    payload = output / "payload" / "tools"
    payload.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        TEMPLATE / "tools" / "run_training.ps1",
        payload / "run_training.ps1",
    )

    (output / "APPLY_AND_RECOVER.bat").write_text(
        """@echo off
title C2 PTv2 Recovery Update
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0apply_and_recover.ps1"
if errorlevel 1 (
  echo.
  echo Recovery failed. Read the message above and do not delete the checkpoint.
  pause
)
""",
        encoding="ascii",
    )
    (output / "apply_and_recover.ps1").write_text(
        r'''param([switch]$InstallOnly)

$ErrorActionPreference = "Stop"
$UpdateRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleRoot = Split-Path -Parent $UpdateRoot
$RunDir = Join-Path $BundleRoot "output\run"
$LastCheckpoint = Join-Path $RunDir "last.pt"

if (-not (Test-Path -LiteralPath (Join-Path $BundleRoot "START_TRAINING.bat"))) {
    throw "Place this whole recovery folder inside the portable training folder."
}
if (-not (Test-Path -LiteralPath $LastCheckpoint)) {
    throw "No last.pt checkpoint was found; recovery was not started."
}

$ActiveTraining = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -and
    $_.CommandLine.Contains($BundleRoot) -and
    $_.CommandLine.Contains("train_b5_ptv2.py")
})
if ($ActiveTraining.Count -gt 0) {
    throw "A training process is still active. Close it before applying recovery."
}

$ProfilePath = Join-Path $BundleRoot "training_profile.json"
if (Test-Path -LiteralPath $ProfilePath) {
    $BackupName = "training_profile.before_recovery_{0}.json" -f (
        Get-Date -Format "yyyyMMdd_HHmmss"
    )
    Copy-Item -LiteralPath $ProfilePath `
        -Destination (Join-Path $BundleRoot $BackupName)
}

Copy-Item `
    -LiteralPath (Join-Path $UpdateRoot "payload\tools\run_training.ps1") `
    -Destination (Join-Path $BundleRoot "tools\run_training.ps1") `
    -Force

$Profile = @{
    schema_version = 1
    name = "recovery-batch40"
    created_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    batch_size = 40
    eval_batch_size = 48
    workers = 0
    progress_every = 10
    note = "Stable recovery profile after the epoch-61 process failure."
} | ConvertTo-Json
[IO.File]::WriteAllText(
    $ProfilePath,
    $Profile + [Environment]::NewLine,
    (New-Object Text.UTF8Encoding($false))
)

$ManifestPath = Join-Path $BundleRoot "transfer_manifest.json"
if (Test-Path -LiteralPath $ManifestPath) {
    $RelativePath = "tools/run_training.ps1"
    $RunnerPath = Join-Path $BundleRoot "tools\run_training.ps1"
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Entries = @(
        $Manifest.files | Where-Object { $_.path -ne $RelativePath }
    )
    $Entries += [pscustomobject]@{
        path = $RelativePath
        bytes = (Get-Item -LiteralPath $RunnerPath).Length
        sha256 = (
            Get-FileHash -LiteralPath $RunnerPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
    }
    $Entries = @($Entries | Sort-Object path)
    $Manifest.files = $Entries
    $Manifest.file_count = $Entries.Count
    $Manifest.total_bytes = (
        $Entries | Measure-Object -Property bytes -Sum
    ).Sum
    $TemporaryManifest = "$ManifestPath.tmp"
    $Json = $Manifest | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText(
        $TemporaryManifest,
        $Json + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $TemporaryManifest `
        -Destination $ManifestPath -Force
}

Write-Host "Recovery profile installed: training batch 40, evaluation batch 48."
Write-Host "The existing last.pt checkpoint will be resumed."

if ($InstallOnly) {
    exit 0
}

$Monitor = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -and
    $_.CommandLine.Contains($BundleRoot) -and
    $_.CommandLine.Contains("monitor_training.py")
})
if ($Monitor.Count -gt 0) {
    $Runner = Join-Path $BundleRoot "tools\run_training.ps1"
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$Runner`""
        ) `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Recovery launched. Wrapper PID: $($Process.Id)"
    Write-Host "The existing monitor will update automatically."
} else {
    & (Join-Path $BundleRoot "tools\start_and_monitor.ps1")
}
''',
        encoding="ascii",
    )
    (output / "使用说明.txt").write_text(
        """C2 PTv2 第60轮断点恢复更新

1. 把整个恢复更新文件夹复制到便携训练包根目录里面。
2. 保持目录层级不变，双击 APPLY_AND_RECOVER.bat。
3. 脚本不会删除或覆盖 last.pt、best.pt 和训练历史。
4. 它会将训练 batch 调整为 40、验证 batch 调整为 48。
5. 训练会从最新完整断点自动续跑，已有监控窗口会继续刷新。
6. 若再次失败，console.err.log 将保存完整 traceback。
""",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()

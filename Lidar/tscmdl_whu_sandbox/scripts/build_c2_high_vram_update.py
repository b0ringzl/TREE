"""Build the small hot-update folder for an existing portable C2 bundle."""

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
    payload_tools = output / "payload" / "tools"
    payload_tools.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        TEMPLATE / "tools" / "run_training.ps1",
        payload_tools / "run_training.ps1",
    )
    shutil.copy2(
        TEMPLATE / "tools" / "switch_to_high_vram.ps1",
        payload_tools / "switch_to_high_vram.ps1",
    )
    shutil.copy2(
        TEMPLATE / "START_TRAINING_24GB.bat",
        output / "payload" / "START_TRAINING_24GB.bat",
    )

    (output / "APPLY_24GB_UPDATE.bat").write_text(
        """@echo off
title Apply C2 PTv2 24 GB Update
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0apply_update.ps1"
if errorlevel 1 (
  echo.
  echo Update failed. Read the message above.
  pause
)
""",
        encoding="ascii",
    )
    (output / "apply_update.ps1").write_text(
        """param([switch]$InstallOnly)

$ErrorActionPreference = "Stop"
$UpdateRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleRoot = Split-Path -Parent $UpdateRoot
if (-not (Test-Path -LiteralPath (Join-Path $BundleRoot "START_TRAINING.bat"))) {
    throw "Place this whole update folder inside the portable training folder."
}
Copy-Item -LiteralPath (Join-Path $UpdateRoot "payload\\tools\\run_training.ps1") `
    -Destination (Join-Path $BundleRoot "tools\\run_training.ps1") -Force
Copy-Item -LiteralPath (Join-Path $UpdateRoot "payload\\tools\\switch_to_high_vram.ps1") `
    -Destination (Join-Path $BundleRoot "tools\\switch_to_high_vram.ps1") -Force
Copy-Item -LiteralPath (Join-Path $UpdateRoot "payload\\START_TRAINING_24GB.bat") `
    -Destination (Join-Path $BundleRoot "START_TRAINING_24GB.bat") -Force

$ManifestPath = Join-Path $BundleRoot "transfer_manifest.json"
if (Test-Path -LiteralPath $ManifestPath) {
    $ChangedPaths = @(
        "START_TRAINING_24GB.bat",
        "tools/run_training.ps1",
        "tools/switch_to_high_vram.ps1"
    )
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Entries = @(
        $Manifest.files | Where-Object { $_.path -notin $ChangedPaths }
    )
    foreach ($RelativePath in $ChangedPaths) {
        $Path = Join-Path $BundleRoot ($RelativePath -replace "/", "\\")
        $Entries += [pscustomobject]@{
            path = $RelativePath
            bytes = (Get-Item -LiteralPath $Path).Length
            sha256 = (
                Get-FileHash -LiteralPath $Path -Algorithm SHA256
            ).Hash.ToLowerInvariant()
        }
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
    Move-Item -LiteralPath $TemporaryManifest -Destination $ManifestPath -Force
}

Write-Host "Update installed and the integrity manifest was refreshed."
if (-not $InstallOnly) {
    Write-Host "Starting the safe 24 GB switch..."
    & (Join-Path $BundleRoot "tools\\switch_to_high_vram.ps1")
}
""",
        encoding="ascii",
    )
    (output / "使用说明.txt").write_text(
        """C2 PTv2 24GB 显存热更新

1. 把整个 C2_24GB_UPDATE 文件夹复制到便携训练包根目录里面。
2. 双击 APPLY_24GB_UPDATE.bat。
3. 如果训练正在进行，脚本会等待当前轮次完成并保存断点。
4. 随后脚本停止旧进程，用真实数据测试 48、40、32 的批大小。
5. 它会选择能够稳定运行的最大值，并自动从原断点继续训练。

更新不会删除已有权重、历史或输出，也不会修改学习率和模型结构。
批大小改变后，后续优化步数会改变，因此它属于加速续训方案，
不再是与原 batch=16 实验严格逐步等价的复现。
""",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()

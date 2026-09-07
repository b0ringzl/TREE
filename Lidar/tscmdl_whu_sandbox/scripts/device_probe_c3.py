"""Validate the portable C3 runtime, 24 GB GPU, dataset, and kNN cache."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import PIL
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "src"))

from tscmdl_whu import PointTransformerV2Classifier  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def nvidia_smi() -> str:
    result = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def main() -> None:
    if sys.maxsize <= 2**32:
        raise RuntimeError("64-bit Python is required")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access CUDA. Install or update the NVIDIA driver."
        )

    config = json.loads((ROOT / "c3_config.json").read_text(encoding="utf-8"))
    if config["batch_size"] != 48 or config["eval_batch_size"] != 64:
        raise ValueError("C3 fixed batch configuration changed unexpectedly")

    dataset_root = ROOT / "input" / "dataset"
    cache_root = ROOT / "input" / "knn_cache"
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    classes = json.loads(
        (dataset_root / "classes.json").read_text(encoding="utf-8")
    )
    cache_index = json.loads(
        (cache_root / "index.json").read_text(encoding="utf-8")
    )
    train_records = sorted(
        (record for record in manifest["records"] if record["split"] == "train"),
        key=lambda record: str(record["sample_key"]),
    )
    first = train_records[0]
    if first["sample_key"] != cache_index["splits"]["train"]["sample_keys"][0]:
        raise ValueError("Dataset and kNN cache sample order do not match")

    with np.load(dataset_root / str(first["point_path"]), allow_pickle=False) as data:
        points = np.array(data["points_xyz"], dtype=np.float32, copy=True)
    references = np.load(
        cache_root / str(cache_index["splits"]["train"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    reference = np.array(references[0], dtype=np.int64, copy=True)
    mmap = getattr(references, "_mmap", None)
    if mmap is not None:
        mmap.close()

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    total_vram_gib = properties.total_memory / (1024**3)
    if total_vram_gib < 20.0:
        raise RuntimeError(
            "This fixed C3 package requires a 24 GB-class GPU; "
            f"detected {total_vram_gib:.2f} GiB."
        )

    model = PointTransformerV2Classifier(len(classes)).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        logits = model(
            torch.from_numpy(points).unsqueeze(0).to(device),
            torch.from_numpy(reference).unsqueeze(0).to(device),
        )
    torch.cuda.synchronize(device)
    if tuple(logits.shape) != (1, len(classes)):
        raise ValueError(f"Unexpected PTv2 output shape: {tuple(logits.shape)}")

    disk = shutil.disk_usage(ROOT)
    report = {
        "status": "passed",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "matplotlib": matplotlib.__version__,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_vram_gib": round(total_vram_gib, 3),
            "probe_peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 3
            ),
        },
        "fixed_training": {
            "seeds": config["seeds"],
            "batch_size": config["batch_size"],
            "eval_batch_size": config["eval_batch_size"],
        },
        "data": {
            "sample_count": int(manifest["summary"]["sample_count"]),
            "class_count": len(classes),
            "split_sizes": manifest["summary"]["split_histogram"],
            "point_shape": list(points.shape),
            "reference_shape": list(reference.shape),
        },
        "free_disk_gib": round(disk.free / (1024**3), 3),
        "nvidia_smi": nvidia_smi(),
    }
    atomic_json(ROOT / "diagnostics" / "device_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Bind the validated D1 package holdout to the exposure-probe registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_REGISTRY = (
    DERIVED_ROOT
    / "d1_four_class_image_quality"
    / "tracked_samples"
    / "tri_modal_probe_registry.json"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def update_registry_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "training_exclusion_applied" not in fieldnames:
        raise ValueError("Probe registry CSV lacks training_exclusion_applied")
    for row in rows:
        if row.get("cohort_id") == "EXPOSURE_NEGATIVE_TRANSFER_V1":
            row["training_exclusion_applied"] = "True"
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--probe-registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    registry_path = args.probe_registry.resolve()
    registry_csv_path = registry_path.with_suffix(".csv")
    exposure_plan_path = registry_path.parent / "exposure_negative_transfer_plan.json"
    validation_path = package_root / "validation.json"
    holdout_manifest_path = package_root / "holdout" / "manifest.json"
    bindings_path = package_root / "source_bindings.json"
    summary_path = package_root / "package_summary.json"
    for path in (
        registry_path,
        validation_path,
        holdout_manifest_path,
        bindings_path,
        summary_path,
        registry_csv_path,
        exposure_plan_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    validation = read_json(validation_path)
    if validation.get("status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("Package validation must pass before binding the holdout")
    if validation.get("errors"):
        raise ValueError("Package validation still contains errors")
    holdout_manifest = read_json(holdout_manifest_path)
    holdout_keys = {
        str(record["sample_key"]) for record in holdout_manifest["records"]
    }
    if len(holdout_keys) != 7:
        raise ValueError(f"Expected seven holdout probes, found {len(holdout_keys)}")

    registry = read_json(registry_path)
    cohort = next(
        (
            item
            for item in registry.get("cohorts", [])
            if item.get("cohort_id") == "EXPOSURE_NEGATIVE_TRANSFER_V1"
        ),
        None,
    )
    if cohort is None:
        raise ValueError("Exposure cohort is missing from registry")
    if holdout_keys != {str(key) for key in cohort.get("sample_keys", [])}:
        raise ValueError("Package holdout keys do not match the registered cohort")

    bound_at = now_iso()
    binding = {
        "package_id": str(holdout_manifest["package_id"]),
        "package_root": relative(package_root),
        "holdout_manifest": relative(holdout_manifest_path),
        "holdout_manifest_sha256": sha256_file(holdout_manifest_path),
        "sample_count": len(holdout_keys),
        "bound_at": bound_at,
        "validation_status_at_binding": str(validation["status"]),
    }
    snapshot_root = registry_path.parent / "registry_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / (
        f"tri_modal_probe_registry_pre_holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy2(registry_path, snapshot_path)

    reporting_policy = registry.setdefault("reporting_policy", {})
    reporting_policy["generalization_warning"] = (
        "The seven exposure probes are excluded from both clean and silver training "
        "packages; matched normal-exposure controls are still required for a causal claim."
    )
    reporting_policy["training_exclusion_applied"] = True
    reporting_policy["holdout_binding"] = binding
    cohort["status"] = "holdout_applied_pending_predictions"
    cohort["training_exclusion_applied"] = True
    cohort["holdout_binding"] = binding
    for sample in registry.get("samples", []):
        key = str(sample["sample_key"])
        if key not in holdout_keys:
            continue
        role = sample.setdefault("experimental_role", {})
        role["training_exclusion_applied"] = True
        role["interpretation_until_holdout"] = (
            "formally held out from clean and silver training; predictions pending"
        )
        role["holdout_package_id"] = binding["package_id"]
    registry["updated_at"] = bound_at
    atomic_json(registry_path, registry)
    registry_hash = sha256_file(registry_path)
    update_registry_csv(registry_csv_path)
    exposure_plan = read_json(exposure_plan_path)
    exposure_plan["source_registry"] = str(registry_path)
    exposure_plan["source_registry_sha256"] = registry_hash
    exposure_plan["cohort"] = cohort
    atomic_json(exposure_plan_path, exposure_plan)

    source_bindings = read_json(bindings_path)
    source_bindings["probe_registry"] = {
        "path": relative(registry_path),
        "sha256": registry_hash,
    }
    atomic_json(bindings_path, source_bindings)
    summary = read_json(summary_path)
    summary["source_bindings"] = source_bindings
    summary["holdout_binding"] = binding
    atomic_json(summary_path, summary)
    atomic_json(package_root / "holdout_binding.json", binding)
    print(
        json.dumps(
            {
                "status": "bound",
                "registry": relative(registry_path),
                "registry_sha256": registry_hash,
                "snapshot": relative(snapshot_path),
                "binding": binding,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

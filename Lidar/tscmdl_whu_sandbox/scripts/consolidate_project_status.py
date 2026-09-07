"""Generate the authoritative TSCMDL project status and experiment registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.status import (  # noqa: E402
    compare_prediction_records,
    load_prediction_records,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b2-run", type=Path, required=True)
    parser.add_argument("--b3-run", type=Path, required=True)
    parser.add_argument("--b4a-run", type=Path)
    parser.add_argument("--b4-run", type=Path)
    parser.add_argument("--b4c-run", type=Path)
    parser.add_argument("--b5-run", type=Path)
    parser.add_argument(
        "--stage-gates", type=Path, default=SANDBOX_ROOT / "STAGE_GATES.md"
    )
    parser.add_argument(
        "--environment-lock",
        type=Path,
        default=SANDBOX_ROOT / "environment" / "whu-tscmdl-environment.yml",
    )
    parser.add_argument(
        "--b1-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B1_validation.json",
    )
    parser.add_argument(
        "--b2-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B2_validation.json",
    )
    parser.add_argument(
        "--b3-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B3_validation.json",
    )
    parser.add_argument(
        "--b4a-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B4a_validation.json",
    )
    parser.add_argument(
        "--b4-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B4_validation.json",
    )
    parser.add_argument(
        "--b4c-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B4c_validation.json",
    )
    parser.add_argument(
        "--b5-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B5_validation.json",
    )
    parser.add_argument(
        "--b5b-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B5b_validation.json",
    )
    parser.add_argument("--b5c-validation", type=Path)
    parser.add_argument("--c1a-validation", type=Path)
    parser.add_argument("--c1b-validation", type=Path)
    parser.add_argument("--c1c-validation", type=Path)
    parser.add_argument("--c1-validation", type=Path)
    parser.add_argument("--c2a-validation", type=Path)
    parser.add_argument("--c2-run", type=Path)
    parser.add_argument(
        "--c2-validation",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "C2_validation.json",
    )
    parser.add_argument(
        "--c2-report",
        type=Path,
        default=SANDBOX_ROOT
        / "reports"
        / "C2b_完整19类PTv2结果收口_阶段报告.md",
    )
    parser.add_argument(
        "--four-model-comparison",
        type=Path,
        default=SANDBOX_ROOT
        / "reports"
        / "B2_B3_B4_B5_prediction_comparison.json",
    )
    parser.add_argument(
        "--three-model-comparison",
        type=Path,
        default=SANDBOX_ROOT
        / "reports"
        / "B2_B3_B4_prediction_comparison.json",
    )
    parser.add_argument(
        "--registry-output",
        type=Path,
        default=SANDBOX_ROOT / "experiment_registry.json",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B2_B3_prediction_comparison.json",
    )
    parser.add_argument(
        "--status-output", type=Path, default=SANDBOX_ROOT / "CURRENT_STATUS.md"
    )
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )


def parse_stage_gates(path: Path) -> list[dict[str, str]]:
    rows = []
    pattern = re.compile(r"^\|\s*([A-Z]\w*)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match and match.group(1) != "阶段":
            rows.append(
                {
                    "stage": match.group(1),
                    "scope": match.group(2),
                    "status": match.group(3),
                }
            )
    if not rows:
        raise ValueError(f"No stage rows found in {path}")
    return rows


def metric_subset(metrics: dict[str, object]) -> dict[str, float]:
    return {
        name: float(metrics[name])
        for name in (
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    }


def run_entry(
    stage: str,
    model_name: str,
    run_dir: Path,
    validation_path: Path,
) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "final_metrics.json"
    predictions_path = run_dir / "predictions.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    required = (
        config_path,
        metrics_path,
        predictions_path,
        best_path,
        last_path,
        validation_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {stage} artifacts: {missing}")

    config = read_json(config_path)
    final = read_json(metrics_path)
    validation = read_json(validation_path)
    if not isinstance(config, dict) or not isinstance(final, dict):
        raise ValueError(f"Invalid {stage} config or final metrics")
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError(f"{stage} validation did not pass")

    return {
        "stage": stage,
        "status": "complete",
        "model": model_name,
        "run_dir": str(run_dir.resolve()),
        "epochs_completed": int(final["epochs_completed"]),
        "best_epoch": int(final["best_epoch"]),
        "class_names": list(final["class_names"]),
        "split_sizes": dict(config["split_sizes"]),
        "metrics": {
            split: metric_subset(final["metrics"][split])
            for split in ("val", "test")
        },
        "validation_status": validation["status"],
        "artifacts": {
            "best_checkpoint": {
                "path": str(best_path.resolve()),
                "bytes": best_path.stat().st_size,
                "sha256": sha256_file(best_path),
            },
            "last_checkpoint": {
                "path": str(last_path.resolve()),
                "bytes": last_path.stat().st_size,
                "sha256": sha256_file(last_path),
            },
            "final_metrics": {
                "path": str(metrics_path.resolve()),
                "sha256": sha256_file(metrics_path),
            },
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": sha256_file(predictions_path),
            },
            "run_config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "sha256": sha256_file(validation_path),
            },
        },
    }


def b4a_entry(run_dir: Path, validation_path: Path) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    result_path = run_dir / "smoke_result.json"
    checkpoint_path = run_dir / "last.pt"
    required = (config_path, result_path, checkpoint_path, validation_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B4a artifacts: {missing}")

    config = read_json(config_path)
    result = read_json(result_path)
    validation = read_json(validation_path)
    if not all(isinstance(item, dict) for item in (config, result, validation)):
        raise ValueError("Invalid B4a config, result, or validation")
    if result.get("status") != "passed" or validation.get("status") != "passed":
        raise ValueError("B4a validation did not pass")

    return {
        "stage": "B4a",
        "status": "complete",
        "model": "TSCMDL frozen-backbone fusion smoke",
        "run_dir": str(run_dir.resolve()),
        "steps_completed": int(result["successful_optimizer_steps"]),
        "class_names": list(config["class_names"]),
        "split_sizes": dict(config["split_sizes"]),
        "feature_shapes": dict(result["feature_shapes"]),
        "parameter_counts": dict(result["parameter_counts"]),
        "metrics_scope": "one-batch smoke only",
        "validation_status": validation["status"],
        "artifacts": {
            "last_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
            },
            "smoke_result": {
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
            },
            "run_config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "sha256": sha256_file(validation_path),
            },
        },
    }


def b5b_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing B5b validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("B5b validation did not pass")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("B5b artifact registry is missing")
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing B5b artifact: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    per_seed = validation.get("per_seed")
    if not isinstance(per_seed, dict) or len(per_seed) != 3:
        raise ValueError("B5b must contain exactly three completed seeds")
    return {
        "stage": "B5b",
        "status": "complete",
        "model": "PTv2 locked three-seed repeatability",
        "seeds": list(validation["seeds"]),
        "class_names": list(validation["class_names"]),
        "per_seed": per_seed,
        "aggregate": validation["aggregate"],
        "ensemble_metrics": validation["ensemble_metrics"],
        "pairwise_prediction_agreement": validation[
            "pairwise_prediction_agreement"
        ],
        "road_split_structure": validation["road_split_structure"],
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def b5c_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing B5c validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("B5c validation did not pass")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("B5c artifact registry is missing")
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing B5c artifact: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    summary = validation.get("summary")
    if not isinstance(summary, dict) or int(summary.get("run_count", 0)) != 3:
        raise ValueError("B5c must contain exactly three road-grouped runs")
    return {
        "stage": "B5c",
        "status": "complete",
        "protocol": "three-fold road-grouped cross-validation",
        "summary": summary,
        "checks": validation["checks"],
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def c1a_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing C1a validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("C1a validation did not pass")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("C1a artifact registry is missing")
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing C1a artifact: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    audit = read_json(Path(str(artifacts["audit"])))
    if not isinstance(audit, dict) or len(audit.get("profiles", [])) < 1:
        raise ValueError("C1a audit profiles are missing")
    return {
        "stage": "C1a",
        "status": "complete",
        "scope": "full 19-class inventory and road coverage audit",
        "summary": validation["summary"],
        "profiles": audit["profiles"],
        "checks": validation["checks"],
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def c1b_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing C1b validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("C1b validation did not pass")

    checks = validation.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("C1b validation checks are missing")
    if checks.get("assets_exported") is not False:
        raise ValueError("C1b must not export physical assets")
    if checks.get("training_started") is not False:
        raise ValueError("C1b must not start training")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("C1b artifact registry is missing")
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing C1b artifact: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    summary = validation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("C1b validation summary is missing")
    if int(summary.get("shared_asset_count", 0)) != 17134:
        raise ValueError("C1b shared asset count is not 17,134")
    if int(summary.get("road_class_count", 0)) != 16:
        raise ValueError("C1b road-domain class count is not 16")
    return {
        "stage": "C1b",
        "status": "complete",
        "scope": "shared 19-class benchmark and 16-class road-domain manifests",
        "summary": summary,
        "checks": checks,
        "review_status": validation.get("review_status", {}),
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def c1c_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing C1c validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("C1c validation did not pass")
    checks = validation.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("C1c validation checks are missing")
    if checks.get("full_export_not_started") is not True:
        raise ValueError("C1c unexpectedly started a full export")
    if checks.get("training_not_started") is not True:
        raise ValueError("C1c unexpectedly started training")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("C1c artifact registry is missing")
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing C1c artifact: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    summary = validation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("C1c validation summary is missing")
    if int(summary.get("sample_count", 0)) != 19:
        raise ValueError("C1c smoke set must contain 19 samples")
    if int(summary.get("reused_asset_count", 0)) <= 0:
        raise ValueError("C1c did not prove asset reuse after resume")
    return {
        "stage": "C1c",
        "status": "complete",
        "scope": "resumable 19-class shared-asset smoke export",
        "summary": summary,
        "checks": checks,
        "review_status": validation.get("review_status", {}),
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def c1_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing C1 validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("C1 validation did not pass")
    checks = validation.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("C1 validation checks are missing")
    if checks.get("no_training_artifacts") is not True:
        raise ValueError("C1 unexpectedly created training artifacts")
    if checks.get("codex_visual_precheck_complete") is not True:
        raise ValueError("C1 Codex visual precheck is incomplete")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("C1 artifact registry is missing")
    for name, raw_path in artifacts.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing C1 artifact: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    summary = validation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("C1 validation summary is missing")
    if int(summary.get("sample_count", 0)) != 17134:
        raise ValueError("C1 full dataset must contain 17,134 samples")
    if int(summary.get("trajectory_count", 0)) != 110:
        raise ValueError("C1 full dataset must contain 110 trajectories")
    return {
        "stage": "C1",
        "status": "complete",
        "scope": "full 19-class shared multimodal dataset export",
        "summary": summary,
        "checks": checks,
        "review_status": validation.get("review_status", {}),
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def c2a_entry(validation_path: Path) -> dict[str, object]:
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing C2a validation: {validation_path}")
    validation = read_json(validation_path)
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("C2a validation did not pass")
    checks = validation.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("C2a validation checks are missing")
    if checks.get("formal_training_not_started") is not True:
        raise ValueError("C2a unexpectedly started formal training")
    if checks.get("model_checkpoint_not_written") is not True:
        raise ValueError("C2a unexpectedly wrote a model checkpoint")
    if checks.get("optimizer_step_succeeded") is not True:
        raise ValueError("C2a GPU optimizer step did not pass")

    artifact_entries: dict[str, object] = {
        "validation": {
            "path": str(validation_path.resolve()),
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
        }
    }
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("C2a artifact registry is missing")
    for name, entry in artifacts.items():
        path = Path(str(entry["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing C2a artifact: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"C2a artifact hash mismatch: {path}")
        artifact_entries[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": entry["sha256"],
        }

    summary = validation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("C2a validation summary is missing")
    if int(summary.get("sample_count", 0)) != 17134:
        raise ValueError("C2a cache must contain 17,134 samples")
    if int(summary.get("class_count", 0)) != 19:
        raise ValueError("C2a smoke must use 19 classes")
    return {
        "stage": "C2a",
        "status": "complete",
        "scope": validation["scope"],
        "summary": summary,
        "checks": checks,
        "validation_status": validation["status"],
        "artifacts": artifact_entries,
    }


def c2_entry(
    run_dir: Path,
    validation_path: Path,
    report_path: Path,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing C2 run configuration: {config_path}")
    config = read_json(config_path)
    live_path = run_dir / "live_progress.json"
    history_path = run_dir / "training_history.json"
    run_state_path = run_dir / "run_state.json"
    launcher_state_path = run_dir / "launcher_state.json"
    live = read_json(live_path) if live_path.is_file() else {}
    history = read_json(history_path) if history_path.is_file() else []
    run_state = read_json(run_state_path) if run_state_path.is_file() else {}
    launcher_state = (
        read_json(launcher_state_path) if launcher_state_path.is_file() else {}
    )
    if not isinstance(config, dict):
        raise ValueError("C2 run configuration is invalid")
    if not isinstance(live, dict) or not isinstance(history, list):
        raise ValueError("C2 live progress or history is invalid")
    final_path = run_dir / "final_metrics.json"
    if final_path.is_file():
        if not validation_path.is_file():
            raise FileNotFoundError(f"Missing C2 validation: {validation_path}")
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing C2 stage report: {report_path}")
        final = read_json(final_path)
        validation = read_json(validation_path)
        if not isinstance(final, dict) or final.get("status") != "complete":
            raise ValueError("C2 final metrics are invalid or incomplete")
        if (
            not isinstance(validation, dict)
            or validation.get("status") != "passed"
            or Path(str(validation.get("run_dir", ""))).resolve() != run_dir
        ):
            raise ValueError("C2 validation did not pass for this run")
        checks = validation.get("checks")
        if not isinstance(checks, dict) or not all(checks.values()):
            raise ValueError("C2 validation checks are incomplete")
        artifacts = validation.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("C2 validation artifact registry is missing")
        artifact_entries: dict[str, object] = {
            "validation": {
                "path": str(validation_path.resolve()),
                "bytes": validation_path.stat().st_size,
                "sha256": sha256_file(validation_path),
            },
            "stage_report": {
                "path": str(report_path.resolve()),
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            },
        }
        for name, entry in artifacts.items():
            path = Path(str(entry["path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing C2 artifact: {path}")
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise ValueError(f"C2 artifact hash mismatch: {path}")
            artifact_entries[str(name)] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        return {
            "stage": "C2",
            "status": "complete",
            "run_dir": str(run_dir),
            "model": config["model"],
            "configuration": config["args"],
            "split_sizes": config["split_sizes"],
            "parameter_count": config["parameter_count"],
            "test_policy": config["test_policy"],
            "class_names": config["class_names"],
            "best_epoch": int(final["best_epoch"]),
            "epochs_completed": int(final["epochs_completed"]),
            "metrics": final["metrics"],
            "summary": validation["summary"],
            "checks": checks,
            "warnings": validation.get("warnings", []),
            "validation_status": validation["status"],
            "artifacts": artifact_entries,
        }
    return {
        "stage": "C2",
        "status": "in_progress",
        "run_dir": str(run_dir),
        "model": config["model"],
        "configuration": config["args"],
        "split_sizes": config["split_sizes"],
        "parameter_count": config["parameter_count"],
        "test_policy": config["test_policy"],
        "live_progress_snapshot": live,
        "completed_epochs": len(history),
        "run_state_snapshot": run_state,
        "launcher_state_snapshot": launcher_state,
        "artifacts": {
            "run_config": {
                "path": str(config_path),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            "live_progress_path": str(live_path),
            "history_path": str(history_path),
            "last_checkpoint_path": str(run_dir / "last.pt"),
            "best_checkpoint_path": str(run_dir / "best.pt"),
        },
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_status(registry: dict[str, object], comparison: dict[str, object]) -> str:
    experiments = registry["experiments"]
    b2 = experiments["B2"]
    b3 = experiments["B3"]
    stages = registry["stage_gates"]
    current_gate = next(
        (row for row in stages if "进行中" in row["status"]),
        next((row for row in stages if "待批准" in row["status"]), None),
    )
    completed_count = sum(row["status"].startswith("已完成") for row in stages)
    total_count = len(stages)

    comparison_rows = []
    for split in ("val", "test"):
        item = comparison["by_split"][split]
        comparison_rows.append(
            "| "
            + " | ".join(
                (
                    split,
                    str(item["sample_count"]),
                    str(item["both_correct"]),
                    str(item["left_only_correct"]),
                    str(item["right_only_correct"]),
                    str(item["neither_correct"]),
                    percent(item["either_model_oracle_accuracy"]),
                )
            )
            + " |"
        )

    stage_rows = "\n".join(
        f"| {row['stage']} | {row['scope']} | {row['status']} |" for row in stages
    )
    current_gate_text = (
        f"{current_gate['stage']}：{current_gate['status']}"
        if current_gate
        else "没有待批准阶段"
    )
    b4a = experiments.get("B4a")
    b4 = experiments.get("B4")
    b4c = experiments.get("B4c")
    b5 = experiments.get("B5")
    b5b = experiments.get("B5b")
    b5c = experiments.get("B5c")
    c1a = experiments.get("C1a")
    c1b = experiments.get("C1b")
    c1c = experiments.get("C1c")
    c1 = experiments.get("C1")
    c2a = experiments.get("C2a")
    c2 = experiments.get("C2")
    b4_conclusion = (
        (
            "- B4c 已完成测试前锁定配置的三种子正则化选择和模态消融；"
            "fusion 测试 macro F1 均值为 "
            f'{percent(b4c["suite_metrics"]["fusion"]["test_macro_f1_mean"])}。'
        )
        if b4c
        else (
        "- B4b 已完成冻结主干的三类 TSCMDL 融合训练、完整评估和独立验收。"
        if b4
        else (
        "- B4a 已完成冻结主干融合接口、梯度、显存和断点恢复冒烟；"
        "B4b 正式融合训练尚未开始。"
        if b4a
        else "- B4 融合训练尚未完成。"
        )
        )
    )
    b4a_artifact = (
        f'- B4a 冒烟运行：`{b4a["run_dir"]}`\n'
        "- B4a 验收：`reports/B4a_validation.json`\n"
        if b4a
        else ""
    )
    b4_artifact = (
        f'- B4 正式运行：`{b4["run_dir"]}`\n'
        "- B4 验收：`reports/B4_validation.json`\n"
        "- B2/B3/B4 对比：`reports/B2_B3_B4_prediction_comparison.json`\n"
        if b4
        else ""
    )
    b4c_artifact = (
        f'- B4c 三种子消融：`{b4c["run_dir"]}`\n'
        "- B4c 验收：`reports/B4c_validation.json`\n"
        if b4c
        else ""
    )
    b5_artifact = (
        f'- B5 PTv2 正式运行：`{b5["run_dir"]}`\n'
        "- B5 验收：`reports/B5_validation.json`\n"
        "- B2/B3/B4/B5 对比：`reports/B2_B3_B4_B5_prediction_comparison.json`\n"
        if b5
        else ""
    )
    b5b_artifact = (
        "- B5b 三种子复验：`reports/B5b_validation.json`\n"
        "- B5b 逐样本诊断：`reports/B5b_sample_diagnostics.csv`\n"
        if b5b
        else ""
    )
    b5c_artifact = (
        "- B5c 道路互斥协议："
        f'`{b5c["artifacts"]["protocol"]["path"]}`\n'
        "- B5c 机器验收："
        f'`{b5c["artifacts"]["validation"]["path"]}`\n'
        if b5c
        else ""
    )
    c1a_artifact = (
        "- C1a 完整 19 类清单："
        f'`{c1a["artifacts"]["inventory"]["path"]}`\n'
        "- C1a 道路覆盖审计："
        f'`{c1a["artifacts"]["audit"]["path"]}`\n'
        "- C1a 机器验收："
        f'`{c1a["artifacts"]["validation"]["path"]}`\n'
        if c1a
        else ""
    )
    c1b_artifact = (
        "- C1b 双轨规划："
        f'`{c1b["artifacts"]["plan"]["path"]}`\n'
        "- C1b 磁盘预算："
        f'`{c1b["artifacts"]["disk_budget"]["path"]}`\n'
        "- C1b 机器验收："
        f'`{c1b["artifacts"]["validation"]["path"]}`\n'
        if c1b
        else ""
    )
    c1c_artifact = (
        "- C1c 冒烟 manifest："
        f'`{c1c["artifacts"]["manifest"]["path"]}`\n'
        "- C1c 视觉总览："
        f'`{c1c["artifacts"]["contact_sheet"]["path"]}`\n'
        "- C1c 机器验收："
        f'`{c1c["artifacts"]["validation"]["path"]}`\n'
        if c1c
        else ""
    )
    c1_artifact = (
        "- C1 全量共享 manifest："
        f'`{c1["artifacts"]["manifest"]["path"]}`\n'
        "- C1 19 类基准 manifest："
        f'`{c1["artifacts"]["benchmark_manifest"]["path"]}`\n'
        "- C1 16 类道路 manifest："
        f'`{c1["artifacts"]["road_manifest"]["path"]}`\n'
        "- C1 全量验收："
        f'`{c1["artifacts"]["validation"]["path"]}`\n'
        if c1
        else ""
    )
    c2a_artifact = (
        "- C2a 全量 kNN 缓存验收："
        f'`{c2a["artifacts"]["cache_validation"]["path"]}`\n'
        "- C2a GPU 单批次结果："
        f'`{c2a["artifacts"]["gpu_smoke_result"]["path"]}`\n'
        "- C2a 阶段验收："
        f'`{c2a["artifacts"]["validation"]["path"]}`\n'
        if c2a
        else ""
    )
    c2_artifact = ""
    if c2:
        if c2["status"] == "complete":
            c2_artifact = (
                f'- C2 正式运行：`{c2["run_dir"]}`\n'
                "- C2 最佳检查点："
                f'`{c2["artifacts"]["best_pt"]["path"]}`\n'
                "- C2/C2b 机器验收："
                f'`{c2["artifacts"]["validation"]["path"]}`\n'
                "- C2b 类别诊断："
                f'`{c2["artifacts"]["C2_diagnostics_json"]["path"]}`\n'
                "- C2b 阶段报告："
                f'`{c2["artifacts"]["stage_report"]["path"]}`\n'
            )
        else:
            c2_artifact = (
                f'- C2 正式运行：`{c2["run_dir"]}`\n'
                "- C2 实时进度："
                f'`{c2["artifacts"]["live_progress_path"]}`\n'
                "- C2 最近检查点："
                f'`{c2["artifacts"]["last_checkpoint_path"]}`\n'
            )
    b4_result_row = (
        f'| B4 | {b4["model"]} | {b4["best_epoch"]} | '
        f'{percent(b4["metrics"]["val"]["accuracy"])} | '
        f'{percent(b4["metrics"]["val"]["macro_f1"])} | '
        f'{percent(b4["metrics"]["test"]["accuracy"])} | '
        f'{percent(b4["metrics"]["test"]["macro_f1"])} |'
        if b4
        else ""
    )
    b4c_result_row = (
        f'| B4c | {b4c["model"]} | {b4c["best_epoch"]} | '
        f'{percent(b4c["metrics"]["val"]["accuracy"])} | '
        f'{percent(b4c["metrics"]["val"]["macro_f1"])} | '
        f'{percent(b4c["metrics"]["test"]["accuracy"])} | '
        f'{percent(b4c["metrics"]["test"]["macro_f1"])} |'
        if b4c
        else ""
    )
    b5_result_row = (
        f'| B5 | {b5["model"]} | {b5["best_epoch"]} | '
        f'{percent(b5["metrics"]["val"]["accuracy"])} | '
        f'{percent(b5["metrics"]["val"]["macro_f1"])} | '
        f'{percent(b5["metrics"]["test"]["accuracy"])} | '
        f'{percent(b5["metrics"]["test"]["macro_f1"])} |'
        if b5
        else ""
    )
    c2_result_row = (
        f'| C2 | {c2["model"]} | {c2["best_epoch"]} | '
        f'{percent(c2["metrics"]["val"]["accuracy"])} | '
        f'{percent(c2["metrics"]["val"]["macro_f1"])} | '
        f'{percent(c2["metrics"]["test"]["accuracy"])} | '
        f'{percent(c2["metrics"]["test"]["macro_f1"])} |'
        if c2 and c2["status"] == "complete"
        else ""
    )
    b5b_summary = (
        (
            "## B5b 三种子复验\n\n"
            "| split | accuracy mean ± std | macro F1 mean ± std | "
            "probability ensemble accuracy | ensemble macro F1 |\n"
            "|---|---:|---:|---:|---:|\n"
            f'| val | {percent(b5b["aggregate"]["val"]["accuracy"]["mean"])} ± '
            f'{percent(b5b["aggregate"]["val"]["accuracy"]["std"])} | '
            f'{percent(b5b["aggregate"]["val"]["macro_f1"]["mean"])} ± '
            f'{percent(b5b["aggregate"]["val"]["macro_f1"]["std"])} | '
            f'{percent(b5b["ensemble_metrics"]["val"]["accuracy"])} | '
            f'{percent(b5b["ensemble_metrics"]["val"]["macro_f1"])} |\n'
            f'| test | {percent(b5b["aggregate"]["test"]["accuracy"]["mean"])} ± '
            f'{percent(b5b["aggregate"]["test"]["accuracy"]["std"])} | '
            f'{percent(b5b["aggregate"]["test"]["macro_f1"]["mean"])} ± '
            f'{percent(b5b["aggregate"]["test"]["macro_f1"]["std"])} | '
            f'{percent(b5b["ensemble_metrics"]["test"]["accuracy"])} | '
            f'{percent(b5b["ensemble_metrics"]["test"]["macro_f1"])} |\n\n'
            "固定划分中，验证集 90/90 样本来自训练未见道路；测试集 "
            "89/90 样本来自训练已见道路。因此测试结果不能视为严格道路域外泛化。\n"
        )
        if b5b
        else ""
    )
    b5c_summary = (
        (
            "## B5c 道路分组互斥协议\n\n"
            f'- 使用 {b5c["summary"]["development_road_count"]} 条道路、'
            f'{b5c["summary"]["development_sample_count"]} 株非官方测试样本'
            "建立三折道路分组交叉验证。\n"
            f'- {b5c["summary"]["official_test_withheld_count"]} 株官方参考'
            "测试样本未进入交叉验证。\n"
            "- 每轮 train/val/test 道路交集为 0；验证和道路域外测试"
            f'各固定 {b5c["summary"]["balanced_test_samples_per_run"]} 株'
            "平衡样本。\n"
            "- 本阶段只完成协议与清单，尚未按新协议重新训练模型。\n"
        )
        if b5c
        else ""
    )
    c1a_summary = (
        (
            "## C1a 完整 19 类清单与道路覆盖\n\n"
            f'- 完成 {c1a["summary"]["trajectory_count"]}/'
            f'{c1a["summary"]["trajectory_count"]} 条轨迹、'
            f'{c1a["summary"]["instance_count"]} 株、19 类实例清单。\n'
            f'- 开发池 {c1a["summary"]["development_instance_count"]} 株，'
            f'官方参考测试池 {c1a["summary"]["official_test_instance_count"]} 株。\n'
            "- 严格三折、每折每类 30 株可联合评估 13 类；"
            "三折、每折每类 20 株可联合评估 16 类。\n"
            "- 完整 19 类只有在两折、每折每类 10 株时全部满足道路分组约束；"
            "该方案不建议作为主要道路域外结论。\n"
            "- 本阶段未导出点云/图像，未启动训练。\n"
        )
        if c1a
        else ""
    )
    c1b_summary = (
        (
            "## C1b 双轨清单与磁盘预算\n\n"
            f'- 19 类基准轨道共 {c1b["summary"]["shared_asset_count"]} 株：'
            f'train {c1b["summary"]["benchmark_split_counts"]["train"]}、'
            f'val {c1b["summary"]["benchmark_split_counts"]["val"]}、'
            f'test {c1b["summary"]["benchmark_split_counts"]["test"]}。\n'
            f'- 16 类道路域外轨道共 {c1b["summary"]["road_run_count"]} 轮，'
            f'每轮覆盖 {c1b["summary"]["road_sample_count_per_run"]} 株开发样本，'
            "train/val/test 道路交集均为 0。\n"
            "- 两条轨道引用同一份共享点云与图像路径，预计重复资产为 0 字节。\n"
            f'- 共享资产 P95 加 25% 安全余量约 '
            f'{c1b["summary"]["shared_assets_p95_with_safety_bytes"] / (1024 ** 3):.2f} GiB，'
            f'规划时可用空间约 {c1b["summary"]["free_disk_bytes"] / (1024 ** 3):.1f} GiB。\n'
            "- 本阶段未导出点云/图像，未启动训练。\n"
        )
        if c1b
        else ""
    )
    c1c_summary = (
        (
            "## C1c 共享资产导出器冒烟验收\n\n"
            f'- 从 {c1c["summary"]["trajectory_count"]} 条开发轨迹导出 '
            f'{c1c["summary"]["sample_count"]} 株，19 类各 1 株。\n'
            f'- 通过两次运行实测断点续传，第二次核验并复用 '
            f'{c1c["summary"]["reused_asset_count"]} 对既有资产。\n'
            f'- 最低投影裁剪可见率 '
            f'{percent(c1c["summary"]["minimum_crop_visible_fraction"])}；'
            "19 张已完成 Codex 视觉预检，无严重投影失败；用户复核待完成。\n"
            f'- 冒烟目录约 '
            f'{c1c["summary"]["output_directory_bytes"] / (1024 ** 2):.2f} MiB，'
            "未启动全量导出或训练。\n"
        )
        if c1c
        else ""
    )
    c1_summary = (
        (
            "## C1 完整 19 类共享数据集\n\n"
            f'- 完成 {c1["summary"]["trajectory_count"]}/110 条轨迹、'
            f'{c1["summary"]["sample_count"]} 株、19 类点云与图像导出。\n'
            f'- 基准划分为 train '
            f'{c1["summary"]["benchmark_split_histogram"]["train"]}、val '
            f'{c1["summary"]["benchmark_split_histogram"]["val"]}、test '
            f'{c1["summary"]["benchmark_split_histogram"]["test"]}；'
            f'16 类道路清单共 {c1["summary"]["road_manifest_row_count"]} 行。\n'
            f'- 共享资产约 '
            f'{c1["summary"]["shared_asset_bytes"] / (1024 ** 3):.3f} GiB，'
            f'完整目录约 '
            f'{c1["summary"]["output_directory_bytes"] / (1024 ** 3):.3f} GiB。\n'
            "- 34,268 个物理资产逐一通过格式和哈希验收；38 张分层预览"
            "已完成 Codex 视觉预检，无严重投影失败；用户复核待完成。\n"
            "- 本阶段未生成训练缓存或模型权重。\n"
        )
        if c1
        else ""
    )
    c2a_summary = (
        (
            "## C2a 19 类 PTv2 缓存与 GPU 冒烟\n\n"
            f'- 完成 {c2a["summary"]["sample_count"]} 株、'
            f'{c2a["summary"]["class_count"]} 类、8192 点、8 邻居的'
            "分片内存映射缓存。\n"
            f'- 缓存占用 {c2a["summary"]["cache_bytes"] / (1024 ** 3):.3f} GiB，'
            f'两次运行共 {c2a["summary"]["cache_elapsed_seconds"]:.1f} 秒；'
            "已实测部分运行后续跑。\n"
            f'- 在 {c2a["summary"]["gpu"]} 上完成 batch '
            f'{c2a["summary"]["batch_size"]} 的前向、反向和参数更新；'
            f'峰值实际分配显存 {c2a["summary"]["train_peak_allocated_mib"]:.1f} MiB。\n'
            f'- {c2a["summary"]["unit_test_count"]} 项测试通过；'
            "未写模型权重，未启动正式训练，冒烟未载入官方测试分片。\n"
        )
        if c2a
        else ""
    )
    c2_summary = ""
    if c2:
        if c2["status"] == "complete":
            val = c2["metrics"]["val"]
            test = c2["metrics"]["test"]
            c2_summary = (
                "## C2/C2b 完整 19 类 PTv2 结果收口\n\n"
                f'- 训练完成 {c2["epochs_completed"]} 轮，最佳轮次 '
                f'{c2["best_epoch"]}；模型参数量 {c2["parameter_count"]:,}。\n'
                f'- 验证集 accuracy {percent(val["accuracy"])}、macro F1 '
                f'{percent(val["macro_f1"])}；测试集 accuracy '
                f'{percent(test["accuracy"])}、macro F1 '
                f'{percent(test["macro_f1"])}。\n'
                "- 测试 accuracy 比 macro F1 高 "
                f'{100.0 * (test["accuracy"] - test["macro_f1"]):.2f} 个百分点，'
                "说明类别不均衡和弱类漏识别明显；正式比较应优先看 macro F1。\n"
                "- 训练、断点、预测、概率、混淆矩阵和可读诊断图均已"
                "通过独立验收；后续推理使用验证集选出的 `best.pt`。\n"
            )
        else:
            live = c2["live_progress_snapshot"]
            current_loss = live.get("loss")
            metric_text = (
                f'当前累计 loss {float(current_loss):.4f}、'
                f'accuracy {percent(float(live["accuracy"]))}、'
                f'macro F1 {percent(float(live["macro_f1"]))}。'
                if current_loss is not None
                else "当前仍在初始化数据与模型。"
            )
            c2_summary = (
                "## C2 完整 19 类 PTv2 正式训练\n\n"
                f'- 状态：进行中；运行目录 `{c2["run_dir"]}`。\n'
                f'- 当前 epoch {int(live.get("epoch", 0) or 0)}/'
                f'{int(c2["configuration"]["epochs"])}，'
                f'阶段 {live.get("phase", "initializing")}，batch '
                f'{int(live.get("batch", 0) or 0)}/'
                f'{int(live.get("total_batches", 0) or 0)}。\n'
                f'- {metric_text}\n'
                "- 每 20 个训练 batch 刷新实时指标；每轮原子保存历史和"
                "断点，正式测试只在最佳验证模型锁定后执行。\n"
            )
    c2_complete = bool(c2 and c2["status"] == "complete")
    c2_conclusion = (
        "- C2 完整 19 类 PTv2 已训练完成，C2b 已完成回传文件、预测、"
        "概率、检查点和类别诊断验收。"
        if c2_complete
        else (
            "- C2 已启动完整 19 类 PTv2 正式训练，当前由实时监控和"
            "原子检查点保护。"
            if c2
            else ""
        )
    )
    remaining_baselines = (
        "- 完整 19 类图像/融合基线、PTv2 多随机种子和自动单树分割"
        "尚未完成。"
        if c2_complete
        else "- 完整 19 类正式训练、其余基线多随机种子和自动单树分割尚未完成。"
    )
    next_recommendation = (
        "C2/C2b 已结束。下一阶段建议执行 C3 三随机种子复验和论文指标"
        "对照；C3 当前仅为待批准状态，不会自动启动。"
        if c2_complete
        else (
            "C2 正式训练正在运行。当前只跟踪训练与验证，不启动 C3；"
            "可运行 `monitor-c2-training.bat` 查看 batch 级实时进度。"
            if c2
            else (
                "C2a 已结束。建议下一步执行 C2“完整 19 类基线训练”；"
                "C2 尚未获得授权，不得自动启动。"
            )
        )
    )
    return f"""# TSCMDL WHU-STree 当前状态

更新时间：{registry["generated_at"]}

> 本文件由 `scripts/consolidate_project_status.py` 生成，是跨对话同步时的
> 当前状态入口。阶段授权仍以 `STAGE_GATES.md` 为准，请勿手工修改本文件。

## 当前结论

- 已完成阶段：{completed_count}/{total_count}。
- 当前阶段门：{current_gate_text}。
- 三类实验已经完成 B1 数据、B2 PointMLP、B3 ResNet50、B5 PTv2 基线、B5b 三种子复验和 B5c 道路互斥协议；C1a/C1b/C1c/C1 已完成完整清单、双轨规划、冒烟和全量共享资产验收。
- C2a 已完成 19 类 PTv2 全量 kNN 缓存、断点续跑和 batch 16 GPU 前后向验收。
{c2_conclusion}
{b4_conclusion}
{remaining_baselines}
- 当前分类数据由 WHU-STree 真实 `tree ID` 提取，不是自动分割模型的输出。

## 基线结果

| 阶段 | 模型 | 最佳轮次 | val accuracy | val macro F1 | test accuracy | test macro F1 |
|---|---|---:|---:|---:|---:|---:|
| B2 | {b2["model"]} | {b2["best_epoch"]} | {percent(b2["metrics"]["val"]["accuracy"])} | {percent(b2["metrics"]["val"]["macro_f1"])} | {percent(b2["metrics"]["test"]["accuracy"])} | {percent(b2["metrics"]["test"]["macro_f1"])} |
| B3 | {b3["model"]} | {b3["best_epoch"]} | {percent(b3["metrics"]["val"]["accuracy"])} | {percent(b3["metrics"]["val"]["macro_f1"])} | {percent(b3["metrics"]["test"]["accuracy"])} | {percent(b3["metrics"]["test"]["macro_f1"])} |
{b4_result_row}
{b4c_result_row}
{b5_result_row}
{c2_result_row}

{b5b_summary}
{b5c_summary}
{c1a_summary}
{c1b_summary}
{c1c_summary}
{c1_summary}
{c2a_summary}
{c2_summary}
## B2/B3 逐样本互补性

`B2 only` 表示只有点云模型正确，`B3 only` 表示只有图像模型正确。
`either oracle` 是理想选择器在两个模型至少一个正确时选对的上限，不是可部署指标。

| split | 样本 | 两者都对 | B2 only | B3 only | 两者都错 | either oracle |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

完整逐类别统计见 `reports/B2_B3_prediction_comparison.json`。

## 阶段门

| 阶段 | 范围 | 状态 |
|---|---|---|
{stage_rows}

## 权威产物

- 数据集：`{registry["dataset"]["root"]}`
- B2 正式运行：`{b2["run_dir"]}`
- B3 正式运行：`{b3["run_dir"]}`
{b4a_artifact}{b4_artifact}{b4c_artifact}{b5_artifact}{b5b_artifact}{b5c_artifact}{c1a_artifact}{c1b_artifact}{c1c_artifact}{c1_artifact}{c2a_artifact}{c2_artifact}- 实验登记：`experiment_registry.json`
- 环境锁：`environment/whu-tscmdl-environment.yml`
- B2 报告：`reports/B2_PointMLP三类训练_阶段报告.md`
- B3 报告：`reports/B3_ResNet50三类图像训练_阶段报告.md`
- B4 报告：`reports/B4_TSCMDL三类融合训练_阶段报告.md`
- B4c 报告：`reports/B4c_融合正则化与模态消融_阶段报告.md`
- B5 报告：`reports/B5_PTv2三类点云训练_阶段报告.md`
- B5b 报告：`reports/B5b_PTv2三种子与道路域诊断_阶段报告.md`
- B5c 报告：`reports/B5c_道路分组互斥划分协议_阶段报告.md`
- C1a 报告：`reports/C1a_完整19类清单与道路覆盖审计_阶段报告.md`
- C1b 报告：`reports/C1b_双轨清单与磁盘预算_阶段报告.md`
- C1c 报告：`reports/C1c_共享资产导出器冒烟验收_阶段报告.md`
- C1 报告：`reports/C1_完整19类共享资产全量导出_阶段报告.md`
- C2a 报告：`reports/C2a_19类PTv2缓存与GPU冒烟_阶段报告.md`
- C2b 报告：`reports/C2b_完整19类PTv2结果收口_阶段报告.md`

完整 SHA-256、配置和验收文件路径保存在 `experiment_registry.json`。

## 已知风险

- B2 的 batch size 3 与多层 BatchNorm 导致训练和验证波动较大。
- B3 验证与测试存在明显道路、视角或图像质量域差异。
- B2/B3 都只有一个随机种子。
- B5b 三种子验证 macro F1 为 75.83% ± 1.39%，测试为 94.46% ± 1.82%。
- 固定划分的验证集全部来自训练未见道路，测试集 89/90 来自训练已见道路；
  当前测试结果主要反映同道路分布性能，不是严格道路域外泛化。
- B5 是保留 PTv2 核心模块的原生 PyTorch 复现，未使用官方 Linux `pointops` 扩展。
- 图像可能包含邻树、车辆、天空和道路背景线索。
- 低点数树通过重复采样补齐，紫薇受影响最明显。
- B5c 已建立严格道路互斥协议，但尚未按该协议重训模型。
- C1a 证明完整 19 类不能统一满足严格三折道路评估；标签 17 只有 2 条开发道路。
- C1b 的 19 类轨道适合官方基准对照，但不保证道路互斥；16 类道路轨道
  不覆盖标签 12、16、17。
- C1c 的 Codex 视觉预检中有 4/19 张存在邻树、杆件或背景干扰；用户复核尚未完成；分类裁剪不能当作
  精确图像分割结果。
- C1 的 Codex 视觉预检中有 11/38 张存在遮挡、模糊或背景干扰；用户复核尚未完成。
- C2a 证明 PTv2 batch 16 能运行，但单批次损失不代表收敛性或精度。
{('- C2 测试 accuracy 79.06%，但 macro F1 仅 62.29%；Sophora japonica 测试 F1 为 0，类别不均衡和弱类混淆仍是主要风险。' if c2_complete else '- 完整 19 类已导出并完成 PTv2 缓存，但 PTv2、ResNet50 和 TSCMDL 尚未按全量数据正式训练。')}
- 完整 19 类 ResNet50 和 TSCMDL 尚未正式训练。

## 下一建议

{next_recommendation}

## 重新生成

```powershell
D:\\TREE\\envs\\whu-tscmdl\\python.exe .\\scripts\\consolidate_project_status.py `
  --dataset-root "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b1_three_species\\dataset" `
  --b2-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b2_pointmlp\\run_batch3_seed20260714" `
  --b3-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b3_resnet50\\run_batch8_seed20260723" `
  --b4a-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b4a_fusion_smoke\\run_seed20260727" `
  --b4-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b4_tscmdl\\run_batch8_l2_lr001_seed20260727" `
  --b4c-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b4c_regularization_ablation\\run_3seed_20260727" `
  --b5-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b5_ptv2\\run_batch16_seed20260728" `
  --b5b-validation ".\\reports\\B5b_validation.json" `
  --b5c-validation "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\b5c_road_group_cv\\validation.json" `
  --c1a-validation "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\c1a_full_inventory\\audit\\validation.json" `
  --c1b-validation "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\c1b_dual_track_plan\\validation.json" `
  --c1c-validation "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\c1c_shared_asset_smoke\\validation.json" `
  --c1-validation "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\c1_full_shared_dataset\\validation.json" `
  --c2a-validation "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\c2a_ptv2\\validation.json" `
  --c2-run "D:\\TREE\\lidar data\\whu\\derived\\tscmdl\\c2_ptv2\\received_24gb_20260803\\run" `
  --c2-validation ".\\reports\\C2_validation.json"
```
"""


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    classes_path = dataset_root / "classes.json"
    manifest_path = dataset_root / "manifest.json"
    validation_path = dataset_root / "validation.json"
    for path in (
        classes_path,
        manifest_path,
        validation_path,
        args.environment_lock,
        args.stage_gates,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    classes = read_json(classes_path)
    manifest = read_json(manifest_path)
    dataset_validation = read_json(validation_path)
    if not isinstance(classes, list) or not isinstance(manifest, dict):
        raise ValueError("Invalid B1 classes or manifest")
    if (
        not isinstance(dataset_validation, dict)
        or dataset_validation.get("status") != "passed"
    ):
        raise ValueError("B1 dataset validation did not pass")

    b2 = run_entry("B2", "PointMLP", args.b2_run.resolve(), args.b2_validation)
    b3 = run_entry(
        "B3",
        "ImageNet-pretrained ResNet50",
        args.b3_run.resolve(),
        args.b3_validation,
    )
    if b2["class_names"] != b3["class_names"]:
        raise ValueError("B2 and B3 class names differ")
    b4a = (
        b4a_entry(args.b4a_run.resolve(), args.b4a_validation)
        if args.b4a_run is not None
        else None
    )
    if b4a is not None and b4a["class_names"] != b2["class_names"]:
        raise ValueError("B4a and baseline class names differ")
    b4 = (
        run_entry(
            "B4",
            "TSCMDL frozen-backbone fusion",
            args.b4_run.resolve(),
            args.b4_validation,
        )
        if args.b4_run is not None
        else None
    )
    if b4 is not None and b4["class_names"] != b2["class_names"]:
        raise ValueError("B4 and baseline class names differ")
    if b4 is not None:
        three_model_comparison = read_json(args.three_model_comparison)
        if (
            not isinstance(three_model_comparison, dict)
            or three_model_comparison.get("status") != "passed"
        ):
            raise ValueError("B2/B3/B4 comparison did not pass")
    b4c = (
        run_entry(
            "B4c",
            "TSCMDL selected three-seed fusion",
            args.b4c_run.resolve(),
            args.b4c_validation,
        )
        if args.b4c_run is not None
        else None
    )
    if b4c is not None:
        if b4c["class_names"] != b2["class_names"]:
            raise ValueError("B4c and baseline class names differ")
        b4c_summary = read_json(args.b4c_run / "final_summary.json")
        b4c["suite_metrics"] = {
            item["modality"]: item
            for item in b4c_summary["modality_summaries"]
        }
    b5 = (
        run_entry(
            "B5",
            "PTv2 mode-1 classification encoder reimplementation",
            args.b5_run.resolve(),
            args.b5_validation,
        )
        if args.b5_run is not None
        else None
    )
    if b5 is not None:
        if b5["class_names"] != b2["class_names"]:
            raise ValueError("B5 and baseline class names differ")
        four_model_comparison = read_json(args.four_model_comparison)
        if (
            not isinstance(four_model_comparison, dict)
            or four_model_comparison.get("status") != "passed"
        ):
            raise ValueError("B2/B3/B4/B5 comparison did not pass")
    b5b = (
        b5b_entry(args.b5b_validation)
        if args.b5b_validation.is_file()
        else None
    )
    if b5b is not None and b5b["class_names"] != b2["class_names"]:
        raise ValueError("B5b and baseline class names differ")
    b5c = (
        b5c_entry(args.b5c_validation.resolve())
        if args.b5c_validation is not None
        else None
    )
    c1a = (
        c1a_entry(args.c1a_validation.resolve())
        if args.c1a_validation is not None
        else None
    )
    c1b = (
        c1b_entry(args.c1b_validation.resolve())
        if args.c1b_validation is not None
        else None
    )
    c1c = (
        c1c_entry(args.c1c_validation.resolve())
        if args.c1c_validation is not None
        else None
    )
    c1 = (
        c1_entry(args.c1_validation.resolve())
        if args.c1_validation is not None
        else None
    )
    c2a = (
        c2a_entry(args.c2a_validation.resolve())
        if args.c2a_validation is not None
        else None
    )
    c2 = (
        c2_entry(
            args.c2_run.resolve(),
            args.c2_validation.resolve(),
            args.c2_report.resolve(),
        )
        if args.c2_run is not None
        else None
    )

    b2_records = load_prediction_records(args.b2_run / "predictions.csv")
    b3_records = load_prediction_records(args.b3_run / "predictions.csv")
    comparison = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "left": {"stage": "B2", "model": "PointMLP"},
        "right": {"stage": "B3", "model": "ImageNet-pretrained ResNet50"},
        "class_names": b2["class_names"],
        **compare_prediction_records(b2_records, b3_records, len(classes)),
    }

    stage_gates = parse_stage_gates(args.stage_gates)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    registry = {
        "schema_version": 1,
        "generated_at": generated_at,
        "authoritative_stage_gate": str(args.stage_gates.resolve()),
        "stage_gates": stage_gates,
        "dataset": {
            "stage": "B1",
            "status": "complete",
            "root": str(dataset_root),
            "sample_count": int(manifest["summary"]["sample_count"]),
            "split_histogram": manifest["summary"]["split_histogram"],
            "classes": classes,
            "validation_status": dataset_validation["status"],
            "artifacts": {
                "classes": {
                    "path": str(classes_path),
                    "sha256": sha256_file(classes_path),
                },
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "validation": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
                "stage_validation": {
                    "path": str(args.b1_validation.resolve()),
                    "sha256": sha256_file(args.b1_validation),
                },
            },
        },
        "experiments": {
            "B2": b2,
            "B3": b3,
            **({"B4a": b4a} if b4a is not None else {}),
            **({"B4": b4} if b4 is not None else {}),
            **({"B4c": b4c} if b4c is not None else {}),
            **({"B5": b5} if b5 is not None else {}),
            **({"B5b": b5b} if b5b is not None else {}),
            **({"B5c": b5c} if b5c is not None else {}),
            **({"C1a": c1a} if c1a is not None else {}),
            **({"C1b": c1b} if c1b is not None else {}),
            **({"C1c": c1c} if c1c is not None else {}),
            **({"C1": c1} if c1 is not None else {}),
            **({"C2a": c2a} if c2a is not None else {}),
            **({"C2": c2} if c2 is not None else {}),
        },
        "comparison": {
            "path": str(args.comparison_output.resolve()),
            "left": comparison["left"],
            "right": comparison["right"],
            **(
                {
                    "three_model_path": str(
                        args.three_model_comparison.resolve()
                    ),
                    "three_model_sha256": sha256_file(
                        args.three_model_comparison
                    ),
                }
                if b4 is not None
                else {}
            ),
            **(
                {
                    "four_model_path": str(args.four_model_comparison.resolve()),
                    "four_model_sha256": sha256_file(args.four_model_comparison),
                }
                if b5 is not None
                else {}
            ),
        },
        "environment": {
            "name": "whu-tscmdl",
            "prefix": r"D:\TREE\envs\whu-tscmdl",
            "lock_path": str(args.environment_lock.resolve()),
            "sha256": sha256_file(args.environment_lock),
            "explicit_export_available": False,
            "explicit_export_note": (
                "Conda explicit export cannot represent the pip/external packages "
                "in this environment; the environment-yaml export includes them."
            ),
        },
        "known_unfinished": [
            "full 19-class image and fusion baselines",
            "full 19-class PTv2 three-seed repeatability study",
            "non-PTv2 baseline three-seed repeatability study",
            "road-grouped three-fold model training on the B5c protocol",
            "automatic individual-tree segmentation baselines",
        ],
    }

    atomic_json(args.comparison_output, comparison)
    atomic_json(args.registry_output, registry)
    atomic_text(args.status_output, render_status(registry, comparison))
    print(
        json.dumps(
            {
                "status": "passed",
                "registry": str(args.registry_output.resolve()),
                "comparison": str(args.comparison_output.resolve()),
                "current_status": str(args.status_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

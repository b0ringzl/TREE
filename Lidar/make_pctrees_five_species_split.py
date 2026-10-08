import argparse
import json
from pathlib import Path

import pandas as pd

TREE_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=TREE_ROOT / "lidar data" / "labels_pctrees_train.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--species", type=int, default=5)
    parser.add_argument("--per-species", type=int, default=105)
    parser.add_argument("--train-per-species", type=int, default=100)
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.labels)
    df = df[df["num_points"] >= args.min_points].copy()
    counts = df["label"].value_counts()
    eligible = counts[counts >= args.per_species].index.to_series()
    selected = eligible.sample(n=args.species, random_state=args.seed).tolist()

    train_parts = []
    val_parts = []
    for i, label in enumerate(selected):
        sampled = df[df["label"] == label].sample(
            n=args.per_species, random_state=args.seed + i
        ).copy().reset_index(drop=True)
        train = sampled.iloc[: args.train_per_species].copy()
        val = sampled.iloc[args.train_per_species :].copy()
        train["split"] = "train"
        val["split"] = "val"
        train_parts.append(train)
        val_parts.append(val)

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1, random_state=args.seed + 100
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(
        frac=1, random_state=args.seed + 200
    ).reset_index(drop=True)
    sample = pd.concat([train_df, val_df], ignore_index=True)

    label_path = args.output_dir / "labels_5species_525_split.csv"
    sample.to_csv(label_path, index=False)

    summary = {
        "seed": args.seed,
        "eligible_species_count": int(len(eligible)),
        "selected_species": selected,
        "train_count": int((sample["split"] == "train").sum()),
        "val_count": int((sample["split"] == "val").sum()),
        "class_count": int(sample["label"].nunique()),
        "epochs": args.epochs,
        "output_channels": args.species,
        "label_path": str(label_path),
        "train_counts": train_df["label"].value_counts().to_dict(),
        "val_counts": val_df["label"].value_counts().to_dict(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

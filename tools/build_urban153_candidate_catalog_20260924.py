"""Create an experimental 153-name taxonomy map without changing production.

This is NOT a trained classifier or a claim that the new species are recognized.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/labels.json"
SPLITS = {
    "Araucaria columnaris(Araucaria heterophylla)": ["Araucaria columnaris", "Araucaria heterophylla"],
    "Ficus microcarpa / Ficus benjamina (榕樹)": ["Ficus benjamina", "Ficus microcarpa"],
}
ADDITIONS = ["Michelia chapensis", "Tabebuia rosea", "Yulania x soulangeana"]
OFFICIAL_NAMES = {"Michelia chapensis": "樂昌含笑", "Tabebuia rosea": "紅花風鈴木",
                  "Yulania x soulangeana": "二喬木蘭", "Ficus benjamina": "垂葉榕",
                  "Ficus microcarpa": "細葉榕", "Araucaria columnaris": "柱狀南洋杉",
                  "Araucaria heterophylla": "異葉南洋杉"}


def main(out):
    base = json.loads(BASE.read_text(encoding="utf-8"))
    assert len(base) == 148 and all(name in base for name in SPLITS)
    labels = []
    old_to_new = {}
    for i, name in enumerate(base):
        children = SPLITS.get(name, [name])
        old_to_new[str(i)] = []
        for child in children:
            old_to_new[str(i)].append(len(labels))
            labels.append(child)
    for name in ADDITIONS:
        assert name not in labels
        labels.append(name)
    assert len(labels) == len(set(labels)) == 153
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "kind": "candidate_taxonomy_only_NOT_TRAINED",
        "base_labels": str(BASE.resolve()), "base_class_count": len(base),
        "candidate_class_count": len(labels), "labels": labels,
        "old_index_to_candidate_indices": old_to_new,
        "split_old_labels": SPLITS, "new_taxa": ADDITIONS,
        "official_chinese_names": OFFICIAL_NAMES,
        "synonym_review": {
            "Michelia chapensis": ["Magnolia chapensis"],
            "Yulania x soulangeana": ["Magnolia × soulangeana"],
            "Araucaria columnaris": ["NOT a synonym of Araucaria heterophylla"],
            "Ficus benjamina": ["NOT a synonym of Ficus microcarpa"],
            "Tabebuia rosea": ["NOT Tabebuia impetiginosa"],
        },
        "status": {name: "requires separately licensed, verified training and group-heldout validation"
                   for name in ADDITIONS},
        "deployed": False, "production_files_changed": False,
    }
    (out / "candidate_catalog.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(out / "candidate_catalog.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())

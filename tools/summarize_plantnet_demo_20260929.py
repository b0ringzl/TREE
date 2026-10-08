"""Validate and summarize the manually recorded paired web/UI comparison."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/plantnet_demo_comparison_20260929_v1"
V3 = ROOT / "outputs/plantnet_demo_comparison_20260929_v3"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    manifest = read_json(V3 / "manifest.json")
    demo = read_json(V3 / "demo_results_v2.json")
    if len(manifest["samples"]) != 200 or len(demo["results"]) != 200:
        raise RuntimeError("Expected 200 manifest rows and 200 Demo results")
    plantnet = read_json(V1 / "plantnet_results.json")["results"]
    plantnet += read_json(V3 / "plantnet_results_25_50.json")["results"]
    for filename in ("plantnet_results_51_100.tsv", "plantnet_results_101_150.tsv", "plantnet_results_151_200.tsv"):
        with (V3 / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                plantnet.append({"i": int(row["i"]), "kind": row["kind"],
                                 "display_score": row["score"], "heading": row["heading"]})
    plantnet.sort(key=lambda row: row["i"])
    if [row["i"] for row in plantnet] != list(range(1, 201)):
        raise RuntimeError("Pl@ntNet rows missing or duplicated")
    if any(not row["heading"] or row["kind"] not in {"species", "genus"} for row in plantnet):
        raise RuntimeError("Incomplete Pl@ntNet result")

    rows = []
    for i, (sample, local, remote) in enumerate(zip(manifest["samples"], demo["results"], plantnet), 1):
        if local["sample_id"] != sample["sample_id"] or remote["i"] != i:
            raise RuntimeError(f"Pairing mismatch at {i}")
        if local["http_status"] != 200 or not local["response"].get("top3"):
            raise RuntimeError(f"Demo result missing at {i}")
        source = sample["species"]
        demo_top = local["response"]["top3"][0]["species"]
        remote_top = remote["heading"]
        remote_match = remote["kind"] == "species" and (
            remote_top == source or remote_top.startswith(source + " ")
        )
        rows.append({"i": i, "sample_id": sample["sample_id"], "observation": sample["group_id"],
                     "source_species": source, "demo_top1": demo_top,
                     "demo_match": demo_top == source,
                     "plantnet_top1": remote_top, "plantnet_rank": remote["kind"],
                     "plantnet_display_score": remote["display_score"],
                     "plantnet_match": remote_match})
    if len({row["sample_id"] for row in rows}) != 200:
        raise RuntimeError("Repeated sample_id")
    if len({row["observation"] for row in rows}) != 196:
        raise RuntimeError("Unexpected observation-group count")

    def stats(items):
        return {
            "images": len(items), "observations": len({row["observation"] for row in items}),
            "species_covered": len({row["source_species"] for row in items}),
            "demo_match": sum(row["demo_match"] for row in items),
            "plantnet_match": sum(row["plantnet_match"] for row in items),
            "plantnet_genus_only": sum(row["plantnet_rank"] == "genus" for row in items),
            "both_match": sum(row["demo_match"] and row["plantnet_match"] for row in items),
            "demo_only": sum(row["demo_match"] and not row["plantnet_match"] for row in items),
            "plantnet_only": sum(not row["demo_match"] and row["plantnet_match"] for row in items),
            "neither": sum(not row["demo_match"] and not row["plantnet_match"] for row in items),
        }

    summary = {"method": "Exact scientific-name Top-1 agreement with iNaturalist source label; genus-only is no species answer",
               "source": "Historical regression split, not new independent blind test",
               "eligible_images": manifest["eligible_count"],
               "stats_24": stats(rows[:24]), "stats_50": stats(rows[:50]),
               "stats_100": stats(rows[:100]), "stats_196": stats(rows[:196]),
               "stats_200": stats(rows),
               "demo_wrong": [row for row in rows if not row["demo_match"]],
               "plantnet_only_cases": [row for row in rows if not row["demo_match"] and row["plantnet_match"]],
               "source_species_counts": dict(sorted(Counter(row["source_species"] for row in rows).items())),
               "rows": rows}
    write_json(V3 / "paired_results.json", summary)
    with (V3 / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in ("stats_24", "stats_50", "stats_100", "stats_196", "stats_200", "demo_wrong", "plantnet_only_cases")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

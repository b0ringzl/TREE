"""Record root-agent visual inspection of the 30 new raw downloads.

This is image-content triage, not a replacement for authoritative human review.
No original files, source manifests or review databases are changed.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_datasets/artocarpus_fresh_inat_20260917_v1'
OUT = ROOT / 'outputs/artocarpus_fresh_negative_triage_20260917_v1'

# Visually inspected raw_000.jpg through raw_002.jpg contact sheets. These
# codes describe framing and usability, not new species identifications.
NOTES = {
 734115297: ('fruit_trunk_detail', 'detail_only', 'Fruit-bearing trunk; no whole crown.'),
 734119262: ('fruit_branch_detail', 'detail_only', 'Branch/fruit context; incomplete canopy.'),
 734120193: ('trunk_only', 'isolate_target_ambiguity', 'Several trunks and little diagnostic foliage; do not treat all trunks as one species.'),
 731691843: ('partial_tree_crown', 'partial_tree_candidate', 'Upward view includes target trunk, fruit and foliage, but no root-to-top outline.'),
 730314769: ('fruit_leaf_detail', 'detail_only', 'Close-up fruit/leaf/trunk; not full tree.'),
 730314876: ('fruit_leaf_detail', 'detail_only', 'Another detail of same observation; not independent tree.'),
 728823755: ('leafy_branch_detail', 'detail_only', 'Leafy branches fill frame; tree outline absent.'),
 726458320: ('fruit_trunk_detail', 'detail_only', 'Fruit on trunk; complete crown absent.'),
 720504881: ('crown_fragment', 'crown_candidate', 'Dense foliage/crown segment, dark background; not a complete whole tree.'),
 698485471: ('mixed_understory', 'isolate_target_ambiguity', 'Small leafy plant among trunks and other vegetation; target association unclear.'),
 512233972: ('flower_leaf_detail', 'detail_only', 'Close-up branch and inflorescences.'),
 512233988: ('flower_leaf_detail', 'detail_only', 'Second view of same observation.'),
 389756608: ('flower_leaf_detail', 'detail_only', 'Close leaf/flower view.'),
 389756655: ('leaf_branch_detail', 'detail_only', 'Small branch view with background lawn.'),
 331092088: ('flower_leaf_detail', 'detail_only', 'Close view of leaf underside and flower.'),
 292040461: ('leaf_branch_detail', 'detail_only', 'Lit branch against dark background.'),
 292040861: ('flower_leaf_detail', 'detail_only', 'Night/flash branch detail from same observation.'),
 292040363: ('crown_fragment', 'isolate_low_visibility', 'Backlit canopy fragment and trunk; foliage is largely dark and overlaps.'),
 258027853: ('seedling_leaf_detail', 'detail_only', 'Small understory plant/leaf detail, not a mature whole-tree example.'),
 710373312: ('fruit_trunk_detail', 'isolate_low_visibility', 'Very dark trunk/fruit detail; coarse brightness failure.'),
 710373347: ('fruit_trunk_detail', 'isolate_low_visibility', 'Night/flash trunk and fruit; most background black.'),
 710373373: ('leaf_detail', 'isolate_low_visibility', 'Very dark leaf detail; coarse brightness failure.'),
 690077436: ('crown_context', 'crown_candidate', 'Small tree/shrub crown is visible; base and total height incomplete.'),
 682447925: ('fruit_trunk_detail', 'detail_only', 'Fruit-covered trunk with wall background; not whole tree.'),
 673684744: ('leaf_branch_detail', 'detail_only', 'Foliage detail; no whole-tree context.'),
 657081567: ('seedling_leaf_detail', 'detail_only', 'Leaves near ground/trunk; not whole-tree context.'),
 644598507: ('seedling_leaf_detail', 'detail_only', 'Leafy seedling, photographer watermark retained.'),
 721658355: ('fruit_detail', 'isolate_existing_rejection', 'Fresh identity audit located an existing explicit human rejection; never reapprove.'),
 391076524: ('fruit_branch_detail', 'isolate_existing_rejection', 'Fresh identity audit located an existing explicit human rejection; never reapprove.'),
 179903686: ('soft_flower_detail', 'isolate_existing_rejection', 'Old human rejection and visibly soft flower detail; never reapprove.'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = DATA / 'manifest.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    assert {r['photo_id'] for r in rows} == set(NOTES)
    reviewed = []
    for row in rows:
        organ, lane, note = NOTES[row['photo_id']]
        reviewed.append({**row, 'agent_content_review': {
            'authority': 'assistant_visual_framing_only_not_human_species_acceptance',
            'organ': organ, 'lane': lane, 'note': note,
            'complete_whole_tree': False,
            'contact_sheet': str(DATA / 'contact_sheets' / f"raw_{rows.index(row)//12:03d}.jpg"),
        }, 'training_eligible': False})
    (OUT / 'manifest.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'source_manifest': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'visually_screened': len(rows), 'complete_whole_tree': 0,
        'framing_lanes': dict(Counter(r['agent_content_review']['lane'] for r in reviewed)),
        'actual_human_review_decisions_changed': False, 'training_performed': False,
        'deployed_files_changed': False,
        'caution': 'Crown/partial-tree candidates still require historical identity gates and task-role assignment. Detail images do not repair lack of whole-tree data.',
    }
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()

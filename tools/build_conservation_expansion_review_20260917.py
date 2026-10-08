"""Freeze agent triage and unresolved queue; never writes human decisions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/conservation_expansion_20260917_curation_v1'


def save_new(name, value):
    path = OUT / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    checks = json.loads((OUT / 'plantnet_checks.json').read_text(encoding='utf-8'))
    checked = {}
    for check in checks['checks']:
        for image in check['inputs']:
            path = Path(image['file'])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != image['sha256']:
                raise ValueError(f'Image changed: {path}')
            checked[str(path)] = actual
    source = ROOT / 'external_datasets/conservation_expansion_20260917_v3/manifest.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    row = next(r for r in rows if r['id'] == 'cons_expand_commons_91916271')
    row.update(
        review_status='human_review_required', agent_review_status='uncertain',
        review_authority='source_plus_agent_not_human', human_review_status='pending',
        training_eligible=False, split='quarantine',
        organ='fruit_and_leaf_underside', whole_crown_visibility=False,
        agent_question='Source labels this as Artocarpus hypargyreus. Fruit and pale veined leaf undersides are visible, but the source is an individual photographer and species-level corroboration is insufficient. Can you confirm the source species? Use U if unsure. / 來源標為白桂木；可見聚合果與淺色網脈葉背，但來源是個人拍攝者，缺少充分種級交叉證據。您能確認來源樹種嗎？拿不準請按 U。',
        agent_reason='Morphology compatible, not diagnostic enough to certify species from this source alone. Pl@ntNet also disagrees at low species confidence. No inferred alternative label.',
        plantnet_reference={'scientific_name': 'Maclura tricuspidata', 'score': .058,
                            'source_url': checks['service_url'], 'checked_at': checks['checked_on']},
        source_manifest=str(source),
    )
    save_new('agent_decisions.json', [row])
    save_new('human_review_required.json', [row])
    save_new('verified_inputs.json', {'image_sha256': checked,
             'source_manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'new_images_reviewed': 1, 'new_images_agent_accepted': 0,
             'human_review_required': 1, 'historical_workflow_test_images': 3,
             'human_decisions_written': False, 'training_performed': False,
             'deployed_weights_changed': False})
    print(json.dumps({'queue': str(OUT / 'human_review_required.json'), 'uncertain': 1,
                      'historical_inputs_unchanged': len(checked)-1}, ensure_ascii=False))


if __name__ == '__main__':
    main()

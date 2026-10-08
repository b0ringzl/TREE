"""Freeze deployment gate and verify physical rollback artifacts. Never writes weights."""
import json
from pathlib import Path

from training_safety import sha256, verify_backup

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/bioclip_flora_integration_20261008_v1'


def main():
    info = verify_backup(RUN / 'pre_training_backup', verify_sources=False)
    affected = []
    for row in info['files']:
        source = Path(row['source'])
        if source.suffix in {'.pt', '.safetensors'} or source.name in {'deployment.json', 'labels.json', 'release.json'}:
            affected.append({'path': str(source), 'original_sha256': row['sha256'],
                             'current_sha256': sha256(source), 'unchanged': sha256(source) == row['sha256']})
    if not all(row['unchanged'] for row in affected):
        raise RuntimeError('Protected deployment artifact changed')
    audit = json.loads((RUN / 'audit_results.json').read_text(encoding='utf-8'))
    if not audit['summary']['pilot_acceptance_pass']:
        raise RuntimeError('Pilot did not pass its predeclared gate; do not deploy automatic scope advice')
    backup_template = next(row for row in info['files'] if Path(row['source']).name == 'index.html')
    old_html = (RUN / 'pre_training_backup' / backup_template['backup']).read_text(encoding='utf-8')
    stale_test = 'id="modelVersion">SSTv1 · 2026-09-18</p>' not in old_html
    result = {'physical_backup_verified': True, 'backup_directory': str(RUN / 'pre_training_backup'),
              'deployed_weights_unchanged': True, 'protected_artifacts': affected,
              'sst_predictions_numerically_unchanged_in_66_image_audit': audit['summary']['max_sst_probability_drift'] == 0,
              'automatic_model_switch': False, 'default_scope': 'hong_kong',
              'broad_branch_training': False, 'android_or_point_fusion_changed': False,
              'preexisting_stale_edition_version_test': stale_test,
              'acceptance': audit['summary']}
    path = RUN / 'deployment_gate.json'
    if path.exists():
        raise RuntimeError('Never overwrite a completed gate record')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'protected_artifacts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

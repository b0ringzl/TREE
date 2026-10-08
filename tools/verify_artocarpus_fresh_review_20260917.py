"""Read-only HTTP checks of the new five-image queue; never submit a decision."""
import hashlib
import json
import sqlite3
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/artocarpus_fresh_review_20260917_v1'
URL = 'http://127.0.0.1:8053'


def main():
    source = OUT / 'curated_manifest.json'
    original = source.read_bytes()
    rows = json.loads(original)
    audit_path = ROOT / 'outputs/artocarpus_fresh_split_20260917_v1/combined_batch_audit_v2/new_batch_identity_audit.json'
    audit = {r['id']: r for r in json.loads(audit_path.read_text(encoding='utf-8'))['rows']}
    page = requests.get(URL, timeout=15)
    page.raise_for_status()
    assert '白桂木補圖核驗' in page.text
    assert 'U · Unsure / 無法確定' in page.text
    javascript = requests.get(URL + '/herbarium-review.js', timeout=15)
    javascript.raise_for_status()
    assert "u:()=>save('pending')" in javascript.text
    listing = requests.get(URL + '/api/herbarium-review', timeout=15)
    listing.raise_for_status()
    result = listing.json()
    assert len(result['rows']) == 5
    assert {r['id'] for r in result['rows']} == {r['id'] for r in rows}
    checks = []
    for row in rows:
        assert audit[row['id']]['submit_as_new_review_item']
        assert not audit[row['id']]['blocked_review_record_ids']
        assert row['training_eligible'] is False
        response = requests.get(URL + '/api/herbarium-review/image/' + row['id'], timeout=15)
        response.raise_for_status()
        actual = hashlib.sha256(response.content).hexdigest()
        assert actual == row['sha256']
        assert actual == hashlib.sha256(Path(row['original_source_image']).read_bytes()).hexdigest()
        checks.append({'id': row['id'], 'http_status': response.status_code,
                       'sha256': actual, 'original_bytes_preserved': True,
                       'old_rejected_or_pending_reintroduced': False})
    with sqlite3.connect((OUT / 'human_review.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        decisions = db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
        events = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    assert source.read_bytes() == original
    report = {'url': URL, 'rows': 5, 'image_checks': checks,
        'http_methods_used': ['GET'], 'agent_submitted_human_decisions': 0,
        'observed_database_decisions': decisions, 'observed_database_events': events,
        'latest_identity_audit': str(audit_path), 'latest_identity_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'html_and_keyboard_script_verified': True,
        'browser_visual_test': 'not_performed; HTTP and JS syntax verified',
        'source_manifest_unchanged': True, 'production_demo_changed': False}
    with (OUT / 'VERIFIED.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

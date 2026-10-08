"""Create a small, separate species-confirmation queue; never accept for a user."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_datasets/artocarpus_whole_source_20260917_v1'
OUT = ROOT / 'outputs/artocarpus_fresh_review_20260917_v1'
IDS = [644765603, 167334063, 167334054, 167334042, 476475668]


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'images').mkdir()
    source = DATA / 'visual_review_manifest.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    audit_path = ROOT / 'outputs/artocarpus_fresh_split_20260917_v1/combined_batch_audit_v1/new_batch_identity_audit.json'
    audit = {r['id']: r for r in json.loads(audit_path.read_text(encoding='utf-8'))['rows']}
    selected = []
    for pid in IDS:
        row = next(r for r in rows if r['photo_id'] == pid)
        identity = audit[row['id']]
        if identity['blocked_review_record_ids']:
            raise RuntimeError('Existing unresolved/rejected human decision must not be reopened: ' + row['id'])
        src = Path(row['source_image'])
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        assert before == row['sha256']
        dest = OUT / 'images' / src.name
        shutil.copy2(src, dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == before
        organ = row['visual_qa']['organ']
        organ_tags = ['tree_or_crown_context'] if row['visual_qa']['whole_crown_relevant_candidate'] else ['foliage_or_leafy_branch']
        message = 'Can you confirm Artocarpus hypargyreus and a usable target? If uncertain, press U. / 請確認是否為白桂木且目標可辨；拿不準請按 U。 '
        message += row.get('review_note', '')
        message += ' Photo / 攝影：' + row['attribution'] + ' · ' + row['license'] + '.'
        if not identity['fresh_holdout_screen_pass']:
            message += ' Photographer overlaps older data; this is NOT a fresh independent test. / 攝影者與舊資料重合，不作新的獨立測試。'
        selected.append({**row, 'original_source_image': str(src), 'file': 'images/' + dest.name,
            'source_image': str(dest), 'source_names': ['白桂木'],
            'organ_review': {'organ_tags': organ_tags, 'visible_content': organ},
            'candidate_lane': message, 'human_review_status': 'pending',
            'review_purpose': 'species_confirmation_and_target_image_quality',
            'identity_screen': identity, 'training_eligible': False, 'split': 'quarantine_pending_human'})
    def write(name, value):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write('curated_manifest.json', selected)
    write('review_purpose.json', {
        'purpose': 'Confirm source species Artocarpus hypargyreus and usable target. Not a license grant, train assignment or fresh-test certification.',
        'A': 'User confirms white guimu species and usable image; SQLite accepted event.',
        'D': 'User rejects species or image suitability; original retained.',
        'U': 'Leave pending/uncertain and advance; never include in fitting.',
        'Z': 'Undo latest decision; append-only audit retained.',
        'selection_basis': 'Three partial tree/crown photographs plus two reasonably clear foliage views for source corroboration; excludes blurry leaf, bark-only and mixed understory.',
        'images': 5, 'observations': 3, 'fresh_identity_screen_images': 1,
        'human_accepted': 0, 'training_eligible': 0,
        'original_manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'identity_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'attribution': 'Per-photo creator and CC license are displayed in each note; watermarks and source bytes retained.',
    })
    html = (ROOT / 'tree_species_demo/static/herbarium-review.html').read_text(encoding='utf-8')
    html = html.replace('Image review / 影像驗收', 'White guimu review / 白桂木補圖核驗')
    html = html.replace('Keep useful images. Every decision is saved. / 保留可用影像，每次操作即時儲存。',
        '5 photos · 3 source observations · Species confirmation, not just sharpness. / 5 張圖、3 個來源觀察；請確認樹種，不僅是清晰度。')
    html = html.replace('A · Keep & next / 保留並下一張', 'A · Confirm species / 確認白桂木')
    html = html.replace('D / Delete · Remove & next / 剔除並下一張', 'D / Delete · Reject / 不是或不適用')
    html = html.replace('→ Skip / 略過下一張', '→ Skip / 略過下一張')
    html = html.replace('<button id="undo">', '<button id="unsure">U · Unsure / 無法確定</button><button id="undo">')
    html = html.replace('<kbd>A</kbd> 保留 / Keep', '<kbd>A</kbd> 確認樹種 / Confirm species　<kbd>U</kbd> 無法確定 / Unsure')
    html = html.replace('Photos: Hong Kong Herbarium, AFCD / 圖片來源：香港植物標本室，漁農自然護理署。',
        'Photos: individual iNaturalist contributors; creator and license shown per photo. / 圖片來自 iNaturalist 個別貢獻者，逐圖列出攝影者及授權。')
    html = html.replace('<script src="/herbarium-review.js"></script>', '<script src="/herbarium-review.js"></script>')
    js = (ROOT / 'tree_species_demo/static/herbarium-review.js').read_text(encoding='utf-8')
    js = js.replace('a:()=>save(\'accepted\'),', 'a:()=>save(\'accepted\'),u:()=>save(\'pending\'),')
    js += "\n$('unsure').onclick=()=>save('pending');\n"
    (OUT / 'review.html').write_text(html, encoding='utf-8')
    (OUT / 'review.js').write_text(js, encoding='utf-8')
    print(json.dumps({'out': str(OUT), 'count': len(selected), 'decisions_written': False}))


if __name__ == '__main__':
    main()

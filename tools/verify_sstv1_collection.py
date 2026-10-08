"""Verify collection/review delivery and unchanged continuation weights."""
import json
import sqlite3
from pathlib import Path
from collections import Counter
import requests
from training_safety import sha256

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'external_datasets/sstv1_incremental_20260914_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    manifest=read(DATA/'manifest.json');queue=read(DATA/'curated_manifest.json');contract=read(DATA/'continuation_contract.json')
    assert len({r['id'] for r in manifest})==len(manifest)
    assert len({r['inat_observation_id'] for r in manifest})==len(manifest)
    for r in manifest:
        assert r['photo_license_code'] in ('cc0','cc-by','cc-by-sa')
        assert r['source_page'] and r['photo_attribution']
        assert sha256(DATA/r['file'])==r['sha256']
        assert not r['training_eligible'] and r['split']=='quarantine'
    for r in queue: assert not r['quality_flags']
    for r in contract['initial_heads']+[contract['backbone']]+contract['protected_artifacts']:
        assert sha256(r['path'])==r['sha256'],r['path']
    endpoint='http://127.0.0.1:8050'
    response=requests.get(endpoint+'/api/herbarium-review',timeout=15);response.raise_for_status()
    listing=response.json();assert len(listing['rows'])==len(queue)
    for url in ('/','/herbarium-review.js',listing['rows'][0]['image_url_local']):
        assert requests.get(endpoint+url,timeout=15).status_code==200
    assert requests.get('http://127.0.0.1:8043/',timeout=15).status_code==200
    framing=read(DATA/'framing_summary.json')
    catalog=read(DATA/'official_catalog.json');present={r['source_species'] for r in manifest}
    missing=[r['scientific_name'] for r in catalog if r['scientific_name'] not in present]
    result={'new_originals':len(manifest),'unique_observations':len(manifest),'clean_pending_review':len(queue),
        'human_review_counts':listing['counts'],'whole_crown_priority':framing['whole_crown_priority'],
        'missing_new_images':missing,'regions':dict(Counter(r['region_query'] for r in manifest)),
        'license_counts':dict(Counter(r['photo_license_code'] for r in manifest)),
        'review_url':endpoint,'current_adapted_heads_verified_unchanged':True,'deployed_files_verified_unchanged':True,
        'training_performed':False,'training_blocker':'New images require quality/species acceptance; unresolved output mappings cannot be silently relabelled.',
        'white_subspecies_resolution':'Recovered via rank-notation-equivalent iNaturalist taxon 537953; two new images included.'}
    (DATA/'VERIFIED_DELIVERY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    header=['# SSTv1 — incremental strengthening / 增量強化','',
        '## Current delivery / 本輪結果','',
        f"- Newly collected originals / 新增原圖：{len(manifest)}（{len(present)} / 27 official entries）。",
        f"- Clean pending review / 初篩待核驗：{len(queue)}；whole-tree/crown priority / 全株樹冠優先：{framing['whole_crown_priority']}。",
        '- Actual initialization: current BioCLIP 2.5 + three adapted MLP heads, not an untrained classifier.',
        '- Not trained yet. Existing image, point and Demo weights verified unchanged. No accuracy gain claimed.',
        '- 待核驗圖片不進入訓練；核驗後需先建立新的實體權重備份，再繼續微調現有 MLP，並保留舊類別回放。',
        '',f"[Review / 核驗]({endpoint}) · A 保留，D/Delete 剔除，Z 撤銷。",'',
        '## Unfilled entries / 本輪未補到新圖','',*[f'- {s}' for s in missing],
        '', 'Sources with no qualifying new licensed research-grade observations were not filled with guessed taxa. This first batch is not exhaustive web coverage.',
        '', '## Taxonomy and provenance','',
        '- Ficus benjamina remains its own source taxon; Demo display renaming does not make it Ficus microcarpa.',
        '- White-paperbark subspecies successfully resolved to Melaleuca cajuputi cumingiana (iNaturalist 537953), two new photos.',
        '- Existing mixed Araucaria and Ficus outputs need explicit compatible mapping; Michelia chapensis, Tabebuia rosea and Yulania x soulangeana are not existing exact output labels.',
        '- Research-grade iNaturalist IDs are source evidence, not independent expert certification. Framing triage is not species validation.',
        '- One photo per observation; old observation IDs/photo IDs, exact hashes and perceptual matches checked. Multi-view identity cannot be guaranteed by observation IDs alone.',
        '- Licenses restricted to CC0, CC BY, CC BY-SA. Original bytes and attribution retained. Nothing published.',
        '',f'[Official target catalog]({catalog[0]["official_page"]})','',
        'See continuation_contract.json for pinned checkpoint hashes; manifest.json for original provenance; framing_triage.json for ranking suggestions; VERIFIED_DELIVERY.json for checks.']
    (DATA/'SSTv1_STATUS.md').write_text('\n'.join(header)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()

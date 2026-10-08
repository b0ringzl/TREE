"""Validate and write a new gallery manifest; old strict selection stays intact."""
import hashlib
import json
import shutil
from collections import Counter
from PIL import Image
from .supplement_detail_references import DATA, OUT
from .collect_references import write_json, LICENSES
from .manual_references import labels

def main():
    previous=json.loads((DATA/'accepted_manifest.json').read_text(encoding='utf-8'))
    pool=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    decisions=json.loads((OUT/'visual_review.json').read_text(encoding='utf-8'))
    extra_path=OUT/'additional/manifest.json'
    if extra_path.exists():
        for r in json.loads(extra_path.read_text(encoding='utf-8')):
            source=OUT/'additional'/r['file'];dest=DATA/r['file']
            if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==r['sha256']
            else:shutil.copy2(source,dest)
            pool.append(r)
    existing={r['id'] for r in previous}
    added=[{**r,'visual_review':'accepted_for_reference','framing':'whole_tree_or_flower_leaf_detail',
        'review_note':decisions[r['id']]['note']} for r in pool if r['id'] not in existing and decisions.get(r['id'],{}).get('status')=='accepted']
    accepted=previous+added
    seen=set()
    for r in accepted:
        path=(DATA/r['file']).resolve()
        assert path.is_relative_to((DATA/'images').resolve()) and r['license'] in LICENSES
        assert r['attribution'] and r['author'] and not r['training_eligible']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        assert r['sha256'] not in seen;seen.add(r['sha256'])
        with Image.open(path) as im:im.verify()
    counts=Counter(r['species'] for r in accepted)
    assert all(c<=2 for c in counts.values())
    report={'target':296,'images':len(accepted),'covered_classes':len(counts),'complete_classes':sum(c==2 for c in counts.values()),
        'missing':{s:counts[s] for s in labels() if counts[s]<2},'preserved_whole_crown_images':len(previous),'added':len(added),
        'policy':'Prefer whole-tree/crown; allow clear flower/leaf details for gaps. No seed-only or trunk-only additions.',
        'manual_selections_preserved':True,'training_eligible':False}
    write_json(OUT/'accepted_manifest.json',accepted);write_json(OUT/'pool_manifest.json',pool);write_json(OUT/'coverage.json',report)
    (OUT/'REPORT.md').write_text('# Mixed reference gallery / 參考圖補充\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\nOriginal strict manifest and all user manual selections are unchanged. New assistant framing decisions are separate from historical decisions and human training reviews. Photo source, attribution, license and original SHA-256 retained. No model training or weights changed.\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()

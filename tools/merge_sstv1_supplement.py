"""Merge a new collection into the review queue without altering source records."""
import argparse
import json
import shutil
from pathlib import Path
from training_safety import sha256

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main(data, extra):
    snapshot=data/'before_white_supplement'
    snapshot.mkdir(exist_ok=False)
    for name in ('manifest.json','curated_manifest.json','totals.json','collection_summary.json','REPORT.md'):
        shutil.copy2(data/name,snapshot/name)
    rows=read(data/'manifest.json'); queue=read(data/'curated_manifest.json'); seen={r['sha256'] for r in rows if r.get('sha256')}
    added=[]
    for row in read(extra/'manifest.json'):
        if row.get('status')!='downloaded' or row['sha256'] in seen: continue
        src=extra/row['file']; dst=data/row['file']
        if dst.exists(): raise FileExistsError(dst)
        if sha256(src)!=row['sha256']: raise ValueError('Supplement source changed')
        shutil.copy2(src,dst)
        if sha256(dst)!=row['sha256']: raise ValueError('Supplement copy failed verification')
        row={**row,'supplement_source_manifest':str((extra/'manifest.json').resolve())}
        rows.append(row);added.append(row);seen.add(row['sha256'])
        if not row.get('quality_flags'): queue.append(row)
    save(data/'manifest.json',rows);save(data/'curated_manifest.json',queue)
    total=read(data/'totals.json')
    total.update(downloaded=sum(r.get('status')=='downloaded' for r in rows),clean_pending_review=len(queue),
        entries_with_images=len({r['source_species'] for r in rows if r.get('status')=='downloaded'}),
        supplement_added=len(added),supplement_path=str(extra.resolve()))
    save(data/'totals.json',total)
    summary=read(data/'collection_summary.json');summary['supplement']=read(extra/'collection_summary.json');save(data/'collection_summary.json',summary)
    with (data/'REPORT.md').open('a',encoding='utf-8') as f:
        f.write(f"\n## White-paperbark subspecies supplement\n\nResolved iNaturalist rank notation: Melaleuca cajuputi cumingiana, taxon 537953. This is NOT an automatic parent-species substitution.\n\nAdded {len(added)} original photos; combined total {total['downloaded']}; combined clean pending review {len(queue)}. Original collection report/manifest preserved in before_white_supplement.\n")
    print(json.dumps(total),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--extra',type=Path,required=True)
    a=p.parse_args();main(a.data,a.extra)

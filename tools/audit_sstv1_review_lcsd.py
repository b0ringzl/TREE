"""Read-only decision audit and LCSD factual link inventory. No image download or training."""
import html
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests
from training_safety import sha256

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'external_datasets/sstv1_incremental_20260914_v1'
LIST='https://www.lcsd.gov.hk/tc/green/education/greeningknowledge/plantphoto/tree_listing.html?type=passage'
NOTICE='https://www.lcsd.gov.hk/tc/notice.html'

class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.active=None;self.entries=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a':
            href=a.get('href','')
            if 'tree_passage/' in href:self.active={'url':urljoin(LIST,href),'text':[]}
        elif tag=='img' and self.active is not None:
            self.active['text'].append(a.get('alt',''))
    def handle_data(self,data):
        if self.active is not None:self.active['text'].append(data)
    def handle_endtag(self,tag):
        if tag=='a' and self.active is not None:
            self.active['text']=' '.join(' '.join(self.active['text']).split())
            self.entries.append(self.active);self.active=None

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    out=ROOT/'outputs'/('sstv1_review_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    queue=read(DATA/'curated_manifest.json');labels=read(ROOT/'outputs/distillation/20260908_0045/classes.json')['labels']
    with sqlite3.connect((DATA/'human_review.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;db.execute('BEGIN')
        decisions={r['id']:dict(r) for r in db.execute('SELECT * FROM decisions')}
        events=[dict(r) for r in db.execute('SELECT * FROM events ORDER BY seq')]
    save(out/'human_review_snapshot.json',{'decisions':decisions,'events':events})
    oldrows=[]
    for phase in ('train','planning','selection','test','field','case'):
        oldrows+=read(ROOT/f'outputs/distillation/20260908_0045/{phase}_rows.json')
    oldhash={r.get('source_sha256') for r in oldrows}
    oldobs={r.get('group_id','').split('observation:')[-1] for r in oldrows if 'observation:' in r.get('group_id','')}
    counts=Counter();accepted=[];errors=[]
    for r in queue:
        d=decisions.get(r['id'],{'status':'pending'})
        counts[d['status']]+=1
        path=DATA/r['file']
        if not path.is_file() or sha256(path)!=r['sha256']:errors.append({'id':r['id'],'error':'source_file_missing_or_changed'})
        if d.get('sha256') and d['sha256']!=r['sha256']:errors.append({'id':r['id'],'error':'decision_hash_mismatch'})
        if d['status']!='accepted':continue
        if r['sha256'] in oldhash or str(r['inat_observation_id']) in oldobs:
            errors.append({'id':r['id'],'error':'existing_dataset_identity_collision'})
        taxon=r['source_species']
        accepted.append({**r,'human_decision':d,'human_acceptance_scope':'image usability; source botanical label retained',
            'exact_existing_target':labels.index(taxon) if taxon in labels else None,
            'mapping_status':'exact_existing_label' if taxon in labels else 'explicit_mapping_required',
            'training_eligible':False,'remaining_gates':['freeze_training_manifest_and_split','verify_run_specific_weight_backup']+
                ([] if taxon in labels else ['resolve_label_mapping'])})
    save(out/'accepted_review_manifest.json',accepted)
    accepted_counts=Counter(r['source_species'] for r in accepted)
    gates=Counter(r['source_species'] for r in accepted if r['exact_existing_target'] is None)
    summary={'review_counts':dict(counts),'queue_total':len(queue),'accepted_source_taxa':len(accepted_counts),
        'accepted_by_taxon':dict(accepted_counts),'exact_existing_label_images':sum(r['exact_existing_target'] is not None for r in accepted),
        'mapping_gate_images':dict(gates),'validation_errors':errors,'last_review_event':events[-1] if events else None,
        'human_decisions_modified':False,'original_images_deleted':False,'training_performed':False,'deployed_models_changed':False}
    save(out/'review_audit.json',summary)
    if errors:raise ValueError('Review audit failed; see '+str(out))
    response=requests.get(LIST,timeout=30,headers={'User-Agent':'SpaceScanTrees/1.0 (single-page botanical link inventory)'})
    response.raise_for_status();parser=Links();parser.feed(response.content.decode('utf-8'))
    links={}
    for entry in parser.entries:
        if entry['url'] not in links or len(entry['text'])>len(links[entry['url']]['text']):links[entry['url']]=entry
    inventory=[]
    for entry in links.values():
        text=html.unescape(entry['text']);match=re.search(r'\b[A-Z][a-z]+\s+[a-z×][A-Za-z .×‘’\-]*',text)
        name=match.group().strip() if match else None
        inventory.append({'source_page':entry['url'],'source_text':text,'scientific_name_from_link':name,
            'image_downloaded':False,'training_eligible':False,'rights_status':'prior_written_authorization_required',
            'rights_notice':NOTICE})
    save(out/'lcsd_species_links.json',inventory)
    save(out/'lcsd_rights_gate.json',{'source_listing':LIST,'notice':NOTICE,
        'notice_checked_on':datetime.now().isoformat(),
        'finding':'LCSD copyright notice expressly requires prior written authorization for copying, adaptation, distribution, publication or making available. No training permission is established by public accessibility or by user image-quality approval.',
        'action':'Record factual species/page links only. Do not bulk download or include LCSD photos in training until scope of written authorization is verified.',
        'image_downloads':0,'training_performed':False,'authorization_request_sent':False})
    lines=['# SSTv1 驗收核對與康文署來源登記','',
        f"已核對 {len(queue)} 張：保留 {counts['accepted']}、剔除 {counts['rejected']}、待驗收 {counts['pending']}。",
        f"保留影像涵蓋 {len(accepted_counts)} 個來源樹種；已核對圖片 SHA-256、驗收記錄雜湊及舊資料觀察 ID／雜湊，異常 {len(errors)}。",
        '','未更改任何人工決定，未刪除原圖，未訓練或切換 Demo。保留表示圖片可用，不等於獨立植物學鑑定。','',
        f"{summary['exact_existing_label_images']} 張可對應現有精確輸出標籤；其餘 {sum(gates.values())} 張需明確標籤對應，不能改成相近樹種。",
        '', '| 需處理的來源標籤 | 保留數 |','|---|---:|',
        *[f'| {s} | {n} |' for s,n in gates.items()], '',
        'Ficus benjamina 不能因 Demo 顯示改名而變成 Ficus microcarpa；Tabebuia rosea 不能改標成 Tabebuia impetiginosa。南洋杉舊混合類別、白千層種／亞種對應也須在訓練清單中明確記錄。','',
        '## 康文署補充來源','',f'已登記 {len(inventory)} 個樹種詳情頁連結；未下載照片。',
        f'[樹種目錄]({LIST}) · [版權公告]({NOTICE})','',
        '康文署公告明確要求相關複製、改編等使用事先取得書面授權。目前沒有此項授權證據，因此康文署照片仍在權限核實階段，不會混入 SSTv1 訓練。',
        '此限制不影響既有 CC0／CC BY／CC BY-SA 來源且已验收影像的後續準備。',
        '', '## 保留資料分佈','', '| 來源樹種 | 保留數 |','|---|---:|',*[f'| {s} | {n} |' for s,n in accepted_counts.items()]]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),**summary,'lcsd_links':len(inventory)},ensure_ascii=True),flush=True)

if __name__=='__main__':main()

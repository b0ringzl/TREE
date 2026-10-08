"""Low-rate public-catalog inventory; isolated from training and production assets.

Uses only endpoints and image URLs exposed by the site's plantSearch.js.
Never guesses unwatermarked/original image paths. No authentication or bypass.
"""
import argparse,hashlib,html,io,json,re,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urljoin,urlparse
from urllib.robotparser import RobotFileParser
import requests
import numpy as np
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE='https://www.herbarium.gov.hk'
ASSETS='/plantdb/src/UHDMS_prd/'
NOTICE=BASE+'/en/important-notices/index.html'
UA='TreeAtlasResearchInventory/0.1 (low-rate public botanical catalog audit)'
PRIORITY=[('Ficus microcarpa','榕樹'),('Ficus benjamina','榕樹'),('Livistona chinensis','蒲葵'),('Wodyetia bifurcata','狐尾椰子'),('Delonix regia','鳳凰木'),('Aleurites moluccana','石栗'),('Leucaena leucocephala','銀合歡'),('Albizia lebbeck','大葉合歡'),('Araucaria columnaris','南洋杉'),('Araucaria heterophylla','南洋杉'),('Bauhinia x blakeana','易混淆種'),('Bauhinia variegata','易混淆種')]
# Explicit existing project mapping (tools/tree_species_label_merge_map.csv),
# not a claim that the source uses the parent name or all related taxa are equal.
PROJECT_ALIASES={'macaranga tanarius var. tomentosa':'macaranga tanarius'}

def write_json(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def norm(name):return re.sub(r'\s+',' ',html.unescape(name).replace('×','x')).strip().lower()

class Client:
    def __init__(self,out,interval=1.5):
        self.out=out;self.interval=max(1.5,interval);self.last=0;self.s=requests.Session();self.s.headers['User-Agent']=UA
        self.cache=out/'cache';self.cache.mkdir(parents=True,exist_ok=True);self.robot=None
    def get(self,url,limit=25_000_000):
        url=urljoin(BASE,url);u=urlparse(url)
        if u.scheme!='https' or u.hostname!='www.herbarium.gov.hk':raise ValueError('Unexpected external URL')
        if self.robot and not self.robot.can_fetch(UA,url):raise RuntimeError('robots.txt disallows '+url)
        key=hashlib.sha256(url.encode()).hexdigest();datafile=self.cache/(key+'.bin');metafile=self.cache/(key+'.json')
        if datafile.exists() and metafile.exists():return datafile.read_bytes(),json.loads(metafile.read_text(encoding='utf-8'))
        time.sleep(max(0,self.interval-(time.monotonic()-self.last)))
        self.last=time.monotonic()
        with self.s.get(url,timeout=(10,40),allow_redirects=False,stream=True) as r:
            if r.status_code in (401,403,429):raise RuntimeError('Access restricted/rate limited; stopped: '+str(r.status_code))
            r.raise_for_status()
            if 300<=r.status_code<400:raise RuntimeError('Redirect requires review: '+url)
            chunks=[];size=0
            for chunk in r.iter_content(65536):
                size+=len(chunk)
                if size>limit:raise RuntimeError('Response too large: '+url)
                chunks.append(chunk)
            raw=b''.join(chunks);meta={'url':url,'status':r.status_code,'content_type':r.headers.get('Content-Type'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_at':datetime.now(timezone.utc).isoformat()}
        datafile.write_bytes(raw);write_json(metafile,meta)
        return raw,meta
    def json(self,url):return json.loads(self.get(url)[0].decode('utf-8-sig'))
    def policy(self):
        raw,meta=self.get('/robots.txt');s=raw.decode('utf-8-sig',errors='replace')
        valid=bool(re.search(r'^\s*User-agent\s*:',s,re.M|re.I)) and '<html' not in s.lower()
        if valid:self.robot=RobotFileParser();self.robot.parse(s.splitlines())
        notice,notice_meta=self.get(NOTICE)
        result={'robots_response':meta,'robots_parseable':valid,'robots_interpretation':'parsed' if valid else 'HTML redirect body; no usable directives, not explicit crawling permission','copyright_url':NOTICE,'copyright_response':notice_meta,'usage':'Non-commercial reproduction with attribution stated; commercial reuse requires prior written authorization. ML training not explicitly addressed.','training_authorization':'not established by this inventory','image_policy':'Only published watermarked URLs; preserve originals and attribution; no public GitHub image upload.'}
        write_json(self.out/'access_and_rights.json',result)
        return result

def quality(raw):
    with Image.open(io.BytesIO(raw)) as original:
        original.load();im=ImageOps.exif_transpose(original).convert('RGB')
    gray=np.asarray(im.convert('L').resize((320,240)),dtype=np.float32)
    score=float(np.var(gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]-4*gray[1:-1,1:-1]))
    flags=[]
    if min(im.size)<224:flags.append('small_resolution')
    if gray.mean()<40:flags.append('dark')
    if gray.mean()>225:flags.append('bright')
    if score<18:flags.append('low_texture_or_blur')
    return {'width':im.width,'height':im.height,'brightness':round(float(gray.mean()),2),'laplacian_variance_320x240':round(score,2),'heuristic_flags':flags,'visual_review':'pending','organ':'unreviewed','whole_tree_coverage':'unreviewed'},im

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=ROOT/'external_datasets/herbarium_inventory_20260907');a.add_argument('--samples-per-priority-species',type=int,default=2);args=a.parse_args()
    if not 0<=args.samples_per_priority_species<=2:raise ValueError('Inventory permits at most two sample photos per priority species')
    out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'samples').mkdir(exist_ok=True)
    c=Client(out);policy=c.policy()
    # Preserve the publicly linked script for endpoint provenance, not execution.
    c.get('/filemanager/template/common/js/plantSearch.js')
    targets={norm(n):{'query':n,'group':g,'priority':True} for n,g in PRIORITY}
    catalog=json.loads((ROOT/'tree_species_demo/catalog.json').read_text(encoding='utf-8'))
    for row in catalog['hongkong']:
        n=row['scientific_name']
        if '(' in n:continue # Explicitly handled as separate Araucaria taxa above.
        targets.setdefault(norm(n),{'query':n,'group':row['chinese_name'],'priority':False})
    entries=[]
    for letter in sorted({v['query'][0].lower() for v in targets.values()}):
        d=c.json(ASSETS+'js/species/species_'+letter+'.json')
        entries.extend(x for group in d['records'].values() for x in group)
    inventory=[];photos=[];sample_records=[];thumbs=[]
    for key,target in targets.items():
        matches=[x for x in entries if norm(x['scientific_name'])==key]
        exact_count=len(matches)
        if not matches:matches=[x for x in entries if PROJECT_ALIASES.get(norm(x['scientific_name']))==key]
        row={**target,'exact_match_count':exact_count,'status':'matched' if exact_count else 'project_alias_match' if matches else 'no_exact_catalog_match','species_records':[],'photo_count':0,'specimen_photo_count':0}
        for match in matches:
            ident=match['species_id'];detail=c.json('/plantdb/GetSpecies.php?id='+ident)
            page=BASE+'/tc/hk-plant-database/plant-detail/index.html?pType=species&oID='+ident
            photo_list=detail.get('photo',[])
            row['species_records'].append({'id':ident,'scientific_name':detail.get('scientific_name_without_authority'),'chinese_names':detail.get('chinese_name',[]),'common_names':detail.get('common_name',[]),'page':page})
            row['photo_count']+=len(photo_list);row['specimen_photo_count']+=sum(len(s.get('photo',[])) for s in detail.get('specimen',[]))
            for idx,p in enumerate(photo_list):
                url=urljoin(BASE,p['filepath']);entry={'query':target['query'],'group':target['group'],'source_species':detail.get('scientific_name_without_authority'),'source_species_id':ident,'source_names':detail.get('chinese_name',[]),'media_id':p.get('photo_multimedia_id'),'image_url':url,'source_page':page,'asset_type':'plant_photo','copyright_holder':'Hong Kong Herbarium, AFCD','rights_url':NOTICE,'review_status':'not_accepted_for_training','source_group':ident,'published_watermark_path':'_WM.' in url.upper()}
                photos.append(entry)
                if target['priority'] and idx<args.samples_per_priority_species:
                    raw,meta=c.get(url);q,im=quality(raw);name=f"{ident}_{p['photo_multimedia_id']}.jpg";path=out/'samples'/name
                    path.write_bytes(raw) # Preserve downloaded bytes, including watermark.
                    sr={**entry,'file':'samples/'+name,'sha256':meta['sha256'],'bytes':len(raw),'quality':q};sample_records.append(sr)
                    tile=Image.new('RGB',(360,300),'white');preview=im.copy();preview.thumbnail((350,250));tile.paste(preview,((360-preview.width)//2,0));draw=ImageDraw.Draw(tile);draw.text((8,254),target['query'],fill='black');draw.text((8,276),f"{ident} / {p['photo_multimedia_id']} / {im.width}x{im.height}",fill='black');thumbs.append(tile)
        inventory.append(row)
        write_json(out/'inventory.json',inventory);write_json(out/'photo_manifest.json',photos);write_json(out/'sample_quality.json',sample_records)
        print(target['query']+': '+str(row['photo_count'])+' photos; '+row['status'],flush=True)
    if thumbs:
        sheet=Image.new('RGB',(360*4,300*((len(thumbs)+3)//4)),(230,235,230))
        for i,tile in enumerate(thumbs):sheet.paste(tile,((i%4)*360,(i//4)*300))
        sheet.save(out/'contact_sheet.jpg',quality=90)
    hashes=Counter(x['sha256'] for x in sample_records)
    summary={'target_taxa':len(inventory),'matched_taxa':sum(bool(x['species_records']) for x in inventory),'exact_matched_taxa':sum(x['status']=='matched' for x in inventory),'alias_matched_taxa':sum(x['status']=='project_alias_match' for x in inventory),'catalog_photo_entries':len(photos),'unique_photo_urls':len({x['image_url'] for x in photos}),'sample_downloads':len(sample_records),'unique_sample_hashes':len(hashes),'sample_duplicate_groups':sum(v>1 for v in hashes.values()),'training_images_accepted':0,'models_modified':False,'cross_existing_dataset_dedup':'not yet performed','quality_scope':'Only first two published photos per priority taxon, not representative accuracy or whole-tree coverage assessment.'}
    write_json(out/'summary.json',summary)
    lines=['# 香港植物標本室圖像盤點','',f"查詢 {summary['target_taxa']} 個分類單元；精確匹配 {summary['exact_matched_taxa']} 個，既有專案別名規則匹配 {summary['alias_matched_taxa']} 個；發現 {len(photos)} 個照片條目、{summary['unique_photo_urls']} 個不同公開圖片網址。",'',f"下載質檢樣圖 {len(sample_records)} 張；尚未批准任何圖片進入訓練，未修改模型。",'','| 查詢學名 | 分組 | 公開照片條目 | 狀態 |','|---|---|---:|---|']
    for r in inventory:lines.append(f"| {r['query']} | {r['group']} | {r['photo_count']} | {r['status']} |")
    lines+=['','## 使用邊界','','只使用頁面公開的帶水印圖片網址，不猜測無水印版本。robots.txt 返回 HTML 跳轉內容，並非明確抓取許可；採單連線低頻訪問，遇到 401/403/429 停止。','',f'[官方使用條款]({NOTICE})：非商業複製／傳播須註明來源；商業使用要求書面授權。條款未明確涵蓋機器學習訓練，本盤點不構成訓練或商業授權。','','南洋杉兩個原始物種分別保留；榕樹保留原始種級學名和使用者分組，不改寫來源學名；銀合歡和大葉合歡不合併。','','樣圖按官方排序取前兩張，未按模型預測選圖。照片條目數不等於獨立植株數；尚未與既有訓練／驗證資料交叉去重。','','新增來源只能補充影像分支，不能生成真实配對點雲。下一階段需人工抽查整樹／器官類型、去重、分組劃分及授權確認，然後另訓候選版本，以固定街景測試集比較分類性能。']
    (out/'inventory_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

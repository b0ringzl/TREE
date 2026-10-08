"""Licensed source-labelled conservation expansion; no fitting or acceptance.

New original downloads stay pending_agent_review. Existing images, human
decisions, weights and deployment are never edited by this tool.
"""
import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw, ImageOps
from collect_sstv1_incremental import ROOT, LICENSES, UA, download, save
from collect_conservation_sources_20260917 import augmented_index, identifiers, query

SPECIES=[('Artocarpus hypargyreus','白桂木'),('Rhodoleia championii','紅花荷'),('Keteleeria fortunei','油杉')]
STRONG='multiple_independent_exact_ids'


def balance(candidates):
    """Strong source evidence first, then round-robin observers within each tier."""
    ordered=[]
    for tier in (STRONG,'source_label_only_requires_human_confirmation'):
        bins={}
        for row in sorted(candidates,key=lambda r:(r['observation_id'],r['photo_id'])):
            if row['source_evidence']['tier']==tier:
                bins.setdefault(row['observer_id'],[]).append(row)
        while any(bins.values()):
            for values in bins.values():
                if values:ordered.append(values.pop(0))
    return ordered


def make_sheets(rows,out,species):
    clean=[r for r in rows if r['status']=='downloaded' and not r['quality_flags']]
    slug=species.replace(' ','_')
    for start in range(0,len(clean),12):
        sheet=Image.new('RGB',(1440,1140),'white'); draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(clean[start:start+12]):
            with Image.open(row['source_image']) as opened:
                im=ImageOps.exif_transpose(opened).convert('RGB');im.thumbnail((350,290))
            x=j%4*360;y=j//4*380
            sheet.paste(im,(x+(350-im.width)//2,y))
            draw.text((x+4,y+297),row['id'],fill='black')
            draw.text((x+4,y+318),row['source_species'],fill='black')
            draw.text((x+4,y+339),('obs '+str(row['observation_id'])) if row.get('observation_id') else 'commons page '+str(row['commons_page_id']),fill='black')
            draw.text((x+4,y+359),'PENDING '+('multi-ID' if row['source_evidence']['tier']==STRONG else 'weak source label'),fill='black')
        sheet.save(out/'contact_sheets'/f'{slug}_{start//12:03d}.jpg',quality=93)


def main(out,cap):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    for directory in ('images','source_responses','contact_sheets'):(out/directory).mkdir()
    old_obs,old_hashes,old_photos,old_visual=augmented_index(out)
    save(out/'PLAN.json',{'created_at':datetime.now(timezone.utc).isoformat(),'species':SPECIES,'max_download_attempts_per_species':cap,
        'max_photos_per_observation':2,'licenses':sorted(LICENSES),'source':'iNaturalist public API',
        'priority':'Multiple current exact species identifiers including a non-observer, no species conflict; otherwise separate weak source-label channel',
        'state':'All originals pending_agent_review and training_eligible=false; source labels are not human acceptance',
        'partition':'quarantine only; observations/observers/location grids are grouping proxies, not verified separate trees',
        'negative_label_policy':'No model-generated labels; no fitting/deployment/old human DB modifications',
        'rate_limit':'Serial >=1.1s delay; stop this source on 401/403/429, never bypass'})
    all_rows=[];audits=[];summaries=[];errors=[];seen_photos=set();seen_hashes=set();seen_visual=[];access_stop=False
    for number,(species,zh) in enumerate(SPECIES):
        print('SOURCE',species,flush=True)
        rows=[];examined=set();candidate_rows=[];total=None;page_count=0
        try:
            taxa=query('taxa/autocomplete',{'q':species,'per_page':10})
            save(out/'source_responses'/f'{number:02d}_taxon.json',taxa)
            exact=[r for r in taxa.get('results',[]) if r.get('is_active') and r.get('name')==species]
            if len(exact)!=1:raise ValueError('Exact active taxon is not uniquely resolved; no silent synonym mapping')
            taxon=exact[0]['id']
            for page in range(1,6):
                response=query('observations',{'taxon_id':taxon,'photos':'true','photo_license':','.join(sorted(LICENSES)),
                    'per_page':200,'page':page,'order_by':'id','order':'desc'})
                save(out/'source_responses'/f'{number:02d}_observations_page{page}.json',response)
                page_count=page;total=response.get('total_results',0)
                for o in response.get('results',[]):
                    if o['id'] in examined:continue
                    examined.add(o['id']);ev=identifiers(o,taxon);reasons=[]
                    if (o.get('taxon') or {}).get('id')!=taxon:reasons.append('not_exact_taxon')
                    if str(o['id']) in old_obs:reasons.append('old_observation_including_pending_rejected')
                    if o.get('flags') or ev['conflicting_current_identifications']:reasons.append('flags_or_current_species_conflict')
                    audits.append({'species':species,'observation_id':o['id'],'source_evidence':ev,'exclusions':reasons})
                    if reasons:continue
                    choices=[p for p in o.get('photos',[]) if p.get('license_code') in LICENSES and p['id'] not in old_photos
                             and p['id'] not in seen_photos and not p.get('flags') and not p.get('hidden')][:2]
                    for p in choices:
                        seen_photos.add(p['id']);loc=o.get('location');grid=None
                        if loc:
                            lat,lng=map(float,loc.split(','));grid=f'{math.floor(lat*100)}:{math.floor(lng*100)}'
                        candidate_rows.append({'id':'cons_expand_inat_'+str(p['id']),'source':'iNaturalist',
                            'source_species':species,'source_names':[zh],'source_url':o.get('uri'),'source_page':o.get('uri'),
                            'source_evidence':ev,'source_evidence_lane':'strong_source_pending' if ev['tier']==STRONG else 'weak_source_pending',
                            'source_response_file':f'source_responses/{number:02d}_observations_page{page}.json',
                            'inat_taxon_id':taxon,'inat_taxon_name':species,'inat_quality_grade':o.get('quality_grade'),
                            'observation_id':o['id'],'inat_observation_id':o['id'],'observer_id':o['user']['id'],
                            'observed_on':o.get('observed_on'),'location_public':loc,'location_obscured':o.get('obscured'),
                            'place_guess':o.get('place_guess'),'location_grid_0p01deg':grid,
                            'group_id':'inat_observation_'+str(o['id']),'observer_group':'inat_observer_'+str(o['user']['id']),
                            'identity_status':'Same observation does not guarantee same individual; different observations may show same tree',
                            'photo_id':p['id'],'photo_url':p['url'].replace('/square.','/large.'),
                            'license':p['license_code'],'photo_license_code':p['license_code'],
                            'attribution':p.get('attribution'),'photo_attribution':p.get('attribution'),
                            'review_status':'pending_agent_review','agent_review_status':'pending','human_review_status':'pending',
                            'training_eligible':False,'split':'quarantine','review_priority':'normal','review_note':''})
                if sum(r['source_evidence']['tier']==STRONG for r in candidate_rows)>=cap or page*200>=total:break
            save(out/f'{species.replace(" ","_")}_candidate_metadata.json',candidate_rows)
            for row in balance(candidate_rows)[:cap]:
                time.sleep(1.1);row=download((row,out));flags=[]
                if row['status']!='downloaded':
                    flags.append('download_failed')
                    if any(str(code) in row.get('error','') for code in (401,403,429)):
                        access_stop=True
                else:
                    row['source_image']=str((out/row['file']).resolve());h=int(row['phash'],16)
                    if row['sha256'] in old_hashes:flags.append('existing_exact_image')
                    if row['sha256'] in seen_hashes:flags.append('new_exact_duplicate')
                    near=[p for v,p in old_visual if (h^v).bit_count()<=6]
                    if near:flags.append('existing_near_image')
                    if any((h^v).bit_count()<=6 for v in seen_visual):flags.append('new_near_duplicate')
                    if min(row['dimensions'])<224:flags.append('low_resolution')
                    if row['quality']['dark_fraction']>.65:flags.append('mostly_dark')
                    row['near_existing_matches']=near[:20]
                    seen_hashes.add(row['sha256']);seen_visual.append(h)
                row['quality_flags']=flags;row['candidate_lane']='quarantined_quality_or_duplicate' if flags else 'pending_agent_review'
                rows.append(row);all_rows.append(row)
                save(out/'manifest.json',all_rows)
                save(out/'review_manifest.json',[r for r in all_rows if r['status']=='downloaded' and not r['quality_flags']])
                if access_stop:break
            make_sheets(rows,out,species)
        except Exception as exc:
            errors.append({'species':species,'error':str(exc)});print('ERROR',species,str(exc),flush=True)
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (401,403,429):access_stop=True
        clean=[r for r in rows if r['status']=='downloaded' and not r['quality_flags']]
        counts=Counter(r['observation_id'] for r in clean)
        summary={'species':species,'source_total_observations':total,'queried_observations':len(examined),'pages_queried':page_count,
            'downloaded':sum(r['status']=='downloaded' for r in rows),'pending_agent_review':len(clean),
            'strong_source_pending':sum(r['source_evidence']['tier']==STRONG for r in clean),
            'weak_source_pending':sum(r['source_evidence']['tier']!=STRONG for r in clean),
            'observations':len(counts),'observers':len({r['observer_id'] for r in clean}),
            'two_photo_observations':sum(n==2 for n in counts.values()),
            'quality_exclusions':dict(Counter(f for r in rows for f in r['quality_flags']))}
        summaries.append(summary)
        save(out/'source_audit.json',audits)
        save(out/'collection_summary.json',{'by_species':summaries,'errors':errors,'access_stopped':access_stop,
            'training_performed':False,'deployed_files_changed':False,'human_decisions_written':False})
        print('SPECIES_COMPLETE',json.dumps(summary),flush=True)
        if access_stop:break
    clean=[r for r in all_rows if r['status']=='downloaded' and not r['quality_flags']]
    assert all(r['review_status']=='pending_agent_review' and r['training_eligible'] is False for r in all_rows)
    verified=[]
    for row in all_rows:
        if row['status']!='downloaded':continue
        path=Path(row['source_image']);actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==row['sha256']
        verified.append({'id':row['id'],'sha256':actual,'bytes':path.stat().st_size})
    save(out/'VERIFIED.json',{'downloaded_images':verified,'human_accepted':0,'training_eligible':0,'fitting_performed':False,
        'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest() if (out/'manifest.json').exists() else None,
        'review_manifest_sha256':hashlib.sha256((out/'review_manifest.json').read_bytes()).hexdigest() if (out/'review_manifest.json').exists() else None})
    totals={'downloaded':len(verified),'pending_agent_review':len(clean),'human_accepted':0,'training_eligible':0,
        'by_species':summaries,'errors':errors,'access_stopped':access_stop}
    save(out/'totals.json',totals)
    lines=['# 保育树种图像扩展 — 来源候选批次','','所有图片仍为 pending_agent_review；没有人工接受、训练或部署。',
        '', '| 树种 | 下载 | 待视觉核验 | 多独立鉴定 | 弱来源标签 | 观察 | 作者 | 双图观察 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['species']} | {s['downloaded']} | {s['pending_agent_review']} | {s['strong_source_pending']} | {s['weak_source_pending']} | {s['observations']} | {s['observers']} | {s['two_photo_observations']} |")
    lines+=['','## 边界','','- 使用公开 iNaturalist API，许可限定 CC0 / CC BY / CC BY-SA；保留作者、来源观察、原响应及下载字节。',
        '- 排除旧观察、照片编号、SHA与近重复，包括既有 pending/rejected 和用户84图；许可与社区鉴定不等同于人工物种核验。',
        '- 每种最多尝试下载60图，每个观察最多2图，强来源证据和跨作者优先；数量不足不降低资格假装充足。',
        '- 新图全部隔离，不继承旧类标签为人工接受，不将同观察两张照片当两株树；同观察是否同一株仍须核实。',
        '- 未划分 train/validation/test；未改动任何旧清单、人审数据库、部署或权重。',
        '- review_manifest.json 仅去掉机械质量／重复隔离项，不等于视觉审核通过；弱来源在 source_evidence_lane 单独记录。',
        '- contact_sheets 按树种分别展示待核验图。下载的是网站 large 版本，不冒称相机原始文件。',
        '- 本批没有点云，不声称具有新增图像＋点云配对。']
    if errors:lines+=['','## 采集错误','',json.dumps(errors,ensure_ascii=False)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('COMPLETE',json.dumps(totals),flush=True)


def commons_query(params):
    time.sleep(1.1)
    response=requests.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json',**params},headers=UA,timeout=40)
    response.raise_for_status()
    data=response.json()
    if data.get('error'):raise ValueError(str(data['error']))
    return data


def commons_main(out,cap):
    """Independent, licensed fallback while iNaturalist public API is unavailable."""
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    for d in ('images','source_responses','contact_sheets'):(out/d).mkdir()
    old_obs,old_hashes,old_photos,old_visual=augmented_index(out)
    save(out/'PLAN.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source':'Wikimedia Commons public API',
        'fallback_reason':'iNaturalist public API returned HTTP503 downtime for all three taxa and a limited recheck; no access-denial bypass.',
        'species':SPECIES,'max_download_attempts_per_species':cap,'licenses':sorted(LICENSES),
        'state':'All downloads pending_agent_review, training_eligible=false, no human decisions.',
        'identity':'Commons file page is not a biological observation; observation_id remains null. Same author/date/species is a conservative series proxy, max 2 images per such group.',
        'source_evidence':'Taxon category/file metadata only; not independent botanist verification. Author and public dates retained.',
        'rate_limit':'Serial requests with >=1.1 s delay; stop source on 401/403/429.',
        'splits':'quarantine; no train/val/test assignment; no model fitting, deployment or old data edits.'})
    all_rows=[];audit=[];summaries=[];errors=[];seen_hashes=set();seen_visual=[];access_stop=False
    for number,(species,zh) in enumerate(SPECIES):
        print('COMMONS_SOURCE',species,flush=True);rows=[];candidates=[];category_count=0
        try:
            category=commons_query({'list':'categorymembers','cmtitle':'Category:'+species,'cmtype':'file','cmlimit':100})
            save(out/'source_responses'/f'{number:02d}_category.json',category)
            members=category.get('query',{}).get('categorymembers',[]);category_count=len(members)
            if category.get('continue'):
                audit.append({'species':species,'note':'Direct category has more than 100 files; initial batch bounded to first100.'})
            for offset in range(0,len(members),25):
                subset=members[offset:offset+25]
                details=commons_query({'pageids':'|'.join(str(r['pageid']) for r in subset),'prop':'imageinfo',
                    'iiprop':'url|extmetadata|user|timestamp|size|sha1|mime','iiurlwidth':1600})
                save(out/'source_responses'/f'{number:02d}_imageinfo_{offset//25:02d}.json',details)
                for page in details.get('query',{}).get('pages',{}).values():
                    information=page.get('imageinfo') or []
                    if not information:continue
                    info=information[0];ext=info.get('extmetadata') or {}
                    value=lambda key:str((ext.get(key) or {}).get('value',''))
                    plain=lambda text:re.sub(r'<[^>]*>','',text).strip()
                    licence_url=value('LicenseUrl');short=value('LicenseShortName');code=None
                    if re.search(r'creativecommons.org/licenses/by-sa/',licence_url,re.I):code='cc-by-sa'
                    elif re.search(r'creativecommons.org/licenses/by/',licence_url,re.I):code='cc-by'
                    elif re.search(r'creativecommons.org/publicdomain/zero/',licence_url,re.I):code='cc0'
                    if code not in LICENSES:
                        audit.append({'species':species,'page_id':page['pageid'],'title':page['title'],'exclusion':'license_not_explicit_allowlist','license_url':licence_url,'license_short':short});continue
                    if info.get('mime') not in ('image/jpeg','image/png','image/webp') and Path(urlparse(info.get('url','')).path).suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):
                        audit.append({'species':species,'page_id':page['pageid'],'exclusion':'not_supported_raster'});continue
                    artist=plain(value('Artist')) or info.get('user','unknown')
                    date=plain(value('DateTimeOriginal')) or plain(value('DateTime'))
                    # Unknown original dates group together by author/species rather than pretend independence.
                    stable_date=date[:10] if date else 'unknown-original-date'
                    series_text=species+'|'+artist+'|'+stable_date
                    series='commons_author_date_'+hashlib.sha256(series_text.encode()).hexdigest()[:16]
                    candidates.append({'id':'cons_expand_commons_'+str(page['pageid']),'source':'Wikimedia Commons',
                        'source_species':species,'source_names':[zh],'source_species_zh_hant':zh,
                        'source_url':info.get('descriptionurl'),'source_page':info.get('descriptionurl'),
                        'source_evidence':{'tier':'source_label_only_requires_human_confirmation','category':'Category:'+species,
                            'description':plain(value('ImageDescription')),'independent_exact_identifiers':None,
                            'note':'Commons category is a source taxon assertion, not independent botanical confirmation.'},
                        'source_evidence_lane':'weak_source_pending','source_response_file':f'source_responses/{number:02d}_imageinfo_{offset//25:02d}.json',
                        'commons_page_id':page['pageid'],'commons_title':page['title'],'commons_sha1':info.get('sha1'),
                        'observation_id':None,'observer_id':artist,'observer_group':'commons_author_'+hashlib.sha256(artist.encode()).hexdigest()[:16],
                        'observed_on':date or None,'uploaded_at':info.get('timestamp'),'location_public':None,
                        'place_guess':None,'group_id':series,'source_group':series,
                        'identity_status':'Author/original-date/species series proxy, not verified same tree or independent individuals',
                        'photo_url':info.get('thumburl') or info['url'],'original_media_url':info['url'],
                        'license':code,'photo_license_code':code,'license_url':licence_url,'license_short_name':short,
                        'attribution':artist,'photo_attribution':artist,'attribution_html':value('Artist'),'credit':plain(value('Credit')),
                        'review_status':'pending_agent_review','agent_review_status':'pending','human_review_status':'pending',
                        'training_eligible':False,'split':'quarantine','review_priority':'normal','review_note':''})
            save(out/f'{species.replace(" ","_")}_candidate_metadata.json',candidates)
            groups=Counter();bins={}
            for row in candidates:
                if groups[row['group_id']]>=2:continue
                groups[row['group_id']]+=1;bins.setdefault(row['observer_id'],[]).append(row)
            selected=[]
            while any(bins.values()) and len(selected)<cap:
                for values in bins.values():
                    if values and len(selected)<cap:selected.append(values.pop(0))
            for row in selected:
                time.sleep(1.1);row=download((row,out));flags=[]
                if row['status']!='downloaded':
                    flags.append('download_failed')
                    if any(str(code) in row.get('error','') for code in (401,403,429)):access_stop=True
                else:
                    row['source_image']=str((out/row['file']).resolve());h=int(row['phash'],16)
                    if row['sha256'] in old_hashes:flags.append('existing_exact_image')
                    if row['sha256'] in seen_hashes:flags.append('new_exact_duplicate')
                    near=[p for v,p in old_visual if (h^v).bit_count()<=6]
                    if near:flags.append('existing_near_image')
                    if any((h^v).bit_count()<=6 for v in seen_visual):flags.append('new_near_duplicate')
                    if min(row['dimensions'])<224:flags.append('low_resolution')
                    if row['quality']['dark_fraction']>.65:flags.append('mostly_dark')
                    row['near_existing_matches']=near[:20];seen_hashes.add(row['sha256']);seen_visual.append(h)
                row['quality_flags']=flags;row['candidate_lane']='quarantined_quality_or_duplicate' if flags else 'pending_agent_review'
                rows.append(row);all_rows.append(row)
                save(out/'manifest.json',all_rows);save(out/'review_manifest.json',[r for r in all_rows if r['status']=='downloaded' and not r['quality_flags']])
                if access_stop:break
            make_sheets(rows,out,species)
        except Exception as exc:
            errors.append({'species':species,'error':str(exc)});print('ERROR',species,str(exc),flush=True)
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (401,403,429):access_stop=True
        clean=[r for r in rows if r['status']=='downloaded' and not r['quality_flags']]
        counts=Counter(r['group_id'] for r in clean)
        summary={'species':species,'direct_category_files':category_count,'explicit_licensed_candidates':len(candidates),
            'downloaded':sum(r['status']=='downloaded' for r in rows),'pending_agent_review':len(clean),'strong_source_pending':0,
            'weak_source_pending':len(clean),'authors':len({r['observer_id'] for r in clean}),'author_date_series':len(counts),
            'two_photo_series':sum(n==2 for n in counts.values()),'quality_exclusions':dict(Counter(f for r in rows for f in r['quality_flags']))}
        summaries.append(summary);save(out/'source_audit.json',audit)
        save(out/'collection_summary.json',{'by_species':summaries,'errors':errors,'access_stopped':access_stop,'training_performed':False})
        print('SPECIES_COMPLETE',json.dumps(summary),flush=True)
        if access_stop:break
    save(out/'manifest.json',all_rows);clean=[r for r in all_rows if r['status']=='downloaded' and not r['quality_flags']]
    save(out/'review_manifest.json',clean)
    verified=[]
    for row in all_rows:
        if row['status']!='downloaded':continue
        path=Path(row['source_image']);actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==row['sha256']
        verified.append({'id':row['id'],'sha256':actual,'bytes':path.stat().st_size})
    save(out/'VERIFIED.json',{'downloaded_images':verified,'human_accepted':0,'training_eligible':0,'fitting_performed':False,
        'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
        'review_manifest_sha256':hashlib.sha256((out/'review_manifest.json').read_bytes()).hexdigest()})
    totals={'downloaded':len(verified),'pending_agent_review':len(clean),'human_accepted':0,'training_eligible':0,
        'by_species':summaries,'errors':errors,'access_stopped':access_stop}
    save(out/'totals.json',totals)
    lines=['# 三树种扩展图像 — Commons 独立来源补充','','iNaturalist API 503 暂不可用，本批来自独立的 Wikimedia Commons 公开 API。所有图仍为 pending_agent_review。',
        '', '| 树种 | 下载 | 待核验 | 作者 | 作者/日期系列 |','|---|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['species']} | {s['downloaded']} | {s['pending_agent_review']} | {s['authors']} | {s['author_date_series']} |")
    lines+=['','- 仅接受逐文件 LicenseUrl 明确为 CC0 / CC BY / CC BY-SA；作者、来源页面、原API响应、许可URL与下载SHA均保留。',
        '- Commons 分类／描述不是专家或多人鉴定，所以本批全在 weak_source_pending，视觉判断和人工核验尚未进行。',
        '- Commons page ID 不是生物观察编号；observation_id=null。不把照片数或作者/日期系列数说成独立树数量。',
        '- 同作者、日期、物种最多取2图，尽量跨作者；同日期系列也不保证是同一株。',
        '- SHA与感知近重复排除旧全库／用户图／待审或拒绝图；尺寸过小或过暗隔离。review_manifest只过机械门，不等于训练批准。',
        '- 图像为网站原文件或其1600像素版本，下载字节保留；不移水印，不上传公开Demo。',
        '- 没有训练、没有部署、没有人审数据库修改、没有点云。原v1的503审计保留未覆盖。']
    if errors:lines+=['','## 采集错误','',json.dumps(errors,ensure_ascii=False)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('COMPLETE',json.dumps(totals),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cap',type=int,default=60)
    parser.add_argument('--source',choices=['inat','commons'],default='inat')
    args=parser.parse_args();assert 1<=args.cap<=60
    (commons_main if args.source=='commons' else main)(args.out,args.cap)

"""Read-only local results viewer and provenance ledger for the 20 m survey."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from distillation_core import read_json
from focus8_data import ROOT,LABELS,TAXA
from run_vmms20m_teacher import write
from training_safety import sha256


def apply_gate(out,frames,gate):
    """Derived presentation only: immutable anchor records remain untouched."""
    import numpy as np
    from PIL import Image,ImageDraw
    decisions={r['id']:r for r in read_json(gate/'decisions.json') if r['group']=='survey'}
    if any(r['decision']=='suppress_non_tree' and 'assistant_visual_review' not in r for r in decisions.values()):
        raise ValueError('Automatic background vetoes require recorded visual review before publishing')
    labels=read_json(out/'class_names.json');source_frames=read_json(out/'frames.json')
    removed=[];recomputed=[];cache={}
    def probabilities(e):
        key=e['frame_key']
        if key not in cache:
            original=read_json(out/'frames'/key/'result.json')
            with np.load(out/'frames'/key/'features.npz') as arrays:
                cache[key]={r['id']:arrays['probabilities'][r['probability_index']].copy() for r in original['instances']}
        return cache[key][e['instance_id']]
    for frame in frames:
        kept=[];rejected=[]
        for row in frame['instances']:
            if row['id'] not in decisions:raise ValueError('Missing gate decision: '+row['id'])
            if decisions[row['id']]['decision']=='suppress_non_tree':
                removed.append(row['id']);rejected.append(row['id']);continue
            evidence=[e for e in row['evidence'] if decisions[e['instance_id']]['decision']!='suppress_non_tree']
            if len(evidence)!=len(row['evidence']):
                ps=[probabilities(e) for e in evidence];weights=[e['quality_score'] for e in evidence]
                p=np.average(ps,axis=0,weights=weights);order=np.argsort(-p)
                agree=len({int(v.argmax()) for v in ps})==1
                strong=bool(p[order[0]]>=.65 and p[order[0]]-p[order[1]]>=.20 and agree and max(weights)>=.55)
                row.update(evidence=evidence,linked_camera_views=len(evidence),view_agreement=agree,
                    fused_top3=[{'species':labels[int(i)],'score':float(p[i])} for i in order[:3]],
                    status='strong_candidate_pending_review' if strong else 'unresolved_pending_review',
                    showcase_candidate=strong and labels[int(order[0])] in LABELS and 'possible_multi_tree_box' not in row['quality']['flags'],
                    best_quality_view=max(evidence,key=lambda e:e['quality_score']))
                recomputed.append(row['id'])
            kept.append(row)
        frame['instances']=kept;frame['suppressed_non_tree_count']=len(rejected)
        if rejected:
            target=gate/'overviews'/f'{frame["frame_key"]}.jpg';target.parent.mkdir(exist_ok=True)
            with Image.open(source_frames[frame['frame_key']]['source']) as im:
                preview=im.convert('RGB');preview.thumbnail((1600,800))
            draw=ImageDraw.Draw(preview)
            for row in kept:
                b=row['box'];xy=[b[0]*preview.width,b[1]*preview.height,b[2]*preview.width,b[3]*preview.height]
                draw.rectangle(xy,outline=(255,210,50),width=2)
                draw.text((xy[0]+3,xy[1]+3),row['id'].split('__tree_')[-1]+' '+str(round(row['detector_score']*100))+'%',fill='yellow')
            preview.save(target,quality=90)
            frame['overview']=target.relative_to(out).as_posix()
    stats={'suppressed_anchor_ids':removed,'suppressed_anchor_count':len(removed),
        'recomputed_after_excluding_background_evidence':recomputed,'raw_anchor_files_modified':False}
    write(gate/'presentation_audit.json',stats)
    return stats


def ledger(out):
    path=ROOT/'hk-tree-vmms/derived/exhaustive_species_frame_labeler/runtime/frame_review_state.json'
    frames=read_json(path);records=[];bauhinia_notes=[]
    for key,frame in frames.items():
        for row in frame.get('labels',[]):
            name=row['species'];note=row.get('note','')
            if name.startswith(('Livistona chinensis','Aleurites moluccana','Bauhinia')) or '羊蹄甲' in name:
                entry={'frame_key':key,'label_id':row['label_id'],'original_species':name,
                    'frame_submitted_at':frame.get('updated_at_utc'),'frame_status':frame.get('frame_status'),
                    'assistant_prelabel_in_note':'助手预标' in note,'original_note':note,
                    'source_file':str(path),'source_sha256':sha256(path) if not records else records[0]['source_sha256'],
                    'training_eligible':False,'individual_human_identity_review_verified':False}
                if name.startswith('Livistona chinensis'):
                    entry.update(user_self_reported_confidence=.70,review_status='provisional_palm_label_needs_independent_check')
                elif name.startswith('Aleurites moluccana'):
                    entry.update(review_status='existing_label_not_confirmed_by_latest_user_message')
                else:
                    entry.update(user_requested_species='Bauhinia x blakeana',review_status='manual_scope_correction_requires_provenance_check')
                records.append(entry)
            if ('Bauhinia' in note or '羊蹄甲' in note) and not name.startswith('Bauhinia'):
                bauhinia_notes.append({'frame_key':key,'label_id':row['label_id'],'current_species':name,'note':note,
                    'status':'historical_inventory_or_proposal_note_not_a_manual_Bauhinia_label'})
    summary={'records':records,'historical_bauhinia_conflicts':bauhinia_notes,
        'user_clarifications':{'manual_sheep_hoof_labels_intended_as':'Bauhinia x blakeana',
            'livistona_confidence_is_user_estimate_not_calibrated':.70,
            'white_acacia_ambiguous_term':'Audit both Melaleuca cajuputi and Acacia confusa; no new taxon alias invented'},
        'manual_Bauhinia_label_matches_in_current_saved_field':sum('user_requested_species' in r for r in records),
        'warning':'A submitted frame is NOT proof every retained instance was independently identified by a person. Assistant prelabels are not gold species truth.',
        'originals_modified':False}
    write(out/'label_review_ledger.json',summary)
    with (ROOT/'hk-tree-vmms/outputs/coordinates/nearby_trees.csv').open(encoding='utf-8-sig') as f:inventory=list(csv.DictReader(f))
    write(out/'inventory_focus_summary.json',{s:[r for r in inventory if r['species_name'].startswith(s)] for s in LABELS})
    return summary


def main(out,gate=None):
    active=out/'active_object_gate.json'
    if gate is None and active.exists():
        gate=(out/read_json(active)['directory']).resolve()
        if not gate.is_relative_to(out.resolve()):raise ValueError('Gate must be within output directory')
    audit=ledger(out)
    plan=read_json(out/'plan.json')
    frames=[read_json(out/'anchors'/f'{key}.json') for key in plan['anchors'] if (out/'anchors'/f'{key}.json').exists()]
    gating=apply_gate(out,frames,gate) if gate else None
    allrows=[r for f in frames for r in f['instances']]
    summary={'completed':len(frames),'planned':len(plan['anchors']),'candidate_count':len(allrows),
        'focus_candidates':sum(r['fused_top3'][0]['species'] in LABELS for r in allrows),
        'strong_focus_candidates':sum(r['showcase_candidate'] for r in allrows),
        'linked_candidates':sum(r['linked_camera_views']>1 for r in allrows),
        'empty_frames':sum(f['empty_detector_result'] for f in frames)}
    for f in frames:
        f.setdefault('overview','frames/'+f['frame_key']+'/overview.jpg')
        camera=read_json(out/'frames'/f['frame_key']/'result.json')['camera'];f['camera']=camera
        for r in f['instances']:
            r['crop_url']=Path(r['crop']).relative_to(out).as_posix()
            translations={
                'Inspect papery bark and leaves; do not rely on dark crown alone':'檢查紙狀樹皮與葉片；不要僅憑偏暗樹冠判斷',
                'Inspect phyllodes; compare non-acacia alternatives':'檢查葉狀柄；同時比較非相思類候選'}
            r['special_checks']=[s+' / '+translations[s] if s in translations else s for s in r['special_checks']]
            for e in r['evidence']:e['crop_url']=Path(e['crop']).relative_to(out).as_posix()
    data={'summary':summary,'frames':frames,'taxa':TAXA}
    encoded=json.dumps(data,ensure_ascii=False).replace('<','\u003c')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tree Atlas · 20 m review / 二十米巡查</title>
<style>body{font:16px system-ui;background:#f3f6f0;color:#183e33;margin:0}header,main{max-width:1450px;margin:auto;padding:24px}header{background:white;border-bottom:1px solid #ccd9cb}h1{font-size:27px;margin:0 0 12px}p{line-height:1.6}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#e8f0e7;padding:12px;border-radius:8px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}select,button,a.button{font:inherit;padding:10px;background:white;border:1px solid #bacdbd;border-radius:7px;color:#183e33}button{cursor:pointer}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}.card{background:white;padding:18px;border:1px solid #cbd7c8;border-radius:12px}.card img{width:100%;height:290px;object-fit:contain;background:#e8eee7}.card.strong{border:2px solid #3f8666}.muted{font-size:14px;color:#5b6f65}.warning{padding:12px;background:#fff1db;border-radius:8px}.other{background:#fafbfa}summary{cursor:pointer}#overview{width:100%;max-height:550px;object-fit:contain;background:#172c24}a{color:#226747}.evidence{display:flex;gap:6px;overflow:auto}.evidence img{height:150px;width:150px}li{margin:8px 0}h3{margin:8px 0}.empty{padding:35px;background:white;text-align:center}</style>
<header><h1>Tree Atlas · 20 m review / 二十米巡查</h1><p>Original BioCLIP teacher · native crops + nearby views / 原始教師 · 原始裁片＋鄰近視角</p><div class="stats" id="stats"></div><p class="warning">Model suggestions, not verified species or detection accuracy. All uncertain and missed cases remain in the records. / 以下為模型候選，不是已驗證樹種或偵測準確率；不確定與未偵測案例均保留。</p><p><a href="sr_quality_trial/comparison.html">Restoration comparison / 修復對照</a> · <a href="label_review_ledger.json">Label provenance / 標註溯源</a> · <a href="google_manual_checks.json">Street View checks / 街景核驗紀錄</a></p></header>
<main><div class="controls"><label>Species / 樹種 <select id="species"><option value="all">All predictions / 全部預測</option value="focus">Focus eight / 重點八類</option></select></label><label>View / 檢視 <select id="view"><option value="frames">20 m frame sequence / 二十米逐幀</option><option value="highlights">Higher-score focus cases / 重點類別高分案例</option><option value="focus">All focus candidates / 全部重點候選</option></select></label><label>Route / 序列 <select id="route"><option value="all">All / 全部</option></select></label></div><div id="navigation" class="controls"><button id="prev">← Previous / 上一幀</button><select id="frame"></select><button id="next">Next / 下一幀 →</button></div><section id="scene"><h2 id="frameTitle"></h2><img id="overview"><p id="frameInfo" class="muted"></p></section><p id="count"></p><div class="cards" id="cards"></div></main>
<script id="payload" type="application/json">'''+encoded+'''</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),names=Object.fromEntries(data.taxa.map(t=>[t[0],t[1]]));
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const s=data.summary;$('stats').innerHTML=[`${s.completed}/${s.planned} anchor frames / 主幀`,`${s.candidate_count} candidates / 候選`,`${s.focus_candidates} focus candidates / 重點候選`,`${s.strong_focus_candidates} higher-score focus / 重點高分`,`${s.linked_candidates} tentative multiview links / 暫定多視角`].map(t=>`<div class="stat">${t}</div>`).join('');
for(const t of data.taxa)$('species').add(new Option(`${t[0]} / ${t[1]}`,t[0]));
for(const r of [...new Set(data.frames.map(f=>f.frame_key.split('__')[0]))])$('route').add(new Option(r,r));
let available=[];
function selectedFrames(){return data.frames.filter(f=>$('route').value==='all'||f.frame_key.startsWith($('route').value+'__'));}
function matches(r){const val=$('species').value,sp=r.fused_top3[0].species;return val==='all'||(val==='focus'?sp in names:sp===val);}
function label(s){return `${s}${names[s]?' / '+names[s]:''}`;}
function card(r){const top=r.fused_top3[0],q=r.quality;return `<article class="card ${r.showcase_candidate?'strong':''}"><h3>${esc(label(top.species))}</h3><p class="muted">${esc(r.id)}</p><a target="_blank" href="${esc(r.crop_url)}"><img loading="lazy" src="${esc(r.crop_url)}"></a><ol>${r.fused_top3.map(p=>`<li>${esc(label(p.species))} · ${(p.score*100).toFixed(1)}%</li>`).join('')}</ol><p class="muted">Model scores, not calibrated correctness / 模型分數，不是校準後正確率</p><p>${r.showcase_candidate?'Higher-score candidate · awaiting review / 高分候選，待驗收':'Unresolved or non-focus candidate / 待核驗或非重點候選'}</p><p class="muted">${q.width}×${q.height} native pixels · dark ${(q.dark_fraction_lt35*100).toFixed(0)}% · ${r.linked_camera_views} camera view(s)<br>原始像素 · 暗區比例 · 相機視角數</p>${r.special_checks.length?'<p class="warning">'+esc(r.special_checks.join('; '))+'</p>':''}<details><summary>View evidence / 查看視角證據</summary><p class="muted">Associations are tentative; background can also match. / 關聯為暫定，背景亦可能匹配。</p><div class="evidence">${r.evidence.map(e=>`<a href="${esc(e.crop_url)}" target="_blank"><img loading="lazy" src="${esc(e.crop_url)}" title="${esc(e.frame_key)}"></a>`).join('')}</div></details><p><a href="${esc(r.google_viewpoint_url)}" target="_blank" rel="noopener">Check Google Street View / 開啟 Google 街景核驗</a></p><p class="muted">Link is not proof of a verified tree or a closer camera. / 連結不代表已核驗同株，亦不保證相機更近。</p></article>`;}
function render(){const mode=$('view').value;available=selectedFrames();$('scene').hidden=mode!=='frames';$('navigation').hidden=mode!=='frames';let rows=[];
if(mode==='frames'){const frame=available.find(f=>f.frame_key===$('frame').value)||available[0];if(!frame){$('cards').textContent='No frames / 無影像';return;}$('frameTitle').textContent=frame.frame_key;$('overview').src=frame.overview;$('frameInfo').textContent=`Route distance / 路程 ${frame.camera.distance_m.toFixed(1)} m · ${frame.instances.length} candidates / 候選 · ${frame.neighbor_frames.length} additional camera frames / 額外視角`;rows=frame.instances.filter(matches);}else{rows=available.flatMap(f=>f.instances).filter(matches).filter(r=>mode==='highlights'?r.showcase_candidate:r.fused_top3[0].species in names);rows.sort((a,b)=>b.fused_top3[0].score-a.fused_top3[0].score);}
$('count').textContent=`${rows.length} displayed candidates / 個顯示候選。Filtering is presentation only, not an accuracy calculation / 篩選僅供檢視，不用來計算準確率。`;$('cards').innerHTML=rows.map(card).join('')||'<div class="empty">No matching candidates. Not evidence that no trees exist. / 沒有符合条件的候選，不代表現場沒有樹。</div>';}
function refill(){const keep=$('frame').value;$('frame').replaceChildren();for(const f of selectedFrames())$('frame').add(new Option(f.frame_key,f.frame_key));if([...$('frame').options].some(o=>o.value===keep))$('frame').value=keep;render();}
for(const id of ['species','view','frame'])$(id).onchange=render;$('route').onchange=refill;
$('prev').onclick=()=>{$('frame').selectedIndex=Math.max(0,$('frame').selectedIndex-1);render();};$('next').onclick=()=>{$('frame').selectedIndex=Math.min($('frame').options.length-1,$('frame').selectedIndex+1);render();};refill();
</script></html>'''
    page=page.replace('<style>','<style>[hidden]{display:none!important}')
    if gate:
        count=gating['suppressed_anchor_count']
        page=page.replace('<main>',f'<p class="warning" style="max-width:1400px;margin:16px auto">Object/background checks enabled: {count} non-tree candidates suppressed; originals retained. / 已啟用樹木與背景復核：排除 {count} 個非樹候選，保留原始紀錄。<br>Passing this check is not proof of a tree or species. / 通過復核仍不代表已確認樹木或樹種。</p><main>')
        page=page.replace('${frame.instances.length} candidates / 候選','${frame.instances.length} candidates / 候選 · ${frame.suppressed_non_tree_count||0} background suppressed / 已排除背景')
        summary['object_gate']=str(gate)
        summary['suppressed_non_tree']=count
    (out/'index.html').write_text(page.replace('符合条件','符合條件'),encoding='utf8')
    write(out/'viewer_summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gate',type=Path)
    args=p.parse_args();main(args.out.resolve(),args.gate.resolve() if args.gate else None)

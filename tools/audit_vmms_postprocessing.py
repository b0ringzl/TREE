"""Read-only effect accounting and separate local human-audit artifact. No fitting."""
import hashlib
import json
from pathlib import Path
import argparse
from collections import Counter
import html
from distillation_core import read_json
from run_vmms20m_teacher import write

def main(root,out):
    out.mkdir(parents=True,exist_ok=False)
    gate=read_json(root/read_json(root/'active_object_gate.json')['directory']/'decisions.json')
    blocked={r['id'] for r in gate if r['decision']=='suppress_non_tree'}
    frames=[read_json(p) for p in sorted((root/'anchors').glob('*.json'))]
    rows=[r for f in frames for r in f['instances'] if r['id'] not in blocked]
    linked=[r for r in rows if r['linked_camera_views']>1]
    changed=[r for r in linked if r['top3'][0]['species']!=r['fused_top3'][0]['species']]
    # This audit uses only retained evidence. Fail closed if fused probabilities need a different reconstruction.
    if any(e['instance_id'] in blocked for r in rows for e in r['evidence']):raise ValueError('Recompute cleaned multiview probabilities first')
    dark=[]
    for r in rows:
        enhanced=next((v for v in r['augmentation_predictions'] if v['variant']=='bounded_shadow_lift'),None)
        if enhanced:dark.append({'id':r['id'],'original':r['original_top3'],'enhanced':enhanced['top3']})
    stats={'retained_anchor_observations':len(rows),'multiview_observations':len(linked),
        'multiview_top1_changed':len(changed),'multiview_top1_unchanged':len(linked)-len(changed),
        'multiview_max_score_increased':sum(r['fused_top3'][0]['score']>r['top3'][0]['score'] for r in linked),
        'dark_enhanced_anchor_crops':len(dark),
        'dark_enhancement_top1_changed':sum(r['original'][0]['species']!=r['enhanced'][0]['species'] for r in dark),
        'dark_enhancement_max_score_increased':sum(r['enhanced'][0]['score']>r['original'][0]['score'] for r in dark),
        'species_truth_available_for_these_comparisons':False,
        'google_scenes_checked':len(read_json(root/'google_manual_checks.json')['checks']),
        'google_species_confirmed':read_json(root/'google_manual_checks.json')['google_species_verifications_completed'],
        'sr_trial_unique_images':len({r['crop_id'] for r in read_json(root/'sr_quality_trial/results.json')}),
        'warning':'Prediction changes and score changes are not accuracy changes. Single-view baseline already includes same-image augmentations; multiview comparison does not isolate restoration.',
        'training':False}
    write(out/'effect_counts.json',stats);write(out/'multiview_changes.json',changed);write(out/'dark_changes.json',dark)
    coordinates=read_json(root/'frames.json');selected=[];used=[]
    def add(row,stratum):
        frame=row['id'].split('__tree_')[0];meta=coordinates[frame]
        if any(meta['stream_id']==s and abs(meta['distance_m']-d)<60 for s,d in used):return False
        selected.append({'row':row,'stratum':stratum});used.append((meta['stream_id'],meta['distance_m']));return True
    # Two known failure modes are deliberate diagnostic inclusions, not random evaluation evidence.
    for key in ['hewentian_pano__001441__tree_001','jianshazui_pano_1__002747__tree_004']:
        match=next((r for r in changed if r['id']==key),None)
        if match:add(match,'changed_priority')
    def stable_order(items):return sorted(items,key=lambda r:hashlib.sha256(('20260910'+r['id']).encode()).hexdigest())
    for r in stable_order(changed):
        if len(selected)>=6:break
        add(r,'changed')
    before=len(selected)
    for r in stable_order([r for r in linked if r not in changed]):
        if len(selected)>=before+4:break
        add(r,'unchanged_control')
    before=len(selected)
    focus={'Melaleuca cajuputi','Bauhinia x blakeana','Aleurites moluccana','Acacia confusa'}
    candidates=sorted(rows,key=lambda r:(r['fused_top3'][0]['species'] not in focus,r['fused_top3'][0]['score'],r['id']))
    for r in candidates:
        if len(selected)>=before+2:break
        if r['fused_top3'][0]['score']<.65:add(r,'low_confidence_google')
    cases=[]
    for i,choice in enumerate(selected,1):
        r=choice['row'];cases.append({'number':i,'id':r['id'],'stratum':choice['stratum'],
            'images':[{'url':'../'+Path(e['crop']).relative_to(root).as_posix(),'label':('Anchor / 主幀' if j==0 else f'Neighbor {j} / 鄰近視角 {j}')} for j,e in enumerate(r['evidence'])],
            'original':r['top3'],'processed':r['fused_top3'],'google':r['google_viewpoint_url'],
            'caution':'First check whether views show the SAME target, not neighboring trees. / 請先確認各視角是同一棵目標樹，而非鄰樹。'})
    sr=read_json(root/'sr_quality_trial/results.json')
    for key in ['0003','0016']:
        variants=[r for r in sr if r['crop_id']==key]
        original=next(r for r in variants if r['variant']=='original')
        superres=next(r for r in variants if 'realesrgan' in r['variant'].lower())
        cases.append({'number':len(cases)+1,'id':'restoration_'+key,'stratum':'restoration_diagnostic',
            'images':[{'url':'../'+Path(r['file']).relative_to(root).as_posix(),'label':r['variant']} for r in variants],
            'original':original['top3'],'processed':superres['top3'],
            'google':read_json(root/'google_manual_checks.json')['checks'][0]['url'] if key=='0016' else None,
            'caution':'Generated texture is not real botanical evidence. / 生成的紋理不是新的植物學證據。'})
    write(out/'manifest.json',{'selection':'Diagnostic panel: 6 changed (including 2 deliberate failure cases), 4 unchanged controls, 2 low-confidence cases, 2 restoration cases; route spacing >=60m reduces repetition but does not prove unique trees.',
        'unbiased_evaluation':False,'cases':cases,'reference_labels_hidden_until_user_reveals':True,'training_eligible':False})
    data=json.dumps(cases,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Postprocessing audit / 後處理核驗</title>
<style>body{font:16px system-ui;background:#f3f6f0;color:#193f33;margin:28px auto;max-width:1400px;padding:0 20px}button,input,select,textarea{font:inherit;padding:10px;margin:5px;border:1px solid #adc3b4;border-radius:6px}button{cursor:pointer;background:#227556;color:white}article{background:white;padding:22px;margin:22px 0;border-radius:12px;border:1px solid #c9d7cc}.images{display:flex;gap:12px;overflow:auto}.image{min-width:230px;max-width:380px;flex:1}.image img{width:100%;height:300px;object-fit:contain;background:#eaf0e8}.note{background:#fff1d7;padding:15px}summary{cursor:pointer}textarea{width:90%}label{display:inline-block}a{color:#216747}</style>
<h1>Postprocessing audit / 後處理核驗</h1><p>14 diagnostic cases, not a random accuracy test. Identify the tree before revealing model answers. / 14 組診斷樣本，不是隨機準確率測試。請先辨認，再展開模型答案。</p>
<p class="note">Uncertain is a valid answer. Do not guess. Confirm tree presence and same-tree association before species. / 可以填「無法確定」，不必猜測；先確認是否為樹、是否同株，再填樹種。<br>Saved only in this browser; export JSON and send it back after review. No labels or weights are modified. / 僅儲存在本瀏覽器；完成後匯出 JSON 交回。此頁不修改標籤或權重。</p>
<button id="export">Export review / 匯出核驗</button><span id="status"></span><main id="cases"></main>
<script id="data" type="application/json">'''+data+'''</script><script>
const cases=JSON.parse(document.getElementById('data').textContent),key='vmms-postprocess-audit-20260910-v1';let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};
const esc=x=>String(x||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('cases').innerHTML=cases.map(c=>`<article data-id="${c.id}"><h2>${c.number}. ${c.id}</h2><p>${esc(c.caution)}</p><div class="images">${c.images.map(im=>`<div class="image"><p>${esc(im.label)}</p><a target="_blank" href="${im.url}"><img src="${im.url}" loading="lazy"></a></div>`).join('')}</div>
<label>Tree present? / 有目標樹嗎？<select data-field="tree"><option value="">Select / 請選擇</option><option value="yes">Yes / 有</option><option value="no">No / 沒有</option><option value="uncertain">Uncertain / 無法確定</option></select></label>
<label>Same tree across views? / 是否同一棵樹？<select data-field="same_tree"><option value="">Select / 請選擇</option><option value="yes">Yes / 是</option><option value="no">No / 不是</option><option value="uncertain">Uncertain / 無法確定</option><option value="not_applicable">Single image or restoration / 單圖或修復對照</option></select></label>
<p><label>Species / 樹種 <input data-field="species" placeholder="Name, or cannot determine / 名稱或無法確定"></label><label>Certainty / 把握<select data-field="certainty"><option value="">Select / 請選擇</option><option value="sure">Certain / 確定</option><option value="probable">Probable / 較可能</option><option value="uncertain">Uncertain / 無法確定</option></select></label></p>
<textarea data-field="evidence" placeholder="Which view or feature helped? Google capture date / 哪個視角或特徵有幫助？谷歌拍攝日期"></textarea>
${c.google?`<p><a href="${esc(c.google)}" target="_blank" rel="noopener">Open Street View / 開啟街景核對</a> — coordinate/heading link only, not verified identity / 僅為座標與方向連結，尚未確認同株</p>`:''}
<details><summary>Reveal model answers after your judgment / 判斷後再查看模型答案</summary><p>Before / 處理前：${c.original.map(p=>esc(p.species)+' '+(p.score*100).toFixed(1)+'%').join('；')}</p><p>After (SR for restoration cases) / 處理後（修復組顯示超解析度）：${c.processed.map(p=>esc(p.species)+' '+(p.score*100).toFixed(1)+'%').join('；')}</p><p>Scores are not correctness probabilities / 分數不是正確率</p></details></article>`).join('');
function persist(){try{localStorage.setItem(key,JSON.stringify(state));document.getElementById('status').textContent=` Saved / 已儲存 ${Object.values(state).filter(r=>r.tree||r.species).length}/${cases.length}`}catch(e){document.getElementById('status').textContent='Storage unavailable: export before leaving / 無法本地儲存：離開前請匯出'}}
for(const article of document.querySelectorAll('article')){const id=article.dataset.id;for(const el of article.querySelectorAll('[data-field]')){el.value=state[id]?.[el.dataset.field]||'';el.oninput=()=>{state[id]??={};state[id][el.dataset.field]=el.value;persist()}}article.querySelector('details').ontoggle=e=>{if(e.target.open){state[id]??={};state[id].model_revealed=true;state[id].species_before_reveal??=state[id].species||'';persist()}}}persist();
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({panel:key,reviewed_at:new Date().toISOString(),answers:state,training_eligible:false},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='vmms_postprocessing_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000)};
</script></html>'''
    (out/'index.html').write_text(page,encoding='utf8');print(json.dumps(stats,ensure_ascii=False));print('Review cases:',len(cases))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())

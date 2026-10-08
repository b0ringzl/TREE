'use strict';
const stressLink=el('a','','Low-quality experiment / 低品質補償實驗');stressLink.href='/point-stress';$('cloudPanel').append(stressLink);
const pointPanel=el('section','point-evidence');pointPanel.id='pointEvidence';pointPanel.hidden=true;$('quality').before(pointPanel);
const imageRun=$('run').onclick,originalRender=renderResult,originalUpdate=update;
const fusionPanel=el('section','fusion-comparison');fusionPanel.hidden=true;$('cards').before(fusionPanel);
function renderFusion(d){
  fusionPanel.replaceChildren();fusionPanel.hidden=!d.comparison;
  if(!d.comparison)return;
  fusionPanel.append(el('h3','','What the cloud adds / 點雲帶來的改變'));
  fusionPanel.append(el('p','','Same seven-class scope · model scores, not accuracy / 相同七類範圍・模型分數，並非準確率'));
  const grid=el('div','fusion-grid');
  for(const [key,title] of [['image','Image / 影像'],['point','Point cloud / 點雲'],['fusion','Image + cloud / 影像＋點雲']]){
    const card=el('div','fusion-branch '+key);card.append(el('h4','',title));
    const list=el('ol');
    for(const row of d.comparison[key]){const li=el('li');li.append(el('span','',row.scientific_name),el('small','',row.chinese_name),el('strong','',(row.score*100).toFixed(1)+'%'));list.append(li);}
    card.append(list);grid.append(card);
  }
  fusionPanel.append(grid);
  fusionPanel.append(el('p','quiet','50% image + 50% point scores · PTv2 · 2,048 points × 4 rotations / 影像與點雲分數各佔 50%・PTv2・2,048 點 × 4 次旋轉'));
  const details=el('details');details.append(el('summary','','Supported species & original image result / 支援樹種與原影像結果'));
  details.append(el('p','','Seven supported classes / 七個支援類別：'+d.class_scope.map(r=>r.scientific_name+' / '+r.chinese_name).join('；')));
  details.append(el('p','','Full-image model: 148 raw classes, with Bauhinia grouped for display (different scope) / 完整影像模型原有 148 類；洋紫荊只在展示時合併（與七類融合範圍不同）：'+d.image_full_catalog_top3.map(r=>r.scientific_name+' / '+r.chinese_name+' '+(r.score*100).toFixed(1)+'%').join('；')));
  details.append(el('p','','Absolute height is not fed to this classifier. / 此分類器不直接輸入絕對樹高。'));
  fusionPanel.append(details);
}
renderResult=function(){
  originalRender();const d=state.result;renderFusion(d||{});
  if(d?.point_evidence){
    renderPointEvidence(pointPanel,d.point_evidence);
    $('resultMeta').textContent+=d.fusion_used?' · Live point fusion / 即時點雲融合':' · Image-only fallback / 回退單源影像';
    $('evidenceText').textContent=d.preprocessing+' '+d.image_score_policy;
  }else pointPanel.hidden=true;
};
let pairedImage=null;
update=function(){const file=state.items[state.active]?.file;if(file!==pairedImage){$('paired').checked=false;pairedImage=file;}originalUpdate();['cloud','cloudUnits','cloudZUp','paired'].forEach(id=>$(id).disabled=state.busy);};
$('paired').onchange=()=>{invalidate();update();};
$('cloud').onchange=()=>{$('cloudName').textContent=$('cloud').files[0]?.name||'No cloud selected / 尚未選擇點雲';$('paired').checked=false;invalidate();update();};
['cloudUnits','cloudZUp'].forEach(id=>$(id).onchange=()=>{invalidate();update();});
$('run').onclick=async()=>{
  if(source()==='image')return imageRun();
  const item=state.items[state.active],cloud=$('cloud').files[0];if(state.busy||!item||!cloud||!$('paired').checked)return;
  if(cloud.size>64*1024*1024){error('Cloud limit: 64 MB / 點雲上限 64 MB');return;}
  const body=new FormData();body.append('image',item.file);body.append('cloud',cloud);body.append('paired','true');body.append('units',$('cloudUnits').value);body.append('z_up',String($('cloudZUp').checked));body.append('scenario',scenario());body.append('projection',item.projection||'auto');
  state.busy=true;invalidate();error('');update();const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),180000);
  try{const r=await fetch('/api/predict-with-cloud',{method:'POST',body,signal:controller.signal}),d=await readApiResponse(r);if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'Request failed / 請求失敗');state.result={...d,input_name:item.file.name};renderResult();}
  catch(e){error(e.name==='AbortError'?'Request timed out / 請求逾時':e.message);}
  finally{clearTimeout(timer);state.busy=false;update();}
};
const about=$('about');const policy=el('p','','Live PTv2 classification and equal-weight image–point score fusion are enabled for seven classes. Image-only retains 148 classes. Model scores are not calibrated accuracy. / 已接通七類 PTv2 即時分類及影像與點雲等權分數融合；單源影像保留 148 類，模型分數並非已校準準確率。');about.append(policy);
update();
async function loadPointExample(){const key=new URLSearchParams(location.search).get('pair');if(!key)return;
  try{const catalog=await fetch('/api/point-assist/catalog').then(r=>r.json()),row=catalog.items.find(r=>r.key===key);if(!row)throw Error('Pair unavailable / 配對不可用');
    const [im,pc]=await Promise.all(['image','cleaned'].map(kind=>fetch('/api/point-assist/asset/'+key+'/'+kind)));if(!im.ok||!pc.ok)throw Error('Pair asset unavailable / 配對資料不可用');
    await addFiles([new File([await im.blob()],'single-tree.jpg',{type:'image/jpeg'})]);state.items[state.active].projection='photo';state.items[state.active].organ='whole';
    const dt=new DataTransfer();dt.items.add(new File([await pc.blob()],'cleaned_local.npz'));$('cloud').files=dt.files;$('cloudName').textContent='cleaned_local.npz';$('cloudUnits').value='metres';$('cloudZUp').checked=true;document.querySelector('input[name=source][value=image_point]').checked=true;$('paired').checked=false;update();
  }catch(e){error(e.message);}}
loadPointExample();

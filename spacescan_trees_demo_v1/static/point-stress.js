'use strict';
const $=id=>document.getElementById(id);let dataset=null,timer=null,requestSerial=0;
const node=(t,txt,cls)=>{const n=document.createElement(t);if(txt)n.textContent=txt;if(cls)n.className=cls;return n;};
function name(s){return dataset.names.find(n=>n.scientific_name===s)||{scientific_name:s,chinese_name:'',name:s};}
function decision(target,title,d){const n=name(d.prediction);target.replaceChildren(node('p',title),node('h3',n.scientific_name),node('p',n.chinese_name),node('div',(100*Math.max(...d.scores)).toFixed(1)+'%','stress-score'),node('p',d.correct?'Matches saved label / 符合既有標籤':'Differs from saved label / 與既有標籤不符',d.correct?'stress-match':'stress-mismatch'));}
function render(){const id=$('stressSample').value,condition=$('stressCondition').value,row=dataset.records.find(r=>r.id===id&&r.condition===condition);if(!row)return;
  $('stressImage').src=row.image;$('stressCaption').textContent=dataset.conditions.find(c=>c.id===condition).name+' · '+row.quality.min_dimension+' px min side / 最短邊';$('stressTruth').textContent='Saved label / 既有標籤: '+name(row.truth).name;
  decision($('imageDecision'),'Image only / 影像單源',row.outputs.yolo_image);decision($('fusionDecision'),'Image + point / 影像＋點雲',row.counterfactuals.correct);
  const m=dataset.metrics[condition],b=dataset.original_results[condition],u=dataset.uncertainty[condition].image;
  $('stressImpact').textContent='This condition / 此條件：'+Math.round(b.yolo_image.accuracy*10)+'/10 → '+Math.round(m.correct.accuracy*10)+'/10 correct / 正確';
  $('stressStats').textContent='Corrected / 糾正 '+u.corrected+' · Regressed / 由對變錯 '+u.regressed+' · OA difference / 準確率差值 '+u.delta_pp.toFixed(1)+' pp · Exploratory 95% interval / 探索性 95% 區間 ['+u.ci95_pp.map(v=>v.toFixed(1)).join(', ')+'] pp';
  document.querySelectorAll('#stressTable tr').forEach(r=>r.classList.toggle('selected',r.dataset.condition===condition));
  $('stressHash').textContent='Point SHA-256, unchanged across conditions / 各條件相同點雲 SHA-256: '+row.point_sha256;
  const serial=++requestSerial;fetch('/api/point-assist/record/'+row.catalog_key).then(r=>{if(!r.ok)throw Error('Cloud unavailable / 點雲不可用');return r.json();}).then(d=>{if(serial===requestSerial){renderPointEvidence($('stressCloud'),d.evidence);$('stressCloud').querySelector('.pe-policy').textContent='The cloud stays unchanged across image conditions. Recorded experimental fusion scores appear above; the main identification model has not changed. / 各影像條件的點雲保持相同。上方顯示已記錄的實驗融合分數；主識別模型未變更。';}}).catch(error);
}
function error(e){$('stressError').textContent=e.message;$('stressError').hidden=false;}
$('stressCondition').onchange=render;$('stressSample').onchange=render;
$('playStress').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('playStress').textContent='▶ Compare levels / 播放品質對照';return;}$('playStress').textContent='Ⅱ Pause / 暫停';timer=setInterval(()=>{$('stressCondition').selectedIndex=($('stressCondition').selectedIndex+1)%dataset.conditions.length;render();},2200);};
fetch('/api/point-stress').then(r=>{if(!r.ok)throw Error('Experiment unavailable / 實驗不可用');return r.json();}).then(d=>{dataset=d;const originals=d.records.filter(r=>r.condition==='original');originals.forEach((r,i)=>$('stressSample').add(new Option((i+1)+' · '+r.id,r.id)));d.conditions.forEach(c=>$('stressCondition').add(new Option(c.name,c.id)));
 d.conditions.forEach(c=>{const tr=node('tr');tr.dataset.condition=c.id;const td=node('td'),b=node('button',c.name);b.onclick=()=>{$('stressCondition').value=c.id;render();};td.append(b);tr.append(td);const m=d.metrics[c.id],o=d.original_results[c.id];[o.yolo_image,m.correct,m.wrong_pair,m.train_mean,m.zero,o.bioclip_image,o.bioclip_assisted].forEach(v=>tr.append(node('td',Math.round(v.accuracy*10)+'/10')));$('stressTable').append(tr);});render();}).catch(error);

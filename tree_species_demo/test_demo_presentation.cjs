const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {decisionView,decisionHTML}=require('./static/decision-view.js');
const base={label:'Unknown',display_name:'Unknown / 未知',score:.42,source:'Image / 影像',reasons:['Review input quality / 請複核輸入品質'],top3:[{name:'A / 甲',score:.42},{name:'B / 乙',score:.3},{name:'C / 丙',score:.12}]};
const before=JSON.stringify(base),v=decisionView(base),h=decisionHTML(v);
const outside=decisionHTML(decisionView({...base,top3:[{name:'Bauhinia x blakeana',score:.937,in_catalog:false}]}));
assert.ok(outside.includes('Bauhinia x blakeana'));assert.ok(!/Outside catalog|清單外|其他候選/.test(outside));assert.ok(outside.includes('93.7%'));
assert.equal(JSON.stringify(base),before);assert.equal(v.candidates.length,3);
assert.ok(h.includes('42.0%')&&h.includes('30.0%')&&h.includes('12.0%'));assert.ok(!/Unknown|未知/.test(h));assert.ok(h.includes(base.reasons[0]));
assert.ok(!decisionView({...base,label:'A',display_name:'A / 甲'}).review);
assert.ok(decisionHTML(decisionView({...base,top3:[],score:null})).includes('Prediction unavailable'));assert.ok(!decisionHTML(decisionView({...base,top3:[],score:null})).includes('<li>'));
assert.ok(!decisionHTML(decisionView({...base,top3:[{name:'<script>alert(1)</script>',score:.4}]})).includes('<script>'));
assert.ok(decisionView({...base,display_name:'Source disagreement / 來源分歧'}).review);
const html=fs.readFileSync(path.join(__dirname,'static/demo.html'),'utf8');assert.ok(!/id="sample"|sampleRun|samplebar|run an example/.test(html));
const ids=[...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const elements=Object.fromEntries(ids.map(id=>[id,{value:'',hidden:false,files:[],classList:{toggle(){}},getBoundingClientRect:()=>({width:320,height:250}),getContext:()=>({fillRect(){},fillText(){}}),removeAttribute(){}}]));
async function main(){
 const url='http://127.0.0.1:8037';
 const catalog=await (await fetch(url+'/api/ui/catalog')).json(),health=await(await fetch(url+'/api/health')).json();
 const samples=await(await fetch(url+'/api/ui/samples')).json();
 const sample=samples.find(s=>s.domain==='hongkong_vmms');
 const response=await fetch(url+'/api/sample/'+sample.id,{method:'POST',body:new URLSearchParams({mode:'image'})});assert.equal(response.status,200);const actual=await response.json();
 elements.domain.value=catalog.models[0].key;
 const context=vm.createContext({document:{getElementById(id){assert.ok(elements[id],'Missing element '+id);return elements[id]}},window:{},location:{search:''},URLSearchParams,devicePixelRatio:1,console,
  fetch:async(route)=>({ok:true,json:async()=>route==='/api/ui/catalog'?catalog:health}),decisionView,decisionHTML});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'static/demo.js'),'utf8'),context);await new Promise(setImmediate);
 assert.ok(elements.health.textContent.includes('Local models'));assert.ok(!elements.error.textContent);
 context.actual=actual;vm.runInContext('render(actual)',context);
 assert.equal(elements.result.hidden,false);assert.ok(elements.decision.innerHTML.includes('Most likely candidates'));assert.ok(!/Unknown|未知/.test(elements.decision.innerHTML));
 for(const c of actual.presentation.decision.top3)assert.ok(elements.decision.innerHTML.includes((c.score*100).toFixed(1)+'%'));
 const exported=vm.runInContext('state.result',context);assert.equal(exported.main_display.candidates.length,3);assert.equal(exported.decision.label,actual.presentation.decision.label);
 vm.runInContext('clear()',context);assert.equal(elements.result.hidden,true);
 const liveHTML=await(await fetch(url+'/')).text();assert.ok(!liveHTML.includes('samplebar'));assert.ok(liveHTML.includes('decision-view.js?v=3'));
 console.log('PASS: removed example UI; known, candidate, no-score and disagreement views; unchanged scores; HTML escaping; full script init/render/clear; live inference and served HTML.');
}
main().catch(e=>{console.error(e);process.exitCode=1});

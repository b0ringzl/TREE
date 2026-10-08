const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync(__dirname+'/static/herbarium-review.html','utf8');
const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{value:'',dataset:{},classList:{toggle(){}},replaceChildren(){},add(){}}]));
elements.status.value='pending';
let keyListener, fail=false, writes=0;
const database=['one','two','three'].map(id=>({id,source_species:'Ficus microcarpa',source_names:['榕樹'],human_review:{status:'pending',revision:0},organ_review:{organ_tags:['tree_or_crown_context']},candidate_lane:'tree_context_candidate',image_url_local:'/image/'+id,source_page:'https://example.test'}));
const context=vm.createContext({document:{getElementById:id=>elements[id],addEventListener:(event,fn)=>{if(event==='keydown')keyListener=fn;}},Option:function(){},Image:function(){},localStorage:{getItem(){return null},setItem(){}},console,
fetch:async(url,options)=>{
 if(!options)return {ok:true,json:async()=>({rows:structuredClone(database)})};
 writes++; if(fail)return {ok:false,json:async()=>({detail:'Fixture save failure'})};
 const r=database.find(r=>url.endsWith('/'+r.id)), action=JSON.parse(options.body);
 assert.equal(action.expected_revision,r.human_review.revision);
 const previous=r.human_review.status;r.human_review={status:action.status,revision:r.human_review.revision+1};
 return {ok:true,json:async()=>({id:r.id,...r.human_review,previous_status:previous})};
}});
const run=code=>vm.runInContext(code,context),tick=()=>new Promise(setImmediate);
function key(key,extra={}){keyListener({key,repeat:false,altKey:false,ctrlKey:false,metaKey:false,target:{closest:()=>false},preventDefault(){},...extra});}
async function main(){
 run(fs.readFileSync(__dirname+'/static/herbarium-review.js','utf8'));await tick();
 assert.equal(run('current'),'one');key('d');await tick();
 assert.equal(database[0].human_review.status,'rejected');assert.equal(run('current'),'two');
 key('a');await tick();assert.equal(run('current'),'three');
 key('z');await tick();assert.equal(run('current'),'two');assert.equal(database[1].human_review.status,'pending');
 key('z');await tick();assert.equal(run('current'),'one');assert.equal(database[0].human_review.status,'pending');
 const count=writes;key('d',{repeat:true});key('d',{target:{closest:()=>true}});key('d',{ctrlKey:true});await tick();assert.equal(writes,count);
 key('ArrowRight');assert.equal(run('current'),'two');key('ArrowLeft');assert.equal(run('current'),'one');
 fail=true;key('Delete');await tick();assert.equal(run('current'),'one');assert.equal(database[0].human_review.status,'pending');assert.match(elements.feedback.textContent,/Fixture save failure/);
 fail=false;key('a');key('d');await tick();assert.equal(writes,count+2);assert.equal(run('current'),'two');
 await run('load()');assert.equal(run("rows[0].human_review.status"),'accepted');
 elements.status.value='accepted';elements.status.onchange();assert.equal(run('current'),'one');key('d');await tick();assert.equal(run('current'),null);assert.equal(elements.photo.hidden,true);
 key('z');await tick();assert.equal(run('current'),'one');assert.equal(elements.photo.hidden,false);
 console.log('PASS: A / D / Delete, automatic next without skipping, arrows, undo, filter recovery, save failure, repeat/focus/busy guards, disk reload and empty list. Fixture only; no user labels changed.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});

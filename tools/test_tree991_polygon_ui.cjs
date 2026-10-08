// DOM stubs exercise event/state transitions without opening or controlling a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),listeners={};
function element(id){if(!elements.has(id))elements.set(id,{id,value:'replace',textContent:'',style:{},classList:{toggle(){}},parentElement:{append(){}},after(){},replaceChildren(){},append(){},clientWidth:800,clientHeight:600,getBoundingClientRect(){return {left:0,top:0,width:800,height:600}},getContext(){return {}},setPointerCapture(){}});return elements.get(id)}
const sandbox={console,structuredClone,performance:{now:()=>1000},requestAnimationFrame:()=>1,fetch:()=>new Promise(()=>{}),document:{getElementById:element,querySelectorAll:()=>[],createElement:()=>element(Math.random())},window:{addEventListener:(k,v)=>listeners[k]=v}};
vm.createContext(sandbox);
for(const file of ['polygon.js','app.js'])vm.runInContext(fs.readFileSync('tools/tree991_box_ui/'+file,'utf8'),sandbox);
function run(s){return vm.runInContext(s,sandbox)}
run("state={revision:2,slots:Object.fromEntries(keys.map(k=>[k,{ops:[],zmin:-10,zmax:80,confirmed:false,note:''}]))};xyz=new Float32Array([0,0,0]);updateMasks=()=>{counts.A=1};form=()=>{};setMode('box')");
function click(x,y){const e={button:0,clientX:x,clientY:y,pointerId:1};element('overlay').onpointerdown(e);element('overlay').onpointerup(e)}
click(100,100);click(300,100);click(300,300);
assert.equal(run('draft.points.length'),3);
run("chooseSlot('B')");assert.equal(run('active'),'A');
const priorSpan=run('span');element('overlay').onwheel({preventDefault(){},deltaY:100});assert.equal(run('span'),priorSpan);
listeners.keydown({key:'Backspace',preventDefault(){},target:{closest:()=>false}});assert.equal(run('draft.points.length'),2);
click(300,300);element('overlay').ondblclick({preventDefault(){}});
assert.equal(run('state.slots.A.ops.length'),1);assert.equal(run('state.slots.A.ops[0].polygon.length'),3);assert.equal(run('draft'),null);
assert.equal(run('dirty'),true);assert.equal(run('state.slots.A.confirmed'),false);
run('closedAt=0');click(100,100);listeners.keydown({key:'Escape',preventDefault(){},target:{closest:()=>false}});assert.equal(run('draft'),null);
run('closedAt=0');click(100,100);click(300,100);click(300,300);click(100,100);assert.equal(run('draft'),null);assert.equal(run('state.slots.A.ops[0].polygon.length'),3);
console.log('PASS: click vertices, double-click/first-point closure, keyboard undo/cancel, camera/slot protection, selection state');

// Inject no_target to exercise rare detector failures; classification uses the live SSTv1 API.
const {chromium}=require('C:/Users/25041725G/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const out=path.resolve('outputs/photo_fallback_'+Date.now());fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1258,height:911}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  let predicts=0;page.on('request',r=>{if(r.url().endsWith('/api/predict'))predicts++;});
  await page.route('**/api/detect-trees',route=>route.fulfill({json:{action:'no_target',candidates:[],detector:'TEST no_target injection',threshold:.4,suppressed:0,projection:{kind:'photo'}}}));
  await page.goto('http://127.0.0.1:8043/?v=photo-fallback1');
  await page.waitForFunction(()=>state.ready);
  const photo=await page.request.get('http://127.0.0.1:8043/api/test-photo');assert.equal(photo.status(),200);
  await page.locator('#images').setInputFiles({name:'test-photo.jpg',mimeType:'image/jpeg',buffer:await photo.body()});
  await page.locator('#projection').selectOption('photo');
  const size=await page.evaluate(()=>[state.items[0].width,state.items[0].height]);
  const open=async()=>{await page.locator('#run').click();await page.locator('dialog.photo-fallback').waitFor({state:'visible'});};
  await open();assert.equal(predicts,0);await page.screenshot({path:path.join(out,'fallback.png')});
  await page.getByRole('button',{name:'Cancel / 取消',exact:true}).click();
  await page.waitForFunction(()=>!state.busy);assert.equal(predicts,0);assert.equal(await page.evaluate(()=>state.result),null);
  await open();await page.getByRole('button',{name:'Identify whole photo / 識別整張圖片',exact:true}).click();
  await page.waitForFunction(()=>!state.busy,{},{timeout:180000});
  const whole=await page.evaluate(()=>state.result);assert.ok(whole,await page.locator('#error').innerText());
  assert.equal(whole.target_selection,'user_confirmed_whole_photo');assert.equal(whole.image_model_version,'sstv1-20260914-v1');assert.deepEqual([whole.width,whole.height],size);assert.equal(predicts,1);
  await open();await page.getByRole('button',{name:'Select target manually / 手動選擇目標',exact:true}).click();
  const canvas=page.locator('.photo-fallback canvas');await canvas.waitFor({state:'visible'});
  assert.equal(await page.getByRole('button',{name:'Identify selected region / 識別所選區域',exact:true}).isDisabled(),true);
  const box=await canvas.boundingBox();await page.mouse.move(box.x+box.width*.15,box.y+box.height*.1);await page.mouse.down();await page.mouse.move(box.x+box.width*.85,box.y+box.height*.9,{steps:6});await page.mouse.up();
  assert.equal(await page.getByRole('button',{name:'Identify selected region / 識別所選區域',exact:true}).isDisabled(),false);
  const fields=page.locator('.photo-fallback-fields input');
  const region=[10,20,Math.min(200,size[0]-10),Math.min(300,size[1]-20)];
  for(let i=0;i<4;i++)await fields.nth(i).fill(String(region[i]));
  await page.setViewportSize({width:760,height:950});
  assert.equal(await page.evaluate(()=>document.querySelector('.photo-fallback').scrollWidth<=document.querySelector('.photo-fallback').clientWidth),true);
  await page.screenshot({path:path.join(out,'manual-mobile.png')});
  await page.getByRole('button',{name:'Identify selected region / 識別所選區域',exact:true}).click();
  await page.waitForFunction(()=>!state.busy,{},{timeout:180000});
  const manual=await page.evaluate(()=>state.result);assert.ok(manual,await page.locator('#error').innerText());
  assert.equal(manual.target_selection,'user_manual_crop');assert.equal(manual.image_model_version,'sstv1-20260914-v1');assert.deepEqual(manual.manual_crop_box,[10,20,10+region[2],20+region[3]]);assert.deepEqual([manual.width,manual.height],region.slice(2));assert.equal(predicts,2);
  // Selected pixels must come from the original resolution, not the displayed preview.
  assert.equal(await page.evaluate(async()=>{
    const image=new Image();image.src=state.items[0].url;await image.decode();const c=document.createElement('canvas');c.width=state.result.width;c.height=state.result.height;
    const [x,y]=state.result.manual_crop_box;c.getContext('2d').drawImage(image,x,y,c.width,c.height,0,0,c.width,c.height);
    return c.toDataURL('image/jpeg',.9)===state.result.crop_preview;
  }),true);
  await page.locator('#projection').selectOption('equirectangular');await page.locator('#run').click();await page.waitForFunction(()=>!state.busy);
  assert.match(await page.locator('#error').innerText(),/perspective/);assert.equal(await page.locator('.photo-fallback').count(),0);assert.equal(predicts,2);
  await page.locator('#projection').selectOption('photo');await open();await page.keyboard.press('Escape');await page.waitForFunction(()=>!state.busy);assert.equal(predicts,2);
  assert.deepEqual(errors,[]);
  const report={passed:true,detector_response_injected:true,real_sstv1_inference_calls:predicts,whole_dimensions:size,manual_dimensions:[manual.width,manual.height],cancel_no_inference:true,escape_no_inference:true,panorama_still_guarded:true,mobile_no_horizontal_overflow:true,page_errors:errors};
  fs.writeFileSync(path.join(out,'verification.json'),JSON.stringify(report,null,2));console.log(out,JSON.stringify(report));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});

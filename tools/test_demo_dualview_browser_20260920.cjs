// Isolated headless regression for our local application, not the user's browser.
const {chromium}=require('C:/Users/25041725G/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const out=path.resolve('outputs/SSTv1_Dualview_Deployment_20260920_v1/browser_acceptance');
 fs.mkdirSync(out,{recursive:false});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:950}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8043/image-only');
  await page.waitForFunction(()=>state.ready,{},{timeout:60000});
  assert.equal(await page.locator('input[name=source]').count(),0);
  assert.match(await page.locator('#modelVersion').innerText(),/SSTv1-H120-DV/);
  await page.locator('#images').setInputFiles('C:/Users/25041725G/Desktop/Aquilaria sinensis/large.jpg');
  await page.locator('#projection').selectOption('photo');await page.locator('#organ').selectOption('leaf');
  await page.locator('#run').click();
  await page.waitForFunction(()=>!state.busy&&state.result,{},{timeout:180000});
  const result=await page.evaluate(()=>state.result);
  assert.equal(result.view_evidence.encoder_passes,2);
  assert.equal(result.inference_pipeline,'sstv1-confirmed-target-whole-center-mean-v2');
  assert.equal(result.top3[0].species,'Aquilaria sinensis');
  await page.locator('.evidence summary').click();
  await page.locator('#modelInputDetail').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('#modelInputDetail').isVisible(),true);
  await page.screenshot({path:path.join(out,'desktop-evidence.png')});
  const event=page.waitForEvent('download');await page.locator('#export').click();
  const downloaded=await event;await downloaded.saveAs(path.join(out,'result.json'));
  const exported=JSON.parse(fs.readFileSync(path.join(out,'result.json'),'utf8'));
  assert.equal(exported.view_evidence.independent_images,1);
  assert.equal(exported.model_input_detail,undefined);
  await page.setViewportSize({width:390,height:844});
  await page.locator('#modelInputDetail').scrollIntoViewIfNeeded();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
  await page.screenshot({path:path.join(out,'mobile-evidence.png')});
  await page.setViewportSize({width:1280,height:950});
  // Native polygon cropping must reach the same new classification path.
  await page.locator('#polygonCrop').click();
  await page.waitForFunction(()=>document.getElementById('polygonStatus')?.textContent.includes('vertices'));
  for(const [x,y] of [[.05,.05],[.9,.05],[.9,.9],[.05,.9]]){
   const b=await page.locator('#polygonCanvas').boundingBox();await page.mouse.click(b.x+b.width*x,b.y+b.height*y);
  }
  await page.locator('#polygonClose').click();await page.locator('#polygonApply').click();
  await page.waitForFunction(()=>!state.busy);
  await page.locator('#run').click();await page.waitForFunction(()=>!state.busy&&state.result,{},{timeout:180000});
  const cropped=await page.evaluate(()=>state.result);
  assert.equal(cropped.target_selection,'user_polygon_crop');assert.equal(cropped.view_evidence.encoder_passes,2);
  assert.equal(cropped.manual_polygon.vertices.length,4);
  await page.goto('http://127.0.0.1:8043/multimodal');
  await page.waitForFunction(()=>state.ready);
  assert.equal(await page.locator('input[name=source]').count(),2);
  await page.locator('input[name=source][value=image_point]').check();
  assert.equal(await page.locator('#cloudPanel').isVisible(),true);
  assert.deepEqual(errors,[]);
  const report={passed:true,headless_local_app_test:true,both_editions:true,real_image_uploaded:true,
   dual_model_previews:true,export_provenance:true,manual_polygon_real_inference:true,mobile_no_overflow:true,
   cloud_input_retained:true,page_errors:errors};
  fs.writeFileSync(path.join(out,'verification.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});

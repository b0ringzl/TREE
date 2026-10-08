const {chromium}=require('C:/Users/25041725G/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const out=path.resolve('outputs/polygon_crop_'+Date.now());fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:950}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  let detect=0,predict=0;page.on('request',r=>{if(r.url().endsWith('/api/detect-trees'))detect++;if(r.url().endsWith('/api/predict'))predict++;});
  await page.goto('http://127.0.0.1:8043/?v=polygon1');await page.waitForFunction(()=>state.ready&&!!document.getElementById('polygonCrop'));
  assert.equal(await page.locator('#polygonCrop').isVisible(),false);
  const photo=await page.request.get('http://127.0.0.1:8043/api/test-photo');assert.equal(photo.status(),200);
  await page.locator('#images').setInputFiles({name:'tree.jpg',mimeType:'image/jpeg',buffer:await photo.body()});
  await page.locator('#projection').selectOption('photo');
  const original=await page.evaluate(()=>({url:state.items[0].url,w:state.items[0].width,h:state.items[0].height}));
  async function clickPoint(x,y){const b=await page.locator('#polygonCanvas').boundingBox();await page.mouse.click(b.x+b.width*x,b.y+b.height*y);}
  async function open(){await page.locator('#polygonCrop').click();await page.waitForFunction(()=>document.getElementById('polygonStatus')?.textContent.includes('vertices'));}
  await open();await clickPoint(.1,.1);await clickPoint(.9,.9);await clickPoint(.1,.9);await clickPoint(.9,.1);await page.locator('#polygonClose').click();
  assert.equal(await page.locator('#polygonApply').isDisabled(),true);assert.match(await page.locator('#polygonStatus').innerText(),/Invalid/);
  await page.locator('#polygonReset').click();await clickPoint(.5,.05);await clickPoint(.9,.85);await clickPoint(.15,.9);
  await page.keyboard.press('Backspace');assert.match(await page.locator('#polygonStatus').innerText(),/^2 /);
  await clickPoint(.15,.9);await page.locator('#polygonClose').click();await page.waitForFunction(()=>!document.getElementById('polygonApply').disabled);
  const dragBox=await page.locator('#polygonCanvas').boundingBox();await page.mouse.move(dragBox.x+dragBox.width*.5,dragBox.y+dragBox.height*.05);await page.mouse.down();await page.mouse.move(dragBox.x+dragBox.width*.55,dragBox.y+dragBox.height*.08,{steps:4});await page.mouse.up();await page.waitForFunction(()=>!document.getElementById('polygonApply').disabled);
  await page.locator('#polygonZoom').fill('1.5');await page.locator('#polygonZoom').dispatchEvent('input');await page.locator('#polygonZoom').fill('1');await page.locator('#polygonZoom').dispatchEvent('input');await page.waitForFunction(()=>!document.getElementById('polygonApply').disabled);
  await page.screenshot({path:path.join(out,'polygon-editor.png')});
  await page.locator('#polygonApply').click();await page.waitForFunction(()=>!state.busy);
  const metadata=await page.evaluate(()=>state.items[0].manualPolygon);assert.equal(metadata.vertices.length,3);
  assert.deepEqual(metadata.source_size,[original.w,original.h]);
  // Check actual decoded crop pixels, not only a thumbnail or CSS mask.
  const pixels=await page.evaluate(async()=>{const item=state.items[0],im=new Image();im.src=item.url;await im.decode();const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d');ctx.drawImage(im,0,0);const corner=Array.from(ctx.getImageData(0,0,1,1).data);const src=new Image();src.src=item.polygonOriginal.url;await src.decode();const [x,y]=item.manualPolygon.crop_box;const cx=Math.floor(im.width/2),cy=Math.floor(im.height/2);const inside=Array.from(ctx.getImageData(cx,cy,1,1).data);ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(src,-x,-y);return {corner,inside,original: Array.from(ctx.getImageData(cx,cy,1,1).data),w:im.width,h:im.height};});
  assert.deepEqual(pixels.corner,[128,128,128,255]);assert.ok(pixels.inside.every((v,i)=>Math.abs(v-pixels.original[i])<=1),'PNG colour conversion tolerance: one level');
  assert.equal(pixels.w,metadata.crop_box[2]-metadata.crop_box[0]);
  await page.locator('#run').click();await page.waitForFunction(()=>!state.busy,{},{timeout:180000});
  const result=await page.evaluate(()=>state.result);assert.ok(result,await page.locator('#error').innerText());assert.equal(result.target_selection,'user_polygon_crop');assert.deepEqual(result.manual_polygon,metadata);assert.equal(detect,0);assert.equal(predict,1);assert.equal(result.image_model_version,'sstv1-aquilaria-user-20260915-v1');
  const downloadEvent=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadEvent;await download.saveAs(path.join(out,'export.json'));assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out,'export.json'))).manual_polygon,metadata);
  await open();await page.locator('#polygonCancel').click();assert.deepEqual(await page.evaluate(()=>state.items[0].manualPolygon),metadata);
  await page.locator('#restorePolygon').click();assert.equal(await page.evaluate(()=>state.items[0].url),original.url);assert.equal(await page.evaluate(()=>state.result),null);
  // Multi-image state must be separate; selecting a crop invalidates prior cloud pairing.
  await page.locator('#images').setInputFiles({name:'second.jpg',mimeType:'image/jpeg',buffer:await photo.body()});assert.equal(await page.locator('#restorePolygon').isVisible(),false);
  await open();await clickPoint(.1,.1);await clickPoint(.85,.15);await clickPoint(.8,.9);await clickPoint(.1,.9);await clickPoint(.1,.1);await page.waitForFunction(()=>!document.getElementById('polygonApply').disabled);
  await page.locator('#polygonApply').click();await page.waitForFunction(()=>!state.busy);
  await page.locator('#clear').click();await page.locator('#undoButton').click();assert.equal(await page.evaluate(()=>state.items.length),2);
  await page.locator('.thumb button').nth(2).click();assert.equal(await page.locator('#restorePolygon').isVisible(),true);
  await page.setViewportSize({width:390,height:844});await open();await page.screenshot({path:path.join(out,'polygon-mobile.png')});
  assert.equal(await page.evaluate(()=>{const d=document.getElementById('polygonEditor');return d.scrollWidth<=d.clientWidth+1;}),true);
  await page.keyboard.press('Escape');await page.waitForFunction(()=>!state.busy);
  await page.setViewportSize({width:1280,height:950});
  await page.locator('input[name=source][value=image_point]').check();
  await page.evaluate(()=>{$('paired').checked=true;update();});assert.equal(await page.locator('#paired').isChecked(),true);
  await open();await page.waitForFunction(()=>!document.getElementById('polygonApply').disabled);await page.locator('#polygonApply').click();await page.waitForFunction(()=>!state.busy);assert.equal(await page.locator('#paired').isChecked(),false);
  await page.locator('#restorePolygon').click();assert.equal(await page.evaluate(()=>state.items[1].polygonOriginal),undefined);
  await page.locator('#projection').selectOption('equirectangular');await open();assert.match(await page.locator('#polygonEditor [role=alert]').innerText(),/does not remove spherical distortion/);await page.keyboard.press('Escape');
  assert.deepEqual(errors,[]);
  const report={passed:true,live_inference:predict,detector_calls:detect,native_mask_pixels_checked:true,export_metadata:true,cancel_restore_undo:true,multiple_images:true,mobile_no_horizontal_overflow:true,cloud_pair_confirmation_reset:true,panorama_warning:true,errors};
  fs.writeFileSync(path.join(out,'verification.json'),JSON.stringify(report,null,2));console.log(out,report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});

const {chromium}=require('C:/Users/25041725G/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const assert=require('assert');const path=require('path');
const out=path.resolve('outputs/hk_point_assist_20260911_v1');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 try{
 const page=await browser.newPage({viewport:{width:1400,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8043/point-assist');await page.locator('#evidence canvas').waitFor();
 assert.equal(await page.locator('#count').innerText(),'95');
 await page.screenshot({path:path.join(out,'audit-page.png'),fullPage:true});
 await page.locator('#species').selectOption('Wodyetia bifurcata');await page.waitForFunction(()=>document.querySelector('#sample').options.length===5);
 await page.locator('#evidence canvas').waitFor();assert.equal(await page.locator('#review input:checked').count(),0);
 await page.locator('#next').click();await page.waitForFunction(()=>document.querySelector('#position').textContent==='2 / 5');
 const catalog=await (await page.request.get('http://127.0.0.1:8043/api/point-assist/catalog')).json();
 // Use a fixed first catalog record, independent of model correctness.
 const row=catalog.items[0];await page.goto('http://127.0.0.1:8043/?pair='+row.key);
 await page.waitForFunction(()=>document.querySelector('#cloud').files.length===1&&document.querySelector('#mainImage').complete);
 await page.waitForFunction(()=>document.querySelector('#health').textContent.includes('ready'));
 assert.equal(await page.locator('#run').isDisabled(),true);
 await page.locator('#paired').check();assert.equal(await page.locator('#run').isEnabled(),true);
 const pending=page.waitForResponse(r=>r.url().endsWith('/api/predict-with-cloud'),{timeout:180000});
 await page.locator('#run').click();const response=await pending;assert.equal(response.status(),200);const d=await response.json();
 await page.locator('#pointEvidence canvas').waitFor();assert.equal(d.fusion_used,false);assert.equal(d.point_evidence.total_tree_height_m,null);
 const image=await (await page.request.get('http://127.0.0.1:8043'+row.image)).body();
 const base=await (await page.request.post('http://127.0.0.1:8043/api/predict',{multipart:{image:{name:'single-tree.jpg',mimeType:'image/jpeg',buffer:image},projection:'photo',scenario:'street'}})).json();
 assert.deepEqual(d.top3.map(r=>[r.species,r.score]),base.top3.map(r=>[r.species,r.score]));
 await page.screenshot({path:path.join(out,'demo-point-result.png'),fullPage:true});
 assert.deepEqual(errors,[]);
 const result={catalog_count:95,foxtail_views:5,next_navigation:true,no_fake_review_saved:true,real_inference_status:response.status(),
   image_scores_exactly_unchanged:true,sample_key:row.key,point_count:d.point_evidence.point_count,
   metric_span:d.point_evidence.observed_vertical_span_m,whole_height_not_claimed:true,page_errors:errors,
   top3:d.top3.map(r=>({name:r.name,score:r.score})),screenshots:['audit-page.png','demo-point-result.png']};
 fs.writeFileSync(path.join(out,'browser_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

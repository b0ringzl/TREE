const {chromium}=require('C:/Users/25041725G/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),assert=require('assert'),path=require('path');
const out=path.resolve('hk-tree-vmms/derived/banyan_coverage_display_v3_20260912');
const reviews=path.resolve('outputs/hk_point_assist_20260911_v1/geometry_reviews.json');
const beforeReview=fs.existsSync(reviews)?fs.readFileSync(reviews):null;
(async()=>{const browser=await chromium.launch({channel:'chrome',headless:true});try{
 const page=await browser.newPage({viewport:{width:1400,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8043/point-assist?species=Ficus%20microcarpa&v=coverage3');
 await page.waitForFunction(()=>document.querySelector('#coverageStatus').textContent.includes('194,106'));
 await page.locator('#evidence canvas').waitFor();await page.waitForFunction(()=>document.querySelector('#projection').naturalWidth===1920);
 assert.equal(await page.locator('#sample option').count(),16);
 const catalog=await (await page.request.get('http://127.0.0.1:8043/api/point-assist/catalog')).json();
 const selectedKey=await page.locator('#sample').inputValue(),selected=catalog.items.find(r=>r.key===selectedKey);
 const expectedChecks=['same_tree','isolated_tree','base_visible','crown_complete','species_verified'].filter(k=>selected.geometry_review?.[k]).length;
 assert.equal(await page.locator('#review input:checked').count(),expectedChecks);
 assert.equal(await page.locator('#coverageControls').isVisible(),true);assert.match(await page.locator('#projection').getAttribute('src'),/coverage.*\/all$/);
 await page.screenshot({path:path.join(out,'demo-whole-cloud.png'),fullPage:true});
 for(const mode of ['upper','lower','archive','all']){await page.locator('#coverageMode').selectOption(mode);assert.match(await page.locator('#projection').getAttribute('src'),mode==='archive'?/asset.*\/projection$/:new RegExp('/'+mode+'$'));}
 await page.locator('#coverageViews summary').click();await page.waitForFunction(()=>document.querySelector('#orthoViews').naturalWidth===1500);
 await page.locator('#next').click();await page.waitForFunction(()=>document.querySelector('#position').textContent==='2 / 16'&&document.querySelector('#coverageStatus').textContent.includes('174,642'));
 await page.locator('#species').selectOption('Wodyetia bifurcata');await page.waitForFunction(()=>document.querySelector('#sample').options.length===5);await page.locator('#evidence canvas').waitFor();
 assert.equal(await page.locator('#coverageControls').isVisible(),false);assert.equal(await page.locator('#coverageViews').isVisible(),false);assert.match(await page.locator('#projection').getAttribute('src'),/asset.*\/projection$/);
 await page.locator('#species').selectOption('Ficus microcarpa');await page.waitForFunction(()=>document.querySelector('#coverageStatus').textContent.includes('194,106'));
 await page.setViewportSize({width:760,height:1000});await page.screenshot({path:path.join(out,'demo-mobile.png'),fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);assert.deepEqual(errors,[]);
 const afterReview=fs.existsSync(reviews)?fs.readFileSync(reviews):null;assert.deepEqual(afterReview,beforeReview);
 const result={passed:true,banyan_views:16,whole_cloud_projection_1920px:true,four_projection_modes:true,orthographic_views:true,next_navigation:true,other_species_original_projection:true,source_reviews_not_written:true,mobile_no_horizontal_overflow:true,page_errors:errors};
 fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1);});

const { chromium } = require('C:/Users/25041725G/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const origin = 'http://127.0.0.1:' + (process.env.SPACESCAN_TEST_PORT || '8055');

(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    const pageErrors = [];
    let detectorCalls = 0;
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('request', request => {if (request.url().includes('/api/detect-trees')) detectorCalls++;});
    await page.goto(origin + '/image-only?sam=1');
    await page.waitForFunction(() => !!document.getElementById('samSelect'));
    const photo = await page.request.get(origin + '/api/test-photo');
    assert.equal(photo.status(), 200);
    await page.locator('#images').setInputFiles({name: 'tree.jpg', mimeType: 'image/jpeg', buffer: await photo.body()});
    await page.locator('#projection').selectOption('photo');
    await page.waitForFunction(() => !document.getElementById('samSelect').disabled);
    const original = await page.evaluate(() => ({width: state.items[0].width, height: state.items[0].height}));

    let called = false;
    await page.route('**/api/sam/segment', async route => {
      called = true;
      await route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({
        polygon: [[.18, .12], [.79, .12], [.85, .8], [.14, .83]],
        model: 'SAM 2.1 Tiny', proposal_only: true, user_confirmation_required: true,
      })});
    });
    await page.locator('#samSelect').click();
    const box = await page.locator('#polygonCanvas').boundingBox();
    await page.mouse.click(box.x + box.width * .5, box.y + box.height * .5);
    await page.waitForFunction(() => document.getElementById('polygonApply') && !document.getElementById('polygonApply').disabled);
    assert.equal(called, true);
    assert.match(await page.locator('.sam-status').innerText(), /SAM proposal/);
    await page.locator('#polygonApply').click();
    await page.waitForFunction(() => !state.busy);
    const result = await page.evaluate(() => ({polygon: state.items[0].manualPolygon, name: state.items[0].file.name, width: state.items[0].width, height: state.items[0].height}));
    assert.equal(result.polygon.assist_model, 'SAM 2.1 Tiny');
    assert.equal(result.polygon.proposal_user_confirmed, true);
    assert.deepEqual(result.polygon.source_size, [original.width, original.height]);
    assert.match(result.name, /-polygon\.png$/);
    assert.ok(result.width > 16 && result.height > 16);
    await page.locator('#run').click();
    await page.waitForFunction(() => !state.busy && !!state.result, {}, {timeout: 180000});
    assert.equal(await page.evaluate(() => state.result.target_selection), 'sam_assisted_user_confirmed');
    assert.equal(detectorCalls, 0);
    await page.locator('#restorePolygon').click();
    assert.equal(await page.evaluate(() => state.items[0].manualPolygon), undefined);
    await page.locator('#projection').selectOption('equirectangular');
    assert.equal(await page.locator('#samSelect').isDisabled(), true);
    assert.equal(await page.locator('#polygonCrop').isDisabled(), false);
    assert.deepEqual(pageErrors, []);
    const dualPage = await browser.newPage();
    await dualPage.goto(origin + '/?sam=1');
    await dualPage.locator('#images').setInputFiles({name: 'tree.jpg', mimeType: 'image/jpeg', buffer: await photo.body()});
    await dualPage.waitForFunction(() => !document.getElementById('samSelect').disabled);
    assert.equal(await dualPage.locator('#samSelect').isVisible(), true);
    console.log(JSON.stringify({passed: true, smart_select_called: called, native_crop: true, identification: true, detector_bypassed: true, restored: true, panorama_guard: true, both_editions: true}));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});

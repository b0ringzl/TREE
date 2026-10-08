const assert=require('node:assert/strict');
const {readApiResponse}=require('./static/api-response.js');
(async()=>{
  assert.deepEqual(await readApiResponse(new Response('{"mode":"adapted"}')),{mode:'adapted'});
  for(const body of ['Internal Server Error','<html>Failure</html>','']){
    await assert.rejects(readApiResponse(new Response(body,{status:500})),/Local service error \(500\)/);
  }
  await assert.rejects(readApiResponse(new Response('{"detail":"Try again / 請重試"}',{status:503})),/Try again/);
  await assert.rejects(readApiResponse(new Response('null')),/Unexpected service response/);
  console.log('API RESPONSE TESTS PASSED');
})().catch(e=>{console.error(e);process.exit(1);});

'use strict';
async function readApiResponse(response){
  const text=await response.text();let data;
  const fallback=response.status>=500
    ?'Local service error ('+response.status+'). Please retry. / 本地服務錯誤（'+response.status+'），請重試。'
    :'Unexpected service response ('+response.status+'). Please retry. / 服務回應格式異常（'+response.status+'），請重試。';
  try{data=JSON.parse(text);}catch{throw new Error(fallback);}
  if(!response.ok)throw new Error(typeof data?.detail==='string'?data.detail:fallback);
  if(!data||typeof data!=='object')throw new Error(fallback);
  return data;
}
if(typeof module!=='undefined')module.exports={readApiResponse};

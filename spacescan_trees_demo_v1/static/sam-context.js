'use strict';
// Geometry only: keep the polygon for provenance, but classify an unmasked
// native-pixel rectangle with a small amount of surrounding context.
function samContextBox(points,width,height,paddingFraction=.12){
  if(!Array.isArray(points)||points.length<3||!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||!Number.isFinite(paddingFraction)||paddingFraction<0||paddingFraction>1)throw Error('Invalid SAM crop geometry');
  const xs=points.map(p=>p?.[0]),ys=points.map(p=>p?.[1]);
  if([...xs,...ys].some(v=>!Number.isFinite(v)))throw Error('Invalid SAM polygon coordinate');
  const selection=[Math.max(0,Math.floor(Math.min(...xs))),Math.max(0,Math.floor(Math.min(...ys))),Math.min(width,Math.ceil(Math.max(...xs))),Math.min(height,Math.ceil(Math.max(...ys)))];
  const spanX=selection[2]-selection[0],spanY=selection[3]-selection[1];
  if(spanX<16||spanY<16)throw Error('SAM selection is too small');
  const marginX=paddingFraction===0?0:Math.max(16,Math.ceil(spanX*paddingFraction));
  const marginY=paddingFraction===0?0:Math.max(16,Math.ceil(spanY*paddingFraction));
  const crop=[Math.max(0,selection[0]-marginX),Math.max(0,selection[1]-marginY),Math.min(width,selection[2]+marginX),Math.min(height,selection[3]+marginY)];
  return {selection_box:selection,crop_box:crop,padding_fraction:paddingFraction};
}
if(typeof module!=='undefined'&&module.exports)module.exports={samContextBox};

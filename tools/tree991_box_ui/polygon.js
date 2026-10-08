'use strict';
// Kept DOM-free so preview/export geometry can be regression-tested together.
function containsPolygon(a,b,p){let inside=false;for(let i=0,j=p.length-1;i<p.length;j=i++){
 const [x1,y1]=p[j],[x2,y2]=p[i],dx=x2-x1,dy=y2-y1;
 if(Math.abs((a-x1)*dy-(b-y1)*dx)<=1e-8*Math.max(1,Math.hypot(dx,dy))&&a>=Math.min(x1,x2)-1e-8&&a<=Math.max(x1,x2)+1e-8&&b>=Math.min(y1,y2)-1e-8&&b<=Math.max(y1,y2)+1e-8)return true;
 if((y1>b)!==(y2>b)&&a<dx*(b-y1)/dy+x1)inside=!inside;
 }return inside}
if(typeof module!=='undefined')module.exports={containsPolygon};

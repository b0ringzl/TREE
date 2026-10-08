'use strict';
const assert=require('node:assert/strict');
const test=require('node:test');
const {samContextBox}=require('./sam-context.js');

test('adds context on all four sides without changing the polygon box',()=>{
  const result=samContextBox([[100,200],[300,200],[300,600],[100,600]],1000,1000);
  assert.deepEqual(result.selection_box,[100,200,300,600]);
  assert.deepEqual(result.crop_box,[76,152,324,648]);
  assert.equal(result.padding_fraction,.12);
});

test('clamps context at image edges and rejects tiny or invalid outlines',()=>{
  assert.deepEqual(samContextBox([[0,0],[100,0],[100,100],[0,100]],120,120).crop_box,[0,0,116,116]);
  assert.throws(()=>samContextBox([[0,0],[4,0],[4,4]],100,100));
  assert.throws(()=>samContextBox([[0,0],[20,0],[NaN,20]],100,100));
});

test('zero padding yields a tight, unexpanded rectangle',()=>{
  const result=samContextBox([[100,200],[300,200],[300,600],[100,600]],1000,1000,0);
  assert.deepEqual(result.crop_box,[100,200,300,600]);
  assert.equal(result.padding_fraction,0);
});

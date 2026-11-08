import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {visibleObservations,TrackVideo} from '../.test-build/TrackVideo.js';
const tracks=[{id:'camera_a:1',observations:[{seconds:1,frame_index:30,bbox:[1,2,30,40],confidence:.9},{seconds:2,frame_index:60,bbox:[4,5,34,45],confidence:.8}]}];
test('overlay shows actual nearest frame box within tolerance',()=>{
 const rows=visibleObservations(tracks,1.01,30);assert.equal(rows[0].trackId,'camera_a:1');assert.deepEqual(rows[0].observation.bbox,[1,2,30,40]);
});
test('overlay does not hallucinate tracks across gaps or before/after track',()=>{
 for(const time of [0,1.5,3])assert.deepEqual(visibleObservations(tracks,time,30),[]);
});
test('overlay handles absent tracks and invalid FPS',()=>{
 assert.deepEqual(visibleObservations([],1,30),[]);assert.equal(visibleObservations(tracks,1,NaN).length,1);
});
test('source player retains independent native playback controls',()=>{
 const html=renderToStaticMarkup(createElement(TrackVideo,{src:'blob:fixture',label:'Camera A',tracks,fps:30}));
 assert.match(html,/controls/);assert.match(html,/Camera A/);assert.match(html,/blank frames may contain untracked/);
});

test('shared seeking preserves independent camera selection and missing-player behavior',async()=>{
 const {seekVideo}=await import('../.test-build/TrackVideo.js');
 const original=globalThis.document;
 const players=[{currentTime:0},{currentTime:0}];
 try{
  globalThis.document={querySelectorAll:selector=>{assert.equal(selector,'video');return players;}};
  seekVideo('camera_a',1.5);assert.equal(players[0].currentTime,1.5);assert.equal(players[1].currentTime,0);
  seekVideo('camera_b',3);assert.equal(players[1].currentTime,3);assert.equal(players[0].currentTime,1.5);
  globalThis.document={querySelectorAll:()=>[]};assert.doesNotThrow(()=>seekVideo('camera_a',2));
 }finally{if(original===undefined)delete globalThis.document;else globalThis.document=original;}
});

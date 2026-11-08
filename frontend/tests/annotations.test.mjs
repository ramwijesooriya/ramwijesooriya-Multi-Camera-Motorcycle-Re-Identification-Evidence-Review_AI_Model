import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {Annotations,makeAnnotationTemplate} from '../.test-build/Annotations.js';
const tracks=[{id:'camera_a:1',camera_id:'camera_a',summary:{start_seconds:0,end_seconds:1}},{id:'camera_b:1',camera_id:'camera_b',summary:{start_seconds:2,end_seconds:3}}];
test('manual labels start unknown and do not infer identities or clock timing',()=>{
 const labels=makeAnnotationTemplate('fixture',tracks,{camera_a:{sha256:'a'},camera_b:{sha256:'b'}});
 assert.equal(labels.footage_relationship,'unconfirmed');assert.equal(labels.annotation_basis,'');
 assert.ok(labels.tracks.every(t=>t.label_status==='unknown'&&t.identity_id===''&&t.query_status==='unknown'));
 assert.deepEqual(labels.source_input_hashes,{camera_a:'a',camera_b:'b'});
});
test('annotation UI makes source evidence and separation from predictions explicit',()=>{
 const html=renderToStaticMarkup(createElement(Annotations,{job:'fixture',tracks,onSeek:()=>{}}));
 assert.match(html,/never change candidate scores or force acceptance/);
 assert.match(html,/Download ground-truth labels JSON/);assert.match(html,/Confirmed no counterpart/);assert.match(html,/fragment group/);
});

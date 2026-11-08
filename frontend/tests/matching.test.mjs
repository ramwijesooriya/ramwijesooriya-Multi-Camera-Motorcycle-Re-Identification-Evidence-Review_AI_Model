// Rendering contract fixtures only; these are not application inference results.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {Matching, TimingControls, defaultTiming} from '../.test-build/Matching.js';

function render(matching){return renderToStaticMarkup(createElement(Matching,{matching,tracks:[],job:'unit-test',onSeek:()=>{}}));}
const fixture=outcome=>({status:'completed',queries:[{query_track_id:'camera_a:1',result:outcome==='ACCEPTED'?'MATCH_ACCEPTED':'NO_RELIABLE_MATCH'}],candidates:[{query_track_id:'camera_a:1',candidate_track_id:'camera_b:2',rank:1,similarity:.92,competition:{margin:.001,reverse_margin:.002,ambiguous:true,mutual_best:true,competing_candidates:[],competing_queries:[]},verification:{outcome,independent_visual_score:.91,evidence:[{kind:'missing',code:'temporal',message:'Clock offset is unknown.',value:null,blocking:true}],uncertainty:{blocking_reasons:['temporal']}}}]});

test('top-ranked uncertain candidate shows no reliable match and missing evidence',()=>{
 const html=render(fixture('UNCERTAIN'));
 assert.match(html,/NO RELIABLE MATCH/);
 assert.match(html,/UNCERTAIN/);
 assert.match(html,/Clock offset is unknown/);
 assert.match(html,/blocks acceptance/);
 assert.match(html,/identity is ambiguous/);
 assert.doesNotMatch(html,/badge accepted/);
});
test('all review statuses render as supplied by the verifier',()=>{
 for(const status of ['ACCEPTED','UNCERTAIN','REJECTED'])assert.match(render(fixture(status)),new RegExp(`badge ${status.toLowerCase()}`));
});
test('identical input warning disclaims external camera evaluation',()=>{
 const matching=fixture('UNCERTAIN');matching.evaluation={identical_inputs:true,note:'No accuracy evaluation performed.'};
 const html=render(matching);
 assert.match(html,/Identical input videos detected/);
 assert.match(html,/not external-camera evaluation/);
});
test('matching failure preserves explicit extraction fallback',()=>{
 const html=render({status:'failed',error:'Matching unavailable; extraction is preserved.',candidates:[],queries:[]});
 assert.match(html,/Matching unavailable; extraction is preserved/);
 assert.doesNotMatch(html,/Track cosine/);
});
test('timing controls preserve unknown offset instead of implying zero',()=>{
 const html=renderToStaticMarkup(createElement(TimingControls,{value:defaultTiming,setValue:()=>{},disabled:true}));
 assert.match(html,/blank = unknown/);
 assert.match(html,/value=""/);
 assert.match(html,/disabled/);
});

test('a negative reciprocal margin is explained as outscored rather than a near tie',()=>{
 const matching=fixture('UNCERTAIN');Object.assign(matching.candidates[0].competition,{outcompeted:true,near_tie:false,reverse_best_query_track_id:'camera_a:3',reverse_best_similarity:.9304});
 const html=render(matching);assert.match(html,/outscored by another track/);assert.match(html,/camera_a:3/);assert.match(html,/does not prove different physical identity/);
});

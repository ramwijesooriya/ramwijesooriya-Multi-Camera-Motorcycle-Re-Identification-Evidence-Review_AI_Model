"""Labeled SYNTHETIC unit fixtures. Numbers here are not real-video accuracy."""
import copy
import csv
import io
import math
import pytest
from app.evaluation import annotation_template, evaluate, write_new
from app.diagnostics import inspect_analysis, fragmentation_hypotheses
from app.exports import candidate_csv
from app.matching import rank_candidates
from app.config import MatchConfig


def track(camera, number, vector, start=0):
    norm = math.sqrt(sum(v*v for v in vector))
    vector = [v/norm for v in vector]
    observations = [{'frame_index': i, 'seconds': start+i/2, 'bbox': [i,0,i+40,50], 'confidence': .9, 'timestamp_source': 'decoder'} for i in range(3)]
    return {'id': f'{camera}:{number}', 'camera_id': camera, 'local_track_id': number,
            'summary': {'start_seconds':start,'end_seconds':start+1,'observation_count':3,'mean_detection_confidence':.9,'displacement_pixels':[30,0]},
            'observations': observations, 'appearance': {'status':'ready','vector':vector,'within_track_similarity':1.,
            'samples':[dict(o,embedding=vector,usable=True) for o in observations]}}


def fixture():
    a = [track('camera_a',1,[1,0]), track('camera_a',2,[0,1]), track('camera_a',3,[0,1]), track('camera_a',4,[1,1])]
    b = [track('camera_b',1,[1,0]),track('camera_b',2,[0,1])]
    def pair(source,target,outcome):
        return {'query_track_id':source,'candidate_track_id':target,'rank':1,'similarity':.5,
                'verification':{'outcome':outcome,'independent_visual_score':.5,'supporting_evidence':[],
                                'contradictory_evidence':[],'missing_evidence':[], 'uncertainty':{'calibrated_probability':None}}}
    result = {'id':'synthetic-evaluator-unit-test','cameras':[{'camera_id':'camera_a','frames_processed':20,'tracks':a},
              {'camera_id':'camera_b','frames_processed':20,'tracks':b}], 'elapsed_seconds':10.,
              'inputs':{'camera_a':{'sha256':'SYNTHETIC-A'},'camera_b':{'sha256':'SYNTHETIC-B'}},
              'matching':{'status':'completed','evaluation':{'identical_inputs':False},'candidates':[
                  pair('camera_a:1','camera_b:1','ACCEPTED'), pair('camera_a:2','camera_b:1','ACCEPTED'),
                  pair('camera_a:3','camera_b:2','UNCERTAIN')],
                  'queries':[{'query_track_id':t['id'],'result':'MATCH_ACCEPTED' if i<2 else 'NO_RELIABLE_MATCH'} for i,t in enumerate(a)]}}
    labels = annotation_template(result)
    labels['footage_relationship'] = 'independent_cameras'
    labels['annotation_basis'] = 'DETERMINISTIC SYNTHETIC UNIT FIXTURE ONLY, not real camera labels'
    identities = ['fixture-X','fixture-Y','fixture-Z','fixture-W','fixture-X','fixture-Z']
    for label, identity in zip(labels['tracks'],identities):
        label.update(label_status='confirmed',identity_id=identity)
        if label['camera_id']=='camera_a':
            label['query_status']='no_counterpart' if identity=='fixture-Y' else 'has_counterpart'
    return result,labels


def test_explicit_synthetic_metrics_match_hand_calculation():
    result,labels=fixture()
    output=evaluate(result,labels)
    metrics=output['metrics']
    assert metrics['rank_1_accuracy']==pytest.approx(2/3)
    assert metrics['recall_at_k']['5']==pytest.approx(2/3)
    assert metrics['mean_average_precision']==pytest.approx(2/3)
    assert metrics['accepted_match_precision']==.5
    assert metrics['accepted_match_recall']==pytest.approx(1/3)
    assert metrics['false_acceptance_rate']==1.
    assert metrics['abstention_rate']==.5
    assert output['processing']['processing_fps']==4.
    assert output['counts']['evaluated_queries']==4
    assert any('counterpart expected' in w for w in output['warnings'])


def test_retrieval_uses_actual_full_vectors_not_top_k_export_scores():
    result,labels=fixture()
    # Deliberately wrong fixture export scores must not influence recomputed retrieval.
    for row in result['matching']['candidates']:
        row['similarity']=-.99
    result['matching']['candidates']=result['matching']['candidates'][:1]
    result['matching']['queries'][1]['result']='NO_RELIABLE_MATCH'
    assert evaluate(result,labels)['metrics']['rank_1_accuracy']==pytest.approx(2/3)


def test_zero_acceptances_precision_is_undefined_not_perfect():
    result,labels=fixture()
    for row in result['matching']['candidates']:
        row['verification']['outcome']='UNCERTAIN'
    for query in result['matching']['queries']:query['result']='NO_RELIABLE_MATCH'
    metrics=evaluate(result,labels)['metrics']
    assert metrics['accepted_match_precision'] is None
    assert metrics['accepted_match_recall']==0
    assert metrics['abstention_rate']==1


@pytest.mark.parametrize('relationship',['unconfirmed','same_scene_duplicate'])
def test_duplicate_or_unconfirmed_camera_claim_refused(relationship):
    result,labels=fixture();labels['footage_relationship']=relationship
    with pytest.raises(ValueError,match='independent cameras'):
        evaluate(result,labels)


def test_identical_hashes_refused_even_if_user_claims_independent():
    result,labels=fixture();result['inputs']['camera_b']['sha256']='SYNTHETIC-A';labels['source_input_hashes']['camera_b']='SYNTHETIC-A'
    with pytest.raises(ValueError,match='Identical'):
        evaluate(result,labels)


def test_unknown_labels_not_silently_negative():
    result,labels=fixture();labels['tracks'][1]['label_status']='unknown'
    with pytest.raises(ValueError,match='Unknown'):
        evaluate(result,labels)
    output=evaluate(result,labels,allow_partial=True)
    assert output['evaluation_kind']=='provisional_partial_labels'
    assert output['counts']['unmatched_queries']==0
    assert output['metrics']['false_acceptance_rate'] is None


def test_unknown_matchability_not_treated_as_unmatched():
    result,labels=fixture();labels['tracks'][1]['query_status']='unknown'
    output=evaluate(result,labels)
    assert output['counts']['unmatched_queries']==0
    assert output['counts']['evaluated_queries']==3
    assert any('matchability unknown' in warning for warning in output['warnings'])
    assert output['evaluation_kind']=='provisional_partial_labels'
    assert output['counts']['unknown_matchability_queries']==1


@pytest.mark.parametrize('change',['job','hash','duplicate_label','missing_label','camera','no_identity','contradictory_unmatched'])
def test_invalid_labels_refused(change):
    result,labels=fixture()
    if change=='job':labels['job_id']='different'
    elif change=='hash':labels['source_input_hashes']={}
    elif change=='duplicate_label':labels['tracks'].append(copy.deepcopy(labels['tracks'][0]))
    elif change=='missing_label':labels['tracks'].pop()
    elif change=='camera':labels['tracks'][0]['camera_id']='camera_b'
    elif change=='no_identity':labels['tracks'][0]['identity_id']=''
    else:labels['tracks'][0]['query_status']='no_counterpart'
    with pytest.raises(ValueError):evaluate(result,labels)


def test_template_never_guesses_identity_or_footage_independence():
    result,_=fixture();labels=annotation_template(result)
    assert all(l['label_status']=='unknown' and l['identity_id']=='' for l in labels['tracks'])
    assert labels['footage_relationship']=='unconfirmed'


def test_outputs_never_overwrite_existing_data(tmp_path):
    path=tmp_path/'original.json';path.write_text('original')
    with pytest.raises(FileExistsError):write_new(path,{'new':1})
    assert path.read_text()=='original'


def test_fragment_hypotheses_are_advisory_and_do_not_merge_scores():
    a1=track('camera_a',1,[1,0]);a3=track('camera_a',3,[1,0],start=2);b1=track('camera_b',1,[1,0])
    cameras=[{'camera_id':'camera_a','tracks':[a1,a3]},{'camera_id':'camera_b','tracks':[b1]}]
    original=copy.deepcopy(cameras)
    hypotheses=fragmentation_hypotheses(cameras)
    assert hypotheses[0]['track_ids']==['camera_a:1','camera_a:3']
    assert hypotheses[0]['gap_seconds']==1
    assert hypotheses[0]['confirmed_same_identity'] is False
    assert cameras==original
    ranking=rank_candidates(cameras,MatchConfig())
    assert len(ranking['candidates'])==2
    assert all(c['verification']['outcome']=='UNCERTAIN' for c in ranking['candidates'])


def test_reported_score_pattern_is_valid_competition_not_identity_proof():
    # Mathematical fixture reproduces the reported pattern; NOT an observed video result.
    a1=track('camera_a',1,[.8246,math.sqrt(1-.8246**2)])
    a3=track('camera_a',3,[.9304,math.sqrt(1-.9304**2)])
    for sample in a1['appearance']['samples']:
        sample['embedding']=[.7198,math.sqrt(1-.7198**2)]
    b1=track('camera_b',1,[1,0])
    cameras=[{'camera_id':'camera_a','tracks':[a1,a3]},{'camera_id':'camera_b','tracks':[b1]}]
    result=rank_candidates(cameras,MatchConfig())
    row=next(r for r in result['candidates'] if r['query_track_id']=='camera_a:1')
    assert row['similarity']==pytest.approx(.8246)
    assert row['competition']['reverse_margin']==pytest.approx(-.1058)
    assert row['competition']['outcompeted'] is True
    assert row['competition']['near_tie'] is False
    assert row['competition']['reverse_best_query_track_id']=='camera_a:3'
    assert row['verification']['independent_visual_score']==pytest.approx(.7198)
    assert row['verification']['outcome']=='UNCERTAIN'
    assert {e['code'] for e in row['verification']['missing_evidence']} >= {'multi_crop_visual','temporal','competition'}
    assert row['verification']['contradictory_evidence']==[]
    inspection=inspect_analysis({'id':'synthetic-pattern','cameras':cameras,'matching':result})
    assert inspection['candidate_checks'][0]['score_delta']==pytest.approx(0)
    assert inspection['candidate_checks'][0]['recomputed_crop_agreement']==pytest.approx(.7198)
    assert inspection['candidate_checks'][0]['recomputed_rank']==1


def test_csv_contains_actual_evidence_and_abstention_rows():
    result,_=fixture()
    rows=list(csv.DictReader(io.StringIO(candidate_csv(result))))
    assert rows[0]['verification_outcome']=='ACCEPTED'
    assert rows[0]['similarity']=='0.5'
    assert rows[-1]['query_track_id']=='camera_a:4'
    assert rows[-1]['query_result']=='NO_RELIABLE_MATCH'
    assert rows[-1]['similarity']==''


def test_no_speed_invented_when_elapsed_time_unavailable():
    result,labels=fixture();result['elapsed_seconds']=None
    assert evaluate(result,labels)['processing']['processing_fps'] is None


def test_human_fragment_groups_remain_annotations_not_predicted_merges():
    result,labels=fixture()
    labels['tracks'][0]['fragment_group']='manual-one'
    labels['tracks'][2]['fragment_group']='manual-one'
    with pytest.raises(ValueError,match='different confirmed identities'):
        evaluate(result,labels)
    labels['tracks'][2]['identity_id']=labels['tracks'][0]['identity_id']
    original=copy.deepcopy(result)
    report=evaluate(result,labels)
    assert report['human_fragment_groups'][0]['track_ids']==['camera_a:1','camera_a:3']
    assert result==original


@pytest.mark.parametrize('value',[None,23,[]])
def test_non_string_identity_annotations_refused(value):
    result,labels=fixture();labels['tracks'][0]['identity_id']=value
    with pytest.raises(ValueError,match='strings'):
        evaluate(result,labels)


def test_cli_runs_reproducibly_on_explicit_synthetic_fixture_without_overwrite(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    result,labels=fixture()
    source=tmp_path/'SYNTHETIC-results.json';truth=tmp_path/'SYNTHETIC-labels.json';output=tmp_path/'SYNTHETIC-metrics.json'
    source.write_text(json.dumps(result));truth.write_text(json.dumps(labels))
    original=source.read_bytes()
    command=[sys.executable,'-m','backend.app.evaluation','evaluate','--results',str(source),'--labels',str(truth),'--output',str(output)]
    root=Path(__file__).resolve().parents[2]
    first=subprocess.run(command,cwd=root,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    report=json.loads(output.read_text())
    assert len(report['provenance']['results_sha256'])==64
    assert report['metrics']['accepted_match_precision']==.5
    previous=output.read_bytes()
    second=subprocess.run(command,cwd=root,capture_output=True,text=True)
    assert second.returncode==2
    assert output.read_bytes()==previous and source.read_bytes()==original


def test_inconsistent_export_outcomes_are_not_evaluated_as_valid_metrics():
    result,labels=fixture()
    result['matching']['queries'][0]['result']='NO_RELIABLE_MATCH'
    with pytest.raises(ValueError,match='conflicts'):
        evaluate(result,labels)

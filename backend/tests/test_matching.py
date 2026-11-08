"""Deterministic mathematical/rule fixtures, never used as application AI results."""
import math
import pytest
from app.config import MatchConfig
from app.embeddings import normalize, aggregate, cosine, select_observations
from app.matching import rank_candidates
from app.temporal import temporal_evidence


def track(camera, local_id, vector=(1., 0.), start=0.):
    observations = [{'frame_index': i, 'seconds': start + i * .5, 'bbox': [0, 0, 100, 100],
                     'confidence': .9, 'timestamp_source': 'decoder'} for i in range(3)]
    samples = [dict(o, embedding=list(vector), usable=True) for o in observations]
    return {'id': f'{camera}:{local_id}', 'camera_id': camera, 'local_track_id': local_id,
            'observations': observations, 'summary': {'start_seconds': start, 'end_seconds': start + 1,
            'observation_count': 3, 'mean_detection_confidence': .9, 'displacement_pixels': [30, 0]},
            'appearance': {'status': 'ready', 'vector': list(vector), 'samples': samples,
                           'within_track_similarity': 1.}}


def match(a, b, **options):
    return rank_candidates([{'tracks': a}, {'tracks': b}], MatchConfig(camera_b_offset_seconds=10, **options))


def test_normalization_and_quality_weighted_aggregation():
    vector = aggregate([[2, 0], [0, 3]], [3, 1])
    assert sum(v * v for v in vector) == pytest.approx(1)
    assert vector == pytest.approx(normalize([3, 1]))
    assert cosine([0, 4], [3, 0]) == pytest.approx(0)
    assert cosine([1, 2], [2, 4]) == pytest.approx(1)


@pytest.mark.parametrize('vector', [[], [0, 0], [float('nan')], [float('inf')]])
def test_invalid_vectors_do_not_produce_scores(vector):
    with pytest.raises(ValueError):
        normalize(vector)


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError):
        cosine([1, 2], [1])
    with pytest.raises(ValueError):
        aggregate([[1, 0], [1]], [1, 1])


def test_temporal_bin_selection_is_diverse_and_unique():
    obs = track('camera_a', 1)['observations']
    selected = select_observations(obs + [obs[0]], 8)
    assert [o['frame_index'] for o in selected] == [0, 1, 2]
    assert len(select_observations(obs, 2)) == 2
    assert select_observations([], 8) == []


def test_unique_supported_candidate_can_be_accepted():
    result = match([track('camera_a', 1)], [track('camera_b', 5)])
    pair = result['candidates'][0]
    assert pair['similarity'] == pytest.approx(1)
    assert pair['verification']['outcome'] == 'ACCEPTED'
    assert result['queries'][0]['result'] == 'MATCH_ACCEPTED'
    assert pair['verification']['uncertainty']['calibrated_probability'] is None


def test_highest_rank_not_forced_to_accept_without_timing():
    result = rank_candidates([{'tracks': [track('camera_a', 1)]}, {'tracks': [track('camera_b', 1)]}], MatchConfig())
    assert result['candidates'][0]['verification']['outcome'] == 'UNCERTAIN'
    assert result['queries'][0]['result'] == 'NO_RELIABLE_MATCH'


def test_lookalikes_are_ambiguous_even_when_only_top_one_exported():
    result = match([track('camera_a', 1)], [track('camera_b', 1), track('camera_b', 2, (.999, .001))], top_k=1)
    row = result['candidates'][0]
    assert row['competition']['ambiguous']
    assert row['competition']['competing_candidates'][0]['track_id'] == 'camera_b:2'
    assert row['verification']['outcome'] == 'UNCERTAIN'


def test_reciprocal_competition_prevents_many_to_one_acceptance():
    result = match([track('camera_a', 1), track('camera_a', 2)], [track('camera_b', 1)])
    assert all(row['verification']['outcome'] == 'UNCERTAIN' for row in result['candidates'])


def test_strong_temporal_contradiction_rejects_high_similarity():
    result = match([track('camera_a', 1)], [track('camera_b', 1)], min_travel_seconds=30)
    review = result['candidates'][0]['verification']
    assert review['outcome'] == 'REJECTED'
    assert any(e['code'] == 'temporal' for e in review['contradictory_evidence'])


def test_low_visual_agreement_rejects_when_evidence_usable():
    result = match([track('camera_a', 1)], [track('camera_b', 1, (0, 1))])
    assert result['candidates'][0]['verification']['outcome'] == 'REJECTED'


def test_aggregate_similarity_cannot_override_missing_sample_evidence():
    b = track('camera_b', 1)
    for sample in b['appearance']['samples']:
        sample['usable'] = False
    result = match([track('camera_a', 1)], [b])
    row = result['candidates'][0]
    assert row['similarity'] == 1
    assert row['verification']['outcome'] == 'UNCERTAIN'


def test_reviewer_checks_crop_vectors_independent_of_track_similarity():
    b = track('camera_b', 1)
    for sample in b['appearance']['samples']:
        sample['embedding'] = [0, 1]
    review = match([track('camera_a', 1)], [b])['candidates'][0]['verification']
    assert review['independent_visual_score'] == 0
    assert review['outcome'] == 'REJECTED'


def test_identical_inputs_never_accepted_or_claimed_as_evaluation():
    cameras = [{'tracks': [track('camera_a', 1)]}, {'tracks': [track('camera_b', 1)]}]
    result = rank_candidates(cameras, MatchConfig(camera_b_offset_seconds=10), identical_inputs=True)
    assert result['candidates'][0]['verification']['outcome'] == 'UNCERTAIN'
    assert result['evaluation']['identical_inputs']
    assert not result['evaluation']['external_camera_evaluation']


def test_no_destination_tracks_yields_no_reliable_match():
    result = match([track('camera_a', 1)], [])
    assert result['candidates'] == []
    assert result['queries'][0]['result'] == 'NO_RELIABLE_MATCH'


def test_temporal_offset_sign_uncertainty_and_reverse_route():
    a, b = track('camera_a', 1), track('camera_b', 1)
    config = MatchConfig(camera_b_offset_seconds=10, boundary_uncertainty_seconds=0, min_travel_seconds=9, max_travel_seconds=9)
    result = temporal_evidence(a, b, config)
    assert result['nominal_travel_seconds'] == 9
    assert result['status'] == 'supporting'
    config.clock_uncertainty_seconds = 2
    assert temporal_evidence(a, b, config)['status'] == 'missing'
    config.route, config.camera_b_offset_seconds = 'b_to_a', -10
    assert temporal_evidence(a, b, config)['nominal_travel_seconds'] == 9


def test_fps_fallback_prevents_false_temporal_certainty():
    a, b = track('camera_a', 1), track('camera_b', 1)
    a['observations'][0]['timestamp_source'] = 'fps_fallback'
    assert temporal_evidence(a, b, MatchConfig(camera_b_offset_seconds=10))['status'] == 'missing'


def test_configured_motion_contradiction():
    result = match([track('camera_a', 1)], [track('camera_b', 1)], camera_a_direction='left')
    assert result['candidates'][0]['verification']['outcome'] == 'REJECTED'


@pytest.mark.parametrize('options', [{'max_travel_seconds': -1}, {'max_travel_seconds': 5, 'min_travel_seconds': 10},
                                    {'camera_b_offset_seconds': math.inf}, {'top_k': 0}, {'typo': 1}])
def test_invalid_config_rejected(options):
    with pytest.raises(ValueError):
        MatchConfig(**options)


def test_weak_detection_blocks_acceptance():
    a = track('camera_a', 1)
    a['summary']['mean_detection_confidence'] = .2
    row = match([a], [track('camera_b', 1)])['candidates'][0]
    assert row['verification']['outcome'] == 'UNCERTAIN'
    assert 'weak_detection' in row['verification']['uncertainty']['blocking_reasons']


def test_tracking_instability_blocks_acceptance():
    a = track('camera_a', 1)
    a['appearance']['within_track_similarity'] = .6
    row = match([a], [track('camera_b', 1)])['candidates'][0]
    assert row['verification']['outcome'] == 'UNCERTAIN'
    assert 'track_instability' in row['verification']['uncertainty']['blocking_reasons']


def test_tracking_gap_blocks_acceptance():
    a = track('camera_a', 1)
    a['observations'][-1]['seconds'] = 5
    row = match([a], [track('camera_b', 1)])['candidates'][0]
    assert row['verification']['outcome'] == 'UNCERTAIN'
    assert 'tracking_gaps' in row['verification']['uncertainty']['blocking_reasons']


def test_direction_reliability_is_local_not_dependent_on_other_track():
    # B has reliable route-contradicting motion even when A's visual quality is poor.
    a, b = track('camera_a', 1), track('camera_b', 1)
    a['summary']['mean_detection_confidence'] = .2
    result = match([a], [b], camera_b_direction='left')
    review = result['candidates'][0]['verification']
    assert review['outcome'] == 'REJECTED'
    assert any(e['code'] == 'direction_conflict' for e in review['contradictory_evidence'])


def test_poor_motion_track_itself_cannot_supply_hard_direction_contradiction():
    a, b = track('camera_a', 1), track('camera_b', 1)
    b['summary']['mean_detection_confidence'] = .2
    review = match([a], [b], camera_b_direction='left')['candidates'][0]['verification']
    assert review['outcome'] == 'UNCERTAIN'
    assert not any(e['code'] == 'direction_conflict' for e in review['contradictory_evidence'])


@pytest.mark.parametrize('case_name,physical_label,expected', [
    ('sufficient_independent_evidence','SYNTHETIC_SAME','ACCEPTED'),
    ('same_but_unknown_timing','SYNTHETIC_SAME','UNCERTAIN'),
    ('strong_visual_contradiction','SYNTHETIC_DIFFERENT','REJECTED'),
    ('no_destination_observation','SYNTHETIC_UNMATCHED','NO_RELIABLE_MATCH'),
])
def test_explicitly_labeled_synthetic_decision_cases(case_name, physical_label, expected):
    a=track('camera_a',1)
    b=track('camera_b',1,(0,1) if physical_label=='SYNTHETIC_DIFFERENT' else (1,0))
    cameras=[{'camera_id':'camera_a','tracks':[a]},{'camera_id':'camera_b','tracks':[] if physical_label=='SYNTHETIC_UNMATCHED' else [b]}]
    config=MatchConfig() if case_name=='same_but_unknown_timing' else MatchConfig(camera_b_offset_seconds=10)
    output=rank_candidates(cameras,config)
    if expected=='NO_RELIABLE_MATCH':
        assert output['queries'][0]['result']==expected
        assert output['candidates']==[]
    else:
        review=output['candidates'][0]['verification']
        assert review['outcome']==expected
        if expected=='UNCERTAIN':
            assert not review['contradictory_evidence']
            assert any(e['code']=='temporal' for e in review['missing_evidence'])
        elif expected=='REJECTED':
            assert any(e['code']=='multi_crop_visual' for e in review['contradictory_evidence'])


def test_shared_crop_agreement_requires_bidirectional_support():
    from app.embeddings import crop_agreement
    # A perfect match to one crop cannot mask a majority of conflicting crops.
    assert crop_agreement([[1., 0.]], [[1., 0.], [0., 1.], [0., 1.]]) == 0.
    assert crop_agreement([[1., 0.]], [[1., 0.]]) == 1.


def test_shared_crop_agreement_rejects_missing_observations():
    from app.embeddings import crop_agreement
    with pytest.raises(ValueError, match='both tracks'):
        crop_agreement([], [[1., 0.]])

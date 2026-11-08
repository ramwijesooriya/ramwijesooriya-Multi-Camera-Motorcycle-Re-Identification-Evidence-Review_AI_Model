"""Independent decision policy: rechecks crop evidence, never accepts by rank alone.

Shares observed crops with retrieval, so it is an independent rules component,
not an independent sensor/model. Its uncalibrated decisions need validation.
"""
from .embeddings import crop_agreement
from .temporal import temporal_evidence

POLICY_VERSION = 'evidence-rules-v2'


def verify(a, b, context, config, identical_inputs=False):
    evidence = []

    def add(kind, code, message, value=None, blocking=False):
        evidence.append({'kind': kind, 'code': code, 'message': message, 'value': value, 'blocking': blocking})

    quality_ok = True
    sample_sets = []
    for track in (a, b):
        track_quality_ok = True
        samples = [s for s in track.get('appearance', {}).get('samples', []) if s['usable']]
        sample_sets.append(samples)
        summary = track['summary']
        span = summary['end_seconds'] - summary['start_seconds']
        if len(samples) < 2 or summary['observation_count'] < 3 or span < config.min_track_seconds:
            track_quality_ok = False
            add('missing', 'track_quality', 'Insufficient usable multi-frame evidence.',
                {'track_id': track['id'], 'usable_samples': len(samples), 'duration_seconds': span}, True)
        else:
            add('supporting', 'track_quality', 'Multiple usable observations support comparison.',
                {'track_id': track['id'], 'usable_samples': len(samples), 'duration_seconds': span})
        detection_confidence = summary.get('mean_detection_confidence')
        if detection_confidence is None or detection_confidence < config.min_detection_confidence:
            track_quality_ok = False
            add('missing', 'weak_detection', 'Detector evidence is weak or its confidence is unavailable.',
                {'track_id': track['id'], 'mean_detection_confidence': detection_confidence}, True)
        consistency = track.get('appearance', {}).get('within_track_similarity')
        if consistency is None or consistency < config.min_within_track_similarity:
            track_quality_ok = False
            add('missing', 'track_instability', 'Appearance varies within the track; possible occlusion, viewpoint change or ID switch.', consistency, True)
        observations = track['observations']
        gaps = [y['seconds'] - x['seconds'] for x, y in zip(observations, observations[1:])]
        if gaps and max(gaps) > config.max_tracking_gap_seconds:
            track_quality_ok = False
            add('missing', 'tracking_gaps', 'Observation gaps exceed the configured tracking limit.', {'track_id': track['id'], 'max_gap_seconds': max(gaps)}, True)
        expected = getattr(config, track['camera_id'] + '_direction')
        if expected == 'unknown':
            add('missing', 'direction_unknown', 'Camera-specific route direction is not configured.', track['id'])
        else:
            dx, dy = summary['displacement_pixels']
            along = {'right': dx, 'left': -dx, 'down': dy, 'up': -dy}[expected]
            if abs(along) < 10:
                add('missing', 'direction_weak', 'Too little image-space motion to verify direction.', track['id'], True)
            elif along < 0:
                # Fragmentation can invalidate motion: only a usable track gives a hard contradiction.
                add('contradictory' if track_quality_ok else 'missing', 'direction_conflict', 'Motion opposes configured camera route direction.', track['id'], True)
            else:
                add('supporting', 'direction_consistent', 'Motion agrees with configured camera route direction.', track['id'])
        quality_ok = quality_ok and track_quality_ok
    visual = None
    if all(sample_sets):
        visual = crop_agreement(
            [sample['embedding'] for sample in sample_sets[0]],
            [sample['embedding'] for sample in sample_sets[1]],
        )
        kind = 'supporting' if quality_ok and visual >= config.visual_accept else 'contradictory' if quality_ok and visual < config.visual_reject else 'missing'
        add(kind, 'multi_crop_visual', 'Bidirectional median nearest-crop cosine; not an identity probability.',
            {'score': visual, 'accept_threshold': config.visual_accept, 'reject_threshold': config.visual_reject}, kind == 'missing')
    else:
        add('missing', 'visual_unavailable', 'No usable visual comparison for one or both tracks.', blocking=True)
    temporal = temporal_evidence(a, b, config)
    add(temporal['status'], 'temporal', temporal['reason'], temporal, temporal['status'] == 'missing')
    if context['ambiguous'] or not context['mutual_best']:
        explanation = ('Another track scores higher in forward or reciprocal comparison. A negative margin is not a near tie or proof of a different physical vehicle.'
                       if context.get('outcompeted') else 'Competing camera-local tracks prevent a unique association. They may be lookalikes or fragments of one vehicle; identity is unresolved.')
        add('missing', 'competition', explanation, context, True)
    else:
        add('supporting', 'competition', 'Candidate is a mutual best match with sufficient separation from alternatives.', context)
    if identical_inputs:
        add('missing', 'identical_inputs', 'Byte-identical input videos are a smoke test, not independent-camera evidence.', blocking=True)
    add('missing', 'unmeasured_attributes', 'Plate identity and occlusion are not independently measured; generic visual descriptors are not motorcycle-specific.')
    contradictions = [e for e in evidence if e['kind'] == 'contradictory']
    missing = [e for e in evidence if e['kind'] == 'missing']
    if contradictions:
        outcome = 'REJECTED'
    elif any(e['blocking'] for e in missing):
        outcome = 'UNCERTAIN'
    else:
        outcome = 'ACCEPTED'
    return {'outcome': outcome, 'policy_version': POLICY_VERSION, 'evidence': evidence,
            'supporting_evidence': [e for e in evidence if e['kind'] == 'supporting'],
            'contradictory_evidence': contradictions, 'missing_evidence': missing,
            'uncertainty': {'calibrated_probability': None, 'blocking_reasons': [e['code'] for e in missing if e['blocking']],
                            'label': 'high' if outcome == 'UNCERTAIN' else 'rule_decision_uncalibrated'},
            'independent_visual_score': visual,
            'source_tracks': [a['id'], b['id']],
            'source_frames': {t['id']: [s['frame_index'] for s in t.get('appearance', {}).get('samples', [])] for t in (a, b)}}

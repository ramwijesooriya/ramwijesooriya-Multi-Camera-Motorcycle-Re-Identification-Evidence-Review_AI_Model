"""Pure cosine retrieval and competition analysis; verification is separate."""
from .embeddings import cosine
from .verification import verify, POLICY_VERSION
from .diagnostics import fragmentation_hypotheses


def rank_candidates(cameras, config, identical_inputs=False):
    a_tracks, b_tracks = cameras[0]['tracks'], cameras[1]['tracks']
    ready_a = [t for t in a_tracks if t.get('appearance', {}).get('status') == 'ready']
    ready_b = [t for t in b_tracks if t.get('appearance', {}).get('status') == 'ready']
    scores = {(a['id'], b['id']): cosine(a['appearance']['vector'], b['appearance']['vector']) for a in ready_a for b in ready_b}
    ranked = {a['id']: sorted(ready_b, key=lambda b: (-scores[a['id'], b['id']], b['id'])) for a in ready_a}
    reverse = {b['id']: sorted(ready_a, key=lambda a: (-scores[a['id'], b['id']], a['id'])) for b in ready_b}
    candidates, queries = [], []
    for a in a_tracks:
        rows = []
        for rank, b in enumerate(ranked.get(a['id'], [])[:config.top_k], 1):
            score = scores[a['id'], b['id']]
            alternatives = [other for other in ranked[a['id']] if other['id'] != b['id']]
            reverse_alts = [other for other in reverse[b['id']] if other['id'] != a['id']]
            margin = score - scores[a['id'], alternatives[0]['id']] if alternatives else None
            reverse_margin = score - scores[reverse_alts[0]['id'], b['id']] if reverse_alts else None
            near_tie = any(m is not None and abs(m) < config.ambiguity_margin for m in (margin, reverse_margin))
            outcompeted = any(m is not None and m < 0 for m in (margin, reverse_margin))
            context = {'margin': margin, 'reverse_margin': reverse_margin,
                       'mutual_best': rank == 1 and reverse[b['id']][0]['id'] == a['id'],
                       'ambiguous': near_tie or outcompeted,
                       'near_tie': near_tie, 'outcompeted': outcompeted,
                       'reverse_best_query_track_id': reverse[b['id']][0]['id'],
                       'reverse_best_similarity': scores[reverse[b['id']][0]['id'], b['id']],
                       'candidate_gallery_count': len(ready_b), 'source_query_count': len(ready_a),
                       'margin_threshold': config.ambiguity_margin,
                       'competing_candidates': [{'track_id': t['id'], 'similarity': scores[a['id'], t['id']]} for t in alternatives[:3]],
                       'competing_queries': [{'track_id': t['id'], 'similarity': scores[t['id'], b['id']]} for t in reverse_alts[:3]]}
            review = verify(a, b, context, config, identical_inputs)
            candidate = {'query_track_id': a['id'], 'candidate_track_id': b['id'], 'rank': rank,
                         'similarity': score, 'competition': context, 'verification': review}
            candidates.append(candidate)
            rows.append(candidate)
        accepted = [r['candidate_track_id'] for r in rows if r['verification']['outcome'] == 'ACCEPTED']
        queries.append({'query_track_id': a['id'], 'accepted_candidates': accepted,
                        'result': 'MATCH_ACCEPTED' if accepted else 'NO_RELIABLE_MATCH',
                        'reason': 'No accepted candidate; inspect uncertain/rejected evidence.' if rows and not accepted else 'No comparable destination tracks.' if not rows else 'Acceptance is an uncalibrated rule decision.'})
    return {'status': 'completed', 'candidates': candidates, 'queries': queries,
            'fragmentation_hypotheses': fragmentation_hypotheses(cameras),
            'policy_version': POLICY_VERSION, 'config': config.model_dump(),
            'evaluation': {'identical_inputs': identical_inputs, 'external_camera_evaluation': False,
                           'note': 'No ground-truth accuracy evaluation performed. Identical videos cannot validate cross-camera re-identification.'},
            'model_limitations': ['ImageNet-pretrained visual descriptors, not a motorcycle identity model.',
                                  'Similarity and rule outcomes are not calibrated identity probabilities.',
                                  'Reviewer shares crop observations with retrieval; it is a separate rules engine, not an independent sensor.']}

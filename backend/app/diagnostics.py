"""Read-only inspection of actual exports; hypotheses are never identity merges."""
import math
from .embeddings import cosine, crop_agreement


def fragmentation_hypotheses(cameras, max_gap_seconds=5.):
    hypotheses = []
    for camera in cameras:
        tracks = sorted(camera['tracks'], key=lambda t: t['summary']['start_seconds'])
        for index, first in enumerate(tracks):
            for second in tracks[index + 1:]:
                gap = second['summary']['start_seconds'] - first['summary']['end_seconds']
                if gap < 0 or gap > max_gap_seconds:
                    continue
                left = first.get('appearance', {}).get('vector')
                right = second.get('appearance', {}).get('vector')
                if left is None or right is None:
                    continue
                def center(observation):
                    x1, y1, x2, y2 = observation['bbox']
                    return (x1 + x2) / 2, (y1 + y2) / 2
                a = center(first['observations'][-1])
                b = center(second['observations'][0])
                hypotheses.append({'camera_id': camera['camera_id'], 'track_ids': [first['id'], second['id']],
                    'gap_seconds': gap, 'endpoint_distance_pixels': math.dist(a, b),
                    'within_camera_cosine': cosine(left, right), 'confirmed_same_identity': False,
                    'note': 'Non-overlapping tracklets within diagnostic window. May be different vehicles; inspect continuity, motion and original frames. Not automatically merged or removed from competition.'})
    return hypotheses


def inspect_analysis(result):
    cameras = result.get('cameras', [])
    by_id = {t['id']: t for c in cameras for t in c['tracks']}
    track_details = []
    for t in by_id.values():
        appearance = t.get('appearance', {})
        vector = appearance.get('vector')
        samples = appearance.get('samples', [])
        track_details.append({'track_id': t['id'], 'summary': t['summary'],
            'appearance_status': appearance.get('status', 'missing'), 'model': appearance.get('model'),
            'dimension': len(vector) if vector else None,
            'vector_norm': math.sqrt(sum(v*v for v in vector)) if vector else None,
            'sample_frames': [s['frame_index'] for s in samples],
            'usable_samples': sum(bool(s['usable']) for s in samples),
            'within_track_similarity': appearance.get('within_track_similarity'),
            'sample_evidence': [{key: s.get(key) for key in ('frame_index', 'seconds', 'crop', 'confidence', 'width', 'height', 'blur_variance', 'usable')} for s in samples]})
    pairs = []
    for pair in result.get('matching', {}).get('candidates', []):
        a, b = by_id[pair['query_track_id']], by_id[pair['candidate_track_id']]
        alternatives = []
        for other in cameras[0]['tracks']:
            if other['id'] != a['id'] and other.get('appearance', {}).get('status') == 'ready':
                alternatives.append({'track_id': other['id'], 'similarity': cosine(other['appearance']['vector'], b['appearance']['vector'])})
        alternatives.sort(key=lambda item: (-item['similarity'], item['track_id']))
        recomputed = cosine(a['appearance']['vector'], b['appearance']['vector'])
        reverse_margin = recomputed - alternatives[0]['similarity'] if alternatives else None
        gallery = [other for other in cameras[1]['tracks'] if other.get('appearance', {}).get('status') == 'ready']
        gallery.sort(key=lambda other: (-cosine(a['appearance']['vector'], other['appearance']['vector']), other['id']))
        forward_rank = next(i for i, other in enumerate(gallery, 1) if other['id'] == b['id'])
        sample_sets = [[s for s in t.get('appearance', {}).get('samples', []) if s.get('usable')] for t in (a, b)]
        recomputed_crop_agreement = None
        if all(sample_sets):
            recomputed_crop_agreement = crop_agreement(
                [sample['embedding'] for sample in sample_sets[0]],
                [sample['embedding'] for sample in sample_sets[1]],
            )
        pairs.append({'query_track_id': a['id'], 'candidate_track_id': b['id'],
            'exported_rank': pair['rank'], 'exported_similarity': pair['similarity'],
            'recomputed_rank': forward_rank, 'gallery_size': len(gallery),
            'recomputed_similarity': recomputed, 'score_delta': recomputed - pair['similarity'],
            'recomputed_crop_agreement': recomputed_crop_agreement,
            'recomputed_reverse_margin': reverse_margin, 'competing_queries': alternatives,
            'verification': pair['verification']})
    return {'job_id': result.get('id'), 'matching_status': result.get('matching', {}).get('status'),
            'saved_config': result.get('matching', {}).get('config'),
            'policy_version': result.get('matching', {}).get('policy_version'),
            'evaluation_metadata': result.get('matching', {}).get('evaluation'),
            'track_details': track_details, 'candidate_checks': pairs,
            'fragmentation_hypotheses': fragmentation_hypotheses(cameras),
            'limitations': ['Descriptors and hypotheses do not establish physical identity.',
                'No tracks are automatically stitched. Inspect crop references and original observation times.',
                'Diagnostic fragmentation window is five seconds; longer gaps require manual review.']}

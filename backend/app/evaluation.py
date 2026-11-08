"""Offline labeled evaluation. No inference, automatic labels, or example accuracy.

Run: python -m backend.app.evaluation --help (from repository root).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from .embeddings import cosine
from .diagnostics import inspect_analysis

EVALUATOR_VERSION = 'offline-evaluator-v1'

def _tracks(result):
    cameras = result.get('cameras', [])
    if [c['camera_id'] for c in cameras] != ['camera_a', 'camera_b']:
        raise ValueError('Expected camera_a then camera_b in a two-camera result')
    tracks = {t['id']: t for c in cameras for t in c['tracks']}
    if len(tracks) != sum(len(c['tracks']) for c in cameras):
        raise ValueError('Duplicate track IDs')
    for c in cameras:
        if any(t['camera_id'] != c['camera_id'] for t in c['tracks']):
            raise ValueError('Track camera mismatch')
    return tracks


def input_hashes(result):
    return {camera: detail['sha256'] for camera, detail in result.get('inputs', {}).items() if 'sha256' in detail}


def annotation_template(result):
    tracks = _tracks(result)
    return {'schema_version': 1, 'job_id': result['id'], 'source_input_hashes': input_hashes(result),
        'footage_relationship': 'unconfirmed', 'annotation_basis': '',
        'tracks': [{'track_id': t['id'], 'camera_id': t['camera_id'], 'label_status': 'unknown',
                    'identity_id': '', 'query_status': 'unknown', 'fragment_group': '', 'notes': ''} for t in tracks.values()]}


def _validated_labels(result, annotations, allow_partial):
    tracks = _tracks(result)
    if not isinstance(annotations, dict):
        raise ValueError('Annotations must be a JSON object')
    if annotations.get('schema_version') != 1 or annotations.get('job_id') != result.get('id'):
        raise ValueError('Annotation schema/job does not match the analysis')
    if annotations.get('footage_relationship') != 'independent_cameras':
        raise ValueError('Declare genuine independent cameras before real-world evaluation; duplicate/unknown footage is not valid')
    basis = annotations.get('annotation_basis')
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError('Document independent annotation evidence in annotation_basis')
    hashes = input_hashes(result)
    if annotations.get('source_input_hashes', {}) != hashes:
        raise ValueError('Annotation input fingerprints do not match the analysis')
    if (result.get('matching', {}).get('evaluation', {}).get('identical_inputs') or
            hashes.get('camera_a') and hashes.get('camera_a') == hashes.get('camera_b')):
        raise ValueError('Identical input videos cannot be used for cross-camera evaluation')
    records = annotations.get('tracks', [])
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError('Annotation tracks must be a list of label objects')
    labels = {r['track_id']: r for r in records}
    if len(labels) != len(records) or set(labels) != set(tracks):
        raise ValueError('Labels must contain every exported track exactly once; unknown is allowed only in partial evaluation')
    for track_id, label in labels.items():
        if label.get('camera_id') != tracks[track_id]['camera_id']:
            raise ValueError('Annotation camera mismatch')
        if label.get('label_status') not in ('confirmed', 'unknown', 'excluded_mixed'):
            raise ValueError('Invalid label_status')
        if label.get('query_status') not in ('unknown', 'has_counterpart', 'no_counterpart'):
            raise ValueError('Invalid query_status')
        if not isinstance(label.get('identity_id'), str):
            raise ValueError('Identity labels must be strings, not numeric guesses')
        if label['label_status'] == 'confirmed' and not label['identity_id'].strip():
            raise ValueError('Confirmed labels require an identity_id')
        if not isinstance(label.get('fragment_group', ''), str):
            raise ValueError('Fragment groups must be strings')
    fragment_groups = defaultdict(list)
    for label in labels.values():
        if label.get('fragment_group', '').strip():
            fragment_groups[(label['camera_id'], label['fragment_group'].strip())].append(label)
    for group in fragment_groups.values():
        identities = {l['identity_id'].strip() for l in group if l['label_status'] == 'confirmed'}
        if len(identities) > 1:
            raise ValueError('One same-camera fragment group cannot contain different confirmed identities')
    unknown = [key for key, label in labels.items() if label['label_status'] == 'unknown']
    if unknown and not allow_partial:
        raise ValueError('Unknown identity labels remain; finish annotation or explicitly use --allow-partial')
    return tracks, labels, unknown


def evaluate(result, annotations, ks=(1, 5), allow_partial=False):
    if result.get('matching', {}).get('status') != 'completed':
        raise ValueError('Matching did not complete; do not evaluate an extraction-only failure as an identity decision')
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError('K values must be positive integers')
    tracks, labels, unknown = _validated_labels(result, annotations, allow_partial)
    a_tracks = result['cameras'][0]['tracks']
    b_tracks = result['cameras'][1]['tracks']
    ready_b = [t for t in b_tracks if t.get('appearance', {}).get('status') == 'ready']
    candidates = result['matching'].get('candidates', [])
    accepted = defaultdict(list)
    seen_pairs = set()
    for pair in candidates:
        source, target = pair['query_track_id'], pair['candidate_track_id']
        if source not in tracks or target not in tracks or tracks[source]['camera_id'] != 'camera_a' or tracks[target]['camera_id'] != 'camera_b':
            raise ValueError('Candidate references an unknown or invalid camera pair')
        if (source, target) in seen_pairs:
            raise ValueError('Duplicate candidate pair')
        seen_pairs.add((source, target))
        outcome = pair['verification']['outcome']
        if outcome not in ('ACCEPTED', 'UNCERTAIN', 'REJECTED'):
            raise ValueError('Invalid verifier outcome')
        if outcome == 'ACCEPTED':
            accepted[source].append(target)
    query_records = result['matching'].get('queries', [])
    if len({q['query_track_id'] for q in query_records}) != len(query_records) or {q['query_track_id'] for q in query_records} != {t['id'] for t in a_tracks}:
        raise ValueError('Query summaries must cover every source track exactly once')
    for query in query_records:
        chosen = accepted[query['query_track_id']]
        expected = 'MATCH_ACCEPTED' if chosen else 'NO_RELIABLE_MATCH'
        if query.get('result') != expected:
            raise ValueError('Query summary conflicts with candidate verification outcomes')
        if 'accepted_candidates' in query and sorted(query['accepted_candidates']) != sorted(chosen):
            raise ValueError('Accepted candidate IDs conflict with query summary')
    rows, rank_hits, ap_values, relevant_recalls = [], Counter(), [], defaultdict(list)
    true_accepts = false_accepts = unevaluable_accepts = 0
    positive_queries = unmatched_queries = correct_accepted_queries = unmatched_false_queries = 0
    queries_with_acceptance = wrong_matchable_queries = 0
    unresolved_queries = 0
    warnings = []
    if len(input_hashes(result)) < 2:
        warnings.append('Input fingerprints unavailable: duplicate exclusion depends on manual source verification')
    for a in a_tracks:
        label = labels[a['id']]
        if label['label_status'] != 'confirmed':
            continue
        identity = label['identity_id'].strip()
        positive_ids = {b['id'] for b in b_tracks if labels[b['id']]['label_status'] == 'confirmed' and labels[b['id']]['identity_id'].strip() == identity}
        query_status = label['query_status']
        if positive_ids and query_status == 'no_counterpart':
            raise ValueError('A query declared unmatched has same-identity gallery labels')
        is_positive = bool(positive_ids) or query_status == 'has_counterpart'
        is_unmatched = query_status == 'no_counterpart'
        if not is_positive and not is_unmatched:
            unresolved_queries += 1
            warnings.append(f"{a['id']}: query matchability unknown; excluded, not assumed unmatched")
            continue
        ordering = []
        if a.get('appearance', {}).get('status') == 'ready':
            ordering = sorted(ready_b, key=lambda b: (-cosine(a['appearance']['vector'], b['appearance']['vector']), b['id']))
        ranks = [i for i, b in enumerate(ordering, 1) if b['id'] in positive_ids]
        average_precision = None
        if is_positive:
            positive_queries += 1
            if not positive_ids:
                warnings.append(f"{a['id']}: counterpart expected but no labeled gallery track represents it; counted as a miss")
            average_precision = sum(index / rank for index, rank in enumerate(ranks, 1)) / len(positive_ids) if positive_ids else 0.
            ap_values.append(average_precision)
            for k in ks:
                rank_hits[k] += bool(ranks and ranks[0] <= k)
                relevant_recalls[k].append(sum(r <= k for r in ranks) / len(positive_ids) if positive_ids else 0.)
        if is_unmatched:
            unmatched_queries += 1
        chosen = accepted[a['id']]
        queries_with_acceptance += bool(chosen)
        if is_unmatched and chosen:
            unmatched_false_queries += 1
        correct_here, wrong_here = False, False
        for target in chosen:
            target_label = labels[target]
            if target in positive_ids:
                true_accepts += 1
                correct_here = True
            elif is_unmatched or target_label['label_status'] == 'confirmed':
                false_accepts += 1
                wrong_here = True
            else:
                unevaluable_accepts += 1
        correct_accepted_queries += bool(is_positive and correct_here)
        wrong_matchable_queries += bool(is_positive and wrong_here)
        rows.append({'query_track_id': a['id'], 'identity_id': identity, 'matchable': is_positive,
                     'unmatched': is_unmatched, 'first_relevant_rank': ranks[0] if ranks else None,
                     'average_precision': average_precision, 'accepted_track_ids': chosen,
                     'correct_accepted': correct_here, 'false_accepted': wrong_here})
    if not rows:
        raise ValueError('No confirmed queries with known matchability; annotate correspondences/unmatched queries first')
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    grouped = defaultdict(list)
    for t in tracks.values():
        label = labels[t['id']]
        if label['label_status'] == 'confirmed':
            grouped[(t['camera_id'], label['identity_id'].strip())].append(t['id'])
    repeated = [{'camera_id': camera, 'identity_id': identity, 'track_ids': ids,
                 'note': 'Human-labeled same identity in several tracks; could be fragmentation or repeat passage. Production results remain track-level.'}
                for (camera, identity), ids in grouped.items() if len(ids) > 1]
    fragment_groups = defaultdict(list)
    for label in labels.values():
        if label.get('fragment_group', '').strip():
            fragment_groups[(label['camera_id'], label['fragment_group'].strip())].append(label['track_id'])
    human_fragments = [{'camera_id': camera, 'fragment_group': group, 'track_ids': ids,
                       'source': 'manual annotation only; no production merge or competition exemption'}
                      for (camera, group), ids in fragment_groups.items() if len(ids) > 1]
    frames = sum(c['frames_processed'] for c in result['cameras'])
    elapsed = result.get('elapsed_seconds')
    valid_elapsed = isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed > 0
    return {'schema_version': 1, 'job_id': result['id'], 'evaluator_version': EVALUATOR_VERSION,
        'evaluation_parameters': {'ks': list(ks), 'allow_partial': allow_partial},
        'evaluation_kind': 'provisional_partial_labels' if unknown or unresolved_queries or unevaluable_accepts or any(l['label_status'] == 'excluded_mixed' for l in labels.values()) else 'manually_labeled_detected_track_evaluation',
        'scope': 'Detected camera-A queries; not detection recall. Recall@K is query hit rate. No thresholds or runtime outcomes were changed.',
        'definitions': {'recall_at_k': 'fraction of known-matchable queries with at least one correct gallery track in top K',
                        'accepted_match_recall': 'known-matchable queries with a correct accepted candidate / known-matchable queries',
                        'false_acceptance_rate': 'confirmed unmatched queries receiving any accepted candidate / confirmed unmatched queries'},
        'counts': {'evaluated_queries': len(rows), 'matchable_queries': positive_queries, 'unmatched_queries': unmatched_queries,
                   'source_queries_total': len(a_tracks), 'gallery_tracks_total': len(b_tracks),
                   'gallery_missing_descriptors': len(b_tracks) - len(ready_b),
                   'unknown_matchability_queries': unresolved_queries,
                   'true_accepted_pairs': true_accepts, 'false_accepted_pairs': false_accepts,
                   'unevaluable_accepted_pairs': unevaluable_accepts, 'accepted_queries': queries_with_acceptance,
                   'correct_accepted_queries': correct_accepted_queries, 'unmatched_false_accepted_queries': unmatched_false_queries,
                   'wrong_accepted_matchable_queries': wrong_matchable_queries, 'unknown_labels': len(unknown),
                   'excluded_mixed_tracks': sum(l['label_status'] == 'excluded_mixed' for l in labels.values())},
        'metrics': {'rank_1_accuracy': ratio(rank_hits[1], positive_queries) if 1 in ks else None,
                    'recall_at_k': {str(k): ratio(rank_hits[k], positive_queries) for k in ks},
                    'gallery_recall_at_k': {str(k): sum(relevant_recalls[k]) / positive_queries if positive_queries else None for k in ks},
                    'mean_average_precision': sum(ap_values) / positive_queries if positive_queries else None,
                    'accepted_match_precision': ratio(true_accepts, true_accepts + false_accepts),
                    'accepted_match_recall': ratio(correct_accepted_queries, positive_queries),
                    'false_acceptance_rate': ratio(unmatched_false_queries, unmatched_queries),
                    'coverage': queries_with_acceptance / len(rows), 'abstention_rate': 1 - queries_with_acceptance / len(rows)},
        'processing': {'frames_total': frames, 'elapsed_seconds': elapsed if valid_elapsed else None,
                       'processing_fps': frames / elapsed if valid_elapsed else None,
                       'note': 'Measured export wall time only; inference speed is not test execution speed.'},
        'same_identity_track_groups': repeated, 'human_fragment_groups': human_fragments, 'per_query': rows, 'warnings': warnings,
        'review_outcome_counts': dict(Counter(p['verification']['outcome'] for p in candidates)),
        'limitations': ['Human labels require independent evidence. No probability calibration or automatic identity validation.',
                       'Multiple fragments of one identity count as multiple track-level queries/gallery items; inspect same_identity_track_groups.',
                       'Unknown gallery identities can invalidate relevance judgments in partial evaluation; do not submit partial numbers as final accuracy.']}


def read_json(path):
    with Path(path).open() as stream:
        def nonfinite(value):
            raise ValueError(f'Non-finite JSON number: {value}')
        return json.load(stream, parse_constant=nonfinite)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite original analyses, recordings, labels, or an older report.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('template', 'diagnose', 'evaluate'):
        child = sub.add_parser(command)
        child.add_argument('--results', required=True)
        child.add_argument('--output', required=True, help='New file only; never overwrites existing data')
        if command == 'evaluate':
            child.add_argument('--labels', required=True)
            child.add_argument('--ks', nargs='+', type=int, default=[1, 5])
            child.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    try:
        result = read_json(args.results)
        if args.command == 'template':
            output = annotation_template(result)
        elif args.command == 'diagnose':
            output = inspect_analysis(result)
        else:
            output = evaluate(result, read_json(args.labels), tuple(sorted(set([1] + args.ks))), args.allow_partial)
            output['provenance'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in [('results_sha256', args.results), ('labels_sha256', args.labels)]}
            output['provenance']['evaluator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_new(args.output, output)
        print(f'Wrote {args.output}. No runtime predictions or source files were modified.')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Evaluation/inspection refused: {error}\n')


if __name__ == '__main__':
    main()

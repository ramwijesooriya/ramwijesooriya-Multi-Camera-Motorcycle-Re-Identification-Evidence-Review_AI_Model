"""Tabular view of stored real prediction evidence; never generates predictions."""
import csv
import io
import json


def candidate_csv(result):
    output = io.StringIO(newline='')
    fields = ['job_id', 'matching_status', 'query_track_id', 'candidate_track_id', 'rank', 'similarity',
              'independent_crop_agreement', 'verification_outcome', 'query_result', 'supporting_evidence',
              'contradictory_evidence', 'missing_evidence', 'uncertainty']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    matching = result.get('matching', {})
    queries = {q['query_track_id']: q for q in matching.get('queries', [])}
    seen = set()
    for pair in matching.get('candidates', []):
        review = pair['verification']
        query_id = pair['query_track_id']
        seen.add(query_id)
        row = {'job_id': result['id'], 'matching_status': matching.get('status', 'unavailable'),
               'query_track_id': query_id, 'candidate_track_id': pair['candidate_track_id'],
               'rank': pair['rank'], 'similarity': pair['similarity'],
               'independent_crop_agreement': review.get('independent_visual_score'),
               'verification_outcome': review['outcome'], 'query_result': queries.get(query_id, {}).get('result', '')}
        for field in ('supporting_evidence', 'contradictory_evidence', 'missing_evidence', 'uncertainty'):
            row[field] = json.dumps(review.get(field), allow_nan=False)
        writer.writerow(row)
    for query_id, query in queries.items():
        if query_id not in seen:
            writer.writerow({'job_id': result['id'], 'matching_status': matching.get('status', 'unavailable'),
                             'query_track_id': query_id, 'query_result': query['result']})
    if not queries and not matching.get('candidates'):
        writer.writerow({'job_id': result['id'], 'matching_status': matching.get('status', 'unavailable')})
    return output.getvalue()

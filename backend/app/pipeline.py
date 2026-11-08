"""Real YOLO detections and independent ByteTrack state per camera."""
from pathlib import Path
import math


def observation_time(raw_seconds, previous_seconds, frame_index, fps):
    """Preserve monotonic relative time when decoder timestamps stall or reset."""
    valid = math.isfinite(raw_seconds) and raw_seconds >= 0 and (
        frame_index == 0 or raw_seconds > previous_seconds
    )
    if valid:
        return raw_seconds, 'decoder'
    return max(frame_index / fps, previous_seconds + 1 / fps), 'fps_fallback'


def summarize(observations):
    if not observations:
        raise ValueError('A track must contain observations')
    first, last = observations[0], observations[-1]
    def center(obs):
        x1, y1, x2, y2 = obs['bbox']
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    a, b = center(first), center(last)
    return {
        'start_seconds': first['seconds'], 'end_seconds': last['seconds'],
        'observation_count': len(observations),
        'mean_detection_confidence': sum(o['confidence'] for o in observations) / len(observations),
        'displacement_pixels': [b[0] - a[0], b[1] - a[1]],
        'representative_observations': sorted(observations, key=lambda o: o['confidence'], reverse=True)[:8],
        'quality_warnings': ['short_track'] if len(observations) < 3 else [],
    }


def process_video(path: Path, camera_id: str, output: Path, weights: str, progress=None):
    import cv2
    from ultralytics import YOLO
    # New model creates a fresh tracker, preventing state leakage between cameras/jobs.
    model = YOLO(weights)
    motorcycle_ids = [int(k) for k, v in model.names.items() if v == 'motorcycle']
    if not motorcycle_ids:
        raise ValueError('Model does not contain the motorcycle class')
    cap = cv2.VideoCapture(str(path))
    tracks = {}
    frame_index = 0
    previous_seconds = -1.0
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot decode video for {camera_id}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has invalid frame rate')
        total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        crop_dir = output / camera_id
        crop_dir.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            seconds, timestamp_source = observation_time(raw_seconds, previous_seconds, frame_index, fps)
            previous_seconds = seconds
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=motorcycle_ids,
                                 conf=0.15, verbose=False, device='cpu')[0]
            if result.boxes is not None and result.boxes.id is not None:
                for box, track_id, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    key = f'{camera_id}:{track_id}'
                    track = tracks.setdefault(key, {'id': key, 'camera_id': camera_id, 'local_track_id': track_id, 'observations': []})
                    observation = {'frame_index': frame_index, 'seconds': seconds,
                                   'timestamp_source': timestamp_source,
                                   'bbox': [x1, y1, x2, y2], 'confidence': confidence}
                    # Keep one best crop per track; all observations remain available for aggregation.
                    if confidence > track.get('best_crop_confidence', -1):
                        crop = crop_dir / f'track_{track_id}.jpg'
                        if not cv2.imwrite(str(crop), frame[y1:y2, x1:x2]):
                            raise RuntimeError('Failed to save evidence crop')
                        track['crop'] = f'{camera_id}/{crop.name}'
                        track['best_crop_confidence'] = confidence
                    track['observations'].append(observation)
            frame_index += 1
            if progress and frame_index % 30 == 0:
                progress({'camera_id': camera_id, 'frames_processed': frame_index, 'total_frames': total})
        if frame_index == 0:
            raise ValueError('Video contains no readable frames')
        if total and frame_index < total - 2:
            raise ValueError(f'Video decoding ended early: {frame_index}/{total} frames')
        for track in tracks.values():
            track['summary'] = summarize(track['observations'])
        return {'camera_id': camera_id, 'fps': fps, 'frames_processed': frame_index,
                'timing_note': 'Video-relative timestamps; camera clock offset is unknown.', 'tracks': list(tracks.values())}
    finally:
        cap.release()

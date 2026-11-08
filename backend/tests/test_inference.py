"""Opt-in real inference check; never substitutes fake model predictions."""
import os
from pathlib import Path
import pytest
from app.pipeline import process_video


@pytest.mark.skipif(not os.getenv('TEST_VIDEO_A') or not os.getenv('TEST_VIDEO_B'), reason='Supply two real videos to enable inference integration test')
def test_real_two_camera_inference(tmp_path):
    weights = os.getenv('YOLO_WEIGHTS', 'yolo11n.pt')
    results = [process_video(Path(os.environ[f'TEST_VIDEO_{suffix}']), camera, tmp_path, weights)
               for suffix, camera in [('A', 'camera_a'), ('B', 'camera_b')]]
    assert all(result['frames_processed'] > 0 for result in results)
    for result in results:
        for track in result['tracks']:
            assert track['id'].startswith(result['camera_id'] + ':')
            assert track['summary']['observation_count'] == len(track['observations'])
            assert (tmp_path / track['crop']).is_file()


def test_invalid_video_rejected(tmp_path):
    # Requires real CV/model dependencies and weights; opt in alongside footage.
    if not os.getenv('YOLO_WEIGHTS'):
        pytest.skip('Set YOLO_WEIGHTS to local weights to test decoder rejection')
    invalid = tmp_path / 'invalid.mp4'
    invalid.write_bytes(b'not video data')
    with pytest.raises(ValueError, match='Cannot decode'):
        process_video(invalid, 'camera_a', tmp_path, os.environ['YOLO_WEIGHTS'])


@pytest.mark.skipif(not os.getenv('TEST_VIDEO_A') or not os.getenv('TEST_VIDEO_B'), reason='Supply two real CCTV videos for full pipeline smoke test')
def test_real_two_camera_matching_smoke(tmp_path):
    import hashlib
    from app.config import MatchConfig
    from app.embeddings import VisualEncoder, extract_embeddings
    from app.matching import rank_candidates
    paths = [Path(os.environ['TEST_VIDEO_A']), Path(os.environ['TEST_VIDEO_B'])]
    encoder = VisualEncoder()
    config = MatchConfig()  # Unknown clock offset must never be silently assumed zero.
    cameras = []
    hashes = []
    for camera, path in zip(('camera_a', 'camera_b'), paths):
        result = process_video(path, camera, tmp_path, os.getenv('YOLO_WEIGHTS', 'yolo11n.pt'))
        extract_embeddings(result, path, tmp_path, config, encoder)
        cameras.append(result)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes.append(digest.hexdigest())
    result = rank_candidates(cameras, config, identical_inputs=hashes[0] == hashes[1])
    assert result['status'] == 'completed'
    assert result['evaluation']['external_camera_evaluation'] is False
    for row in result['candidates']:
        assert -1 <= row['similarity'] <= 1
        assert row['verification']['outcome'] in ('UNCERTAIN', 'REJECTED')
        assert row['verification']['evidence']

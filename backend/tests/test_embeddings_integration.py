"""Real pretrained inference on generated image inputs; pipeline integrity, not accuracy."""
from pathlib import Path
import math
import pytest


def test_real_pretrained_multi_crop_embeddings(tmp_path):
    import torch
    # Avoid network in tests. Docker prefetches this exact published checkpoint.
    cached = Path(torch.hub.get_dir()) / 'checkpoints/mobilenet_v3_small-047dcff4.pth'
    if not cached.exists():
        pytest.skip('Pretrained weights not cached; no random-weight substitution')
    import cv2
    import numpy as np
    from app.embeddings import VisualEncoder, extract_embeddings, cosine
    from app.config import MatchConfig
    from app.matching import rank_candidates
    from app.pipeline import summarize
    video = tmp_path / 'generated.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (128, 128))
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    for _ in range(12):
        writer.write(frame)
    writer.release()
    encoder = VisualEncoder()
    cameras = []
    for camera_id in ('camera_a', 'camera_b'):
        obs = [{'frame_index': i, 'seconds': i / 10, 'bbox': [8, 8, 120, 120], 'confidence': .9,
                'timestamp_source': 'decoder'} for i in range(12)]
        track = {'id': f'{camera_id}:1', 'camera_id': camera_id, 'local_track_id': 1,
                 'observations': obs, 'summary': summarize(obs)}
        camera = {'camera_id': camera_id, 'tracks': [track]}
        extract_embeddings(camera, video, tmp_path, MatchConfig(samples_per_track=3), encoder)
        appearance = track['appearance']
        assert appearance['model'] == 'torchvision/mobilenet_v3_small/IMAGENET1K_V1'
        assert appearance['dimension'] == 576
        assert appearance['sample_count'] == 3
        assert sum(v * v for v in appearance['vector']) == pytest.approx(1)
        assert all((tmp_path / s['crop']).exists() for s in appearance['samples'])
        assert all(math.isfinite(v) for v in appearance['vector'])
        cameras.append(camera)
    assert cosine(cameras[0]['tracks'][0]['appearance']['vector'], cameras[1]['tracks'][0]['appearance']['vector']) == pytest.approx(1)
    result = rank_candidates(cameras, MatchConfig(camera_b_offset_seconds=10), identical_inputs=True)
    assert result['candidates'][0]['verification']['outcome'] == 'UNCERTAIN'
    assert result['evaluation']['external_camera_evaluation'] is False

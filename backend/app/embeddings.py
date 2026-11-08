"""Genuine pretrained visual descriptors; no random-weight or fabricated fallback."""
from pathlib import Path
import math
import statistics


def normalize(vector):
    values = [float(v) for v in vector]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Embedding must contain finite values')
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        raise ValueError('Zero-length embedding')
    return [v / norm for v in values]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError('Embedding dimensions differ')
    a, b = normalize(a), normalize(b)
    return max(-1., min(1., sum(x * y for x, y in zip(a, b))))


def crop_agreement(left_vectors: list[list[float]], right_vectors: list[list[float]]) -> float:
    """Symmetric median nearest-crop cosine used by review and export inspection."""
    if not left_vectors or not right_vectors:
        raise ValueError('Crop agreement requires observations from both tracks')
    matrix = [[cosine(left, right) for right in right_vectors] for left in left_vectors]
    left_nearest = [max(row) for row in matrix]
    right_nearest = [max(row[index] for row in matrix) for index in range(len(matrix[0]))]
    return min(statistics.median(left_nearest), statistics.median(right_nearest))


def aggregate(vectors, weights):
    if not vectors or len(vectors) != len(weights):
        raise ValueError('Vectors and weights must be nonempty and aligned')
    vectors = [normalize(v) for v in vectors]
    if len({len(v) for v in vectors}) != 1:
        raise ValueError('Embedding dimensions differ')
    if not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError('Weights must be positive and finite')
    return normalize([sum(v[i] * w for v, w in zip(vectors, weights)) for i in range(len(vectors[0]))])


def select_observations(observations, count):
    """Highest-confidence/area sample per temporal bin, without duplicate frames."""
    ordered = sorted({o['frame_index']: o for o in observations}.values(), key=lambda o: o['frame_index'])
    bins = min(count, len(ordered))
    selected = []
    for i in range(bins):
        group = ordered[i * len(ordered) // bins:(i + 1) * len(ordered) // bins]
        def quality(o):
            x1, y1, x2, y2 = o['bbox']
            return o['confidence'] * math.sqrt(max(0, (x2 - x1) * (y2 - y1)))
        selected.append(max(group, key=quality))
    return selected


class VisualEncoder:
    name = 'torchvision/mobilenet_v3_small/IMAGENET1K_V1'

    def __init__(self):
        import torch
        import torchvision
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        # Raises if real weights are unavailable. Never replace with weights=None.
        self.model = mobilenet_v3_small(weights=weights).eval().cpu()
        self.model.classifier = torch.nn.Identity()
        self.transform = weights.transforms()
        self.metadata = {'model': self.name, 'weights_url': weights.url,
                         'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
                         'feature_layer': 'global average pool, classifier removed', 'dimension': 576,
                         'preprocessing': str(self.transform), 'training_domain': 'ImageNet classification; not motorcycle identity'}

    def encode(self, images):
        import torch
        with torch.inference_mode():
            batch = torch.stack([self.transform(image) for image in images])
            return [normalize(v) for v in self.model(batch).cpu().tolist()]


def extract_embeddings(camera, video_path: Path, output: Path, config, encoder):
    """One sequential second pass decodes selected observations for every track."""
    import cv2
    from PIL import Image
    requests = {}
    for track in camera['tracks']:
        track['appearance'] = {'model': encoder.name, 'samples': [], 'status': 'missing'}
        for index, obs in enumerate(select_observations(track['observations'], config.samples_per_track)):
            requests.setdefault(obs['frame_index'], []).append((track, index, obs))
    if not requests:
        return
    cap = cv2.VideoCapture(str(video_path))
    pending = []

    def flush():
        if not pending:
            return
        vectors = encoder.encode([item[2] for item in pending])
        if len(vectors) != len(pending):
            raise RuntimeError('Encoder output count mismatch')
        for (track, sample, _), vector in zip(pending, vectors):
            sample['embedding'] = vector
            track['appearance']['samples'].append(sample)
        pending.clear()

    try:
        if not cap.isOpened():
            raise ValueError('Cannot reopen source video for representative crops')
        frame_index = 0
        last = max(requests)
        while frame_index <= last:
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video ended before selected evidence frame')
            for track, index, obs in requests.get(frame_index, []):
                x1, y1, x2, y2 = obs['bbox']
                crop = frame[y1:y2, x1:x2]
                if not crop.size:
                    continue
                blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                h, w = crop.shape[:2]
                good = min(h, w) >= config.min_crop_side and blur >= config.min_blur_variance
                filename = f"track_{track['local_track_id']}_sample_{index}.jpg"
                relative = f"{camera['camera_id']}/{filename}"
                (output / camera['camera_id']).mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output / relative), crop):
                    raise RuntimeError('Cannot write representative crop')
                sample = {'frame_index': frame_index, 'seconds': obs['seconds'], 'crop': relative,
                          'bbox': obs['bbox'], 'confidence': obs['confidence'], 'width': w, 'height': h,
                          'blur_variance': blur, 'usable': good,
                          'weight': max(.01, float(obs['confidence'])) * min(1., min(h, w) / 96) * min(1., max(blur, 1) / 100)}
                pending.append((track, sample, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                if len(pending) >= 16:
                    flush()
            frame_index += 1
        flush()
    finally:
        cap.release()
    for track in camera['tracks']:
        appearance = track['appearance']
        samples = appearance['samples']
        if not samples:
            continue
        # Poor samples remain visible as evidence but never dominate a usable descriptor.
        usable = [s for s in samples if s['usable']]
        chosen = usable or samples
        vector = aggregate([s['embedding'] for s in chosen], [s['weight'] for s in chosen])
        appearance.update({'status': 'ready', 'vector': vector, 'dimension': len(vector),
                           'usable_samples': len(usable), 'sample_count': len(samples),
                           'within_track_similarity': statistics.median(cosine(s['embedding'], vector) for s in chosen),
                           'normalization': 'L2; quality-weighted mean of L2 crop vectors'})

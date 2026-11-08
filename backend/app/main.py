from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4
import json
import logging
import os
import threading
import time
import hashlib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError
from .pipeline import process_video
from .config import MatchConfig
from .embeddings import VisualEncoder, extract_embeddings
from .matching import rank_candidates
from .exports import candidate_csv

DATA = Path(os.getenv('DATA_DIR', 'data'))
WEIGHTS = os.getenv('YOLO_WEIGHTS', 'yolo11n.pt')
MAX_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(512 * 1024 * 1024)))
lock = threading.Lock()
logger = logging.getLogger(__name__)


def read_status(path: Path) -> dict:
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get('status') not in ('queued', 'processing', 'completed', 'failed'):
        raise ValueError('Missing status field')
    return record


def save_status(directory: Path, value: dict) -> None:
    with lock:
        target = directory / 'status.json'
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(value))
        temporary.replace(target)


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    for status in DATA.glob('*/status.json'):
        try:
            record = read_status(status)
        except (OSError, ValueError):
            # Preserve the damaged artifact and let unrelated jobs remain usable.
            logger.exception('Unreadable persisted job status: %s', status)
            continue
        if record['status'] in ('queued', 'processing'):
            save_status(status.parent, {'id': status.parent.name, 'status': 'failed', 'error': 'Processing interrupted; submit again.'})
    # A fresh executor belongs to each lifespan; a shut-down pool cannot restart.
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield
    finally:
        app.state.executor.shutdown(wait=True)


app = FastAPI(title='Motorcycle Re-Identification and Evidence Review', lifespan=lifespan)


def job_dir(job_id: str) -> Path:
    try:
        if str(UUID(job_id)) != job_id:
            raise ValueError()
    except ValueError:
        raise HTTPException(404, 'Unknown job')
    directory = DATA / job_id
    if not (directory / 'status.json').exists():
        raise HTTPException(404, 'Unknown job')
    return directory


def run_job(directory: Path) -> None:
    started = time.monotonic()
    try:
        def progress(detail):
            save_status(directory, {'id': directory.name, 'status': 'processing', 'progress': detail})
        results = []
        for camera in ('camera_a', 'camera_b'):
            progress({'camera_id': camera, 'frames_processed': 0})
            results.append(process_video(directory / f'{camera}.video', camera, directory, WEIGHTS, progress))
        # Preserve successful extraction even if appearance weights or review fail.
        extraction_elapsed = time.monotonic() - started
        config_path = directory / 'config.json'
        config = MatchConfig.model_validate_json(config_path.read_text()) if config_path.exists() else MatchConfig()
        inputs_path = directory / 'inputs.json'
        inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else {}
        identical = bool(inputs) and inputs['camera_a']['sha256'] == inputs['camera_b']['sha256']
        try:
            encoder = None
            if any(camera['tracks'] for camera in results):
                progress({'camera_id': 'appearance_embeddings', 'frames_processed': 0})
                encoder = VisualEncoder()
                for camera in results:
                    extract_embeddings(camera, directory / f"{camera['camera_id']}.video", directory, config, encoder)
            progress({'camera_id': 'ranking_and_verification', 'frames_processed': 0})
            matching = rank_candidates(results, config, identical)
            if encoder is not None:
                matching['encoder'] = encoder.metadata
        except Exception:
            logger.exception('Appearance/review stage failed for job %s', directory.name)
            matching = {'status': 'failed', 'candidates': [], 'queries': [], 'config': config.model_dump(),
                        'error': 'Matching unavailable; extraction is preserved. Check pretrained weights and server logs.'}
        elapsed = time.monotonic() - started
        result = {'id': directory.name, 'schema_version': 2, 'stage': 'evidence_review', 'cameras': results,
                  'elapsed_seconds': elapsed, 'processing_fps': sum(r['frames_processed'] for r in results) / elapsed,
                  'extraction_elapsed_seconds': extraction_elapsed, 'inputs': inputs, 'matching': matching,
                  'limitations': ['Generic appearance descriptors require motorcycle-specific validation.',
                                  'Camera calibration, plate recognition and ground-truth accuracy metrics are not provided.']}
        temporary = directory / 'results.tmp'
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(directory / 'results.json')
        save_status(directory, {'id': directory.name, 'status': 'completed'})
    except Exception:
        logger.exception('Job %s failed', directory.name)
        save_status(directory, {'id': directory.name, 'status': 'failed', 'error': 'Processing failed. Check video decoding, model availability, and server logs.'})


@app.get('/api/health')
def health():
    return {'status': 'ok', 'stage': 'track_extraction'}


@app.post('/api/jobs', status_code=202)
async def create_job(camera_a: UploadFile = File(...), camera_b: UploadFile = File(...), matching_config: str = Form('{}')):
    import shutil
    try:
        config = MatchConfig.model_validate_json(matching_config)
    except ValidationError as exc:
        await camera_a.close()
        await camera_b.close()
        # Do not echo non-finite inputs or non-JSON ValueError context into JSONResponse.
        raise HTTPException(422, [{'loc': list(e['loc']), 'msg': e['msg'], 'type': e['type']}
                                  for e in exc.errors()])
    directory = DATA / str(uuid4())
    directory.mkdir(parents=True)
    try:
        inputs = {}
        for camera, upload in [('camera_a', camera_a), ('camera_b', camera_b)]:
            size = 0
            digest = hashlib.sha256()
            with (directory / f'{camera}.video').open('wb') as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, 'Video exceeds upload limit')
                    target.write(chunk)
                    digest.update(chunk)
            if size == 0:
                raise HTTPException(400, 'Both videos must be nonempty')
            inputs[camera] = {'sha256': digest.hexdigest(), 'bytes': size}
        (directory / 'config.json').write_text(config.model_dump_json())
        (directory / 'inputs.json').write_text(json.dumps(inputs))
        save_status(directory, {'id': directory.name, 'status': 'queued'})
        app.state.executor.submit(run_job, directory)
        return {'id': directory.name, 'status': 'queued'}
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await camera_a.close()
        await camera_b.close()


@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    try:
        return read_status(job_dir(job_id) / 'status.json')
    except (OSError, ValueError):
        raise HTTPException(500, 'Stored job status is unreadable; preserved for inspection.')


@app.get('/api/jobs/{job_id}/results')
def results(job_id: str):
    path = job_dir(job_id) / 'results.json'
    if not path.exists():
        raise HTTPException(409, 'Results are not ready')
    return FileResponse(path, media_type='application/json', filename='tracks.json')


@app.get('/api/jobs/{job_id}/media/{camera}/{filename}')
def media(job_id: str, camera: str, filename: str):
    directory = job_dir(job_id)
    if camera not in ('camera_a', 'camera_b') or not filename.startswith('track_') or not filename.endswith('.jpg'):
        raise HTTPException(404, 'Unknown media')
    path = directory / camera / filename
    if not path.is_file() or path.resolve().parent != (directory / camera).resolve():
        raise HTTPException(404, 'Unknown media')
    return FileResponse(path)


@app.get('/api/jobs/{job_id}/results.csv')
def csv_results(job_id: str):
    path = job_dir(job_id) / 'results.json'
    if not path.exists():
        raise HTTPException(409, 'Results are not ready')
    try:
        content = candidate_csv(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(500, 'Stored results are unreadable; preserved for inspection.')
    return Response(content, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="candidates.csv"'})

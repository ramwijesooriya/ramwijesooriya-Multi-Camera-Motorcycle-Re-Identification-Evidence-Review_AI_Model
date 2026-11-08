import json
import pytest
from fastapi.testclient import TestClient
from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    # Do not start inference in API contract tests; no fabricated AI output.
    with TestClient(main.app) as client:
        monkeypatch.setattr(main.app.state.executor, 'submit', lambda *args: None)
        yield client


def test_health(client):
    assert client.get('/api/health').json()['status'] == 'ok'


def test_two_inputs_required(client):
    assert client.post('/api/jobs', files={'camera_a': ('a.mp4', b'video')}).status_code == 422


def test_empty_upload_cleaned(client):
    response = client.post('/api/jobs', files={'camera_a': ('a.mp4', b''), 'camera_b': ('b.mp4', b'x')})
    assert response.status_code == 400
    assert not list(main.DATA.iterdir())


def test_upload_limits(client, monkeypatch):
    monkeypatch.setattr(main, 'MAX_BYTES', 2)
    response = client.post('/api/jobs', files={'camera_a': ('a', b'123'), 'camera_b': ('b', b'1')})
    assert response.status_code == 413
    assert not list(main.DATA.iterdir())


def test_queue_and_not_ready(client):
    response = client.post('/api/jobs', files={'camera_a': ('a', b'1'), 'camera_b': ('b', b'2')})
    assert response.status_code == 202
    job_id = response.json()['id']
    assert client.get(f'/api/jobs/{job_id}').json()['status'] == 'queued'
    assert client.get(f'/api/jobs/{job_id}/results').status_code == 409
    assert (main.DATA / job_id / 'camera_a.video').read_bytes() == b'1'


def test_unknown_job(client):
    assert client.get('/api/jobs/not-a-uuid').status_code == 404


def test_pipeline_failure_is_persisted(client, monkeypatch):
    response = client.post('/api/jobs', files={'camera_a': ('a', b'1'), 'camera_b': ('b', b'2')})
    job_id = response.json()['id']
    def fail(*args):
        raise ValueError('invalid video')
    monkeypatch.setattr(main, 'process_video', fail)
    main.run_job(main.DATA / job_id)
    assert client.get(f'/api/jobs/{job_id}').json()['status'] == 'failed'
    assert not (main.DATA / job_id / 'results.json').exists()


def test_interrupted_job_marked_failed(tmp_path, monkeypatch):
    directory = tmp_path / 'previous'
    directory.mkdir()
    (directory / 'status.json').write_text(json.dumps({'status': 'processing'}))
    monkeypatch.setattr(main, 'DATA', tmp_path)
    with TestClient(main.app):
        assert json.loads((directory / 'status.json').read_text())['status'] == 'failed'


def test_executor_recreated_on_lifespan_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    for _ in range(2):
        with TestClient(main.app):
            assert main.app.state.executor.submit(lambda: 'ready').result(timeout=2) == 'ready'


@pytest.mark.parametrize('config', ['{', '{"min_travel_seconds":5,"max_travel_seconds":1}', '{"camera_b_offset_seconds":NaN}'])
def test_invalid_matching_config_before_job_created(client, config):
    response = client.post('/api/jobs', data={'matching_config': config},
                           files={'camera_a': ('a', b'1'), 'camera_b': ('b', b'2')})
    assert response.status_code == 422
    assert not list(main.DATA.iterdir())


def test_matching_config_and_input_hashes_persisted(client):
    response = client.post('/api/jobs', data={'matching_config': '{"camera_b_offset_seconds":12}'},
                           files={'camera_a': ('a', b'same'), 'camera_b': ('b', b'same')})
    directory = main.DATA / response.json()['id']
    inputs = json.loads((directory / 'inputs.json').read_text())
    assert inputs['camera_a']['sha256'] == inputs['camera_b']['sha256']
    assert json.loads((directory / 'config.json').read_text())['camera_b_offset_seconds'] == 12


def test_embedding_failure_preserves_extraction(client, monkeypatch):
    response = client.post('/api/jobs', files={'camera_a': ('a', b'1'), 'camera_b': ('b', b'2')})
    directory = main.DATA / response.json()['id']
    # Unit-test orchestration only: do not represent this fixture as model inference.
    monkeypatch.setattr(main, 'process_video', lambda path, camera, *args: {'camera_id': camera, 'frames_processed': 2, 'tracks': [{'id': camera + ':1'}]})
    def no_weights():
        raise RuntimeError('weights unavailable')
    monkeypatch.setattr(main, 'VisualEncoder', no_weights)
    main.run_job(directory)
    result = json.loads((directory / 'results.json').read_text())
    assert result['cameras'][0]['frames_processed'] == 2
    assert result['matching']['status'] == 'failed'
    assert result['matching']['candidates'] == []
    assert client.get(f"/api/jobs/{directory.name}").json()['status'] == 'completed'


@pytest.mark.parametrize('content', ['{broken', 'null', '{}', '{"status":null}', '{"status":"invalid"}'])
def test_corrupt_status_does_not_prevent_startup(tmp_path, monkeypatch, content):
    from uuid import uuid4
    directory = tmp_path / str(uuid4())
    directory.mkdir()
    path = directory / 'status.json'
    path.write_text(content)
    monkeypatch.setattr(main, 'DATA', tmp_path)
    with TestClient(main.app) as client:
        assert client.get('/api/health').status_code == 200
        response = client.get(f'/api/jobs/{directory.name}')
        assert response.status_code == 500
        assert 'preserved' in response.json()['detail']
    assert path.read_text() == content


def test_json_export_preserves_stored_evidence_artifact(client):
    response = client.post('/api/jobs', files={'camera_a': ('a', b'1'), 'camera_b': ('b', b'2')})
    job_id = response.json()['id']
    # Serialization contract fixture, not a fabricated AI result or metric.
    artifact = {'schema_version': 2, 'id': job_id, 'cameras': [],
                'matching': {'status': 'completed', 'candidates': [], 'queries': [],
                             'evaluation': {'external_camera_evaluation': False}}}
    (main.DATA / job_id / 'results.json').write_text(json.dumps(artifact))
    exported = client.get(f'/api/jobs/{job_id}/results')
    assert exported.status_code == 200
    assert exported.headers['content-type'] == 'application/json'
    assert 'attachment' in exported.headers['content-disposition']
    assert exported.json() == artifact


def test_csv_export_waits_for_actual_results(client):
    response=client.post('/api/jobs', files={'camera_a': ('a', b'1'), 'camera_b': ('b', b'2')})
    job=response.json()['id']
    assert client.get(f'/api/jobs/{job}/results.csv').status_code==409
    artifact={'id':job,'matching':{'status':'completed','candidates':[], 'queries':[{'query_track_id':'camera_a:1','result':'NO_RELIABLE_MATCH'}]}}
    (main.DATA/job/'results.json').write_text(json.dumps(artifact))
    exported=client.get(f'/api/jobs/{job}/results.csv')
    assert exported.status_code==200
    assert exported.headers['content-type'].startswith('text/csv')
    assert 'NO_RELIABLE_MATCH' in exported.text
    assert (main.DATA/job/'results.json').read_text()==json.dumps(artifact)

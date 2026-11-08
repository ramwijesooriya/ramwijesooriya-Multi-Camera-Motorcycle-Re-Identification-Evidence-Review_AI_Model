# MotoTrace AI

**Multi-Camera Motorcycle Re-Identification & Evidence Review** — a local application for comparing motorcycle tracks from two recorded CCTV videos. It uses genuine pretrained inference, ranks candidate identities, and independently evaluates the evidence instead of forcing the highest-ranked match to be accepted. Core functions require no paid credentials.

This README is the primary submission document. Detailed development histories, original docs, logs, raw analyses and rollback archives remain locally intact but excluded from Git. Development and review used **Codex-assisted tooling**; that provenance and the verification limitations are not concealed.

## Architecture and requirements

```text
React / TypeScript / Vite → Nginx :8080 → FastAPI background worker
YOLO → camera-local ByteTrack → representative crops → MobileNetV3
     → normalized track features → cosine ranking → evidence review
```

OpenCV decodes videos and crops. PyTorch/torchvision compute features. A single worker persists jobs, observations, crops and JSON results in the Docker `evidence` volume. `backend/app/` separates tracking, embeddings, matching, temporal checks, verification, exports and evaluation; `frontend/src/` contains the existing UI components. No additional database or queue service is needed.

Use Docker Desktop with Compose, or Python **3.11** and Node **22+** for local development. The first build needs internet for packages and genuine upstream model weights. Docker selects CPU-only PyTorch/torchvision wheels and caches weights for a non-root runtime user.

## Installation and usage

```sh
docker compose up --build
```

Open **http://localhost:8080**:

1. Upload two CCTV clips. Each file is limited to 512 MiB by default.
2. Enter camera timing and route metadata only when known; unknown is not zero.
3. Select **Extract tracks and review matches**.
4. Inspect independent side-by-side playback, moving boxes/local track IDs, crop strips, ranked candidates and supporting/conflicting/missing evidence.
5. Download JSON/CSV. Human ground-truth annotations are separate from model predictions.

Preserve the `evidence` volume when rebuilding. Existing completed results are not rewritten by code updates. H.264 MP4 is broadly browser-compatible; OpenCV decoding does not guarantee browser playback. Selected local file URLs and the current view do not survive refresh as a job-history player.

For development, reuse working dependencies and create/install only when needed:

```sh
python3.11 -m venv .venv-py311  # only if absent
.venv-py311/bin/python -m pip install -r backend/requirements.txt
(cd frontend && npm ci)
(cd backend && ../.venv-py311/bin/uvicorn app.main:app --reload --port 8000)
# Second terminal:
(cd frontend && npm run dev -- --port 8080)
```

Vite proxies `/api` to port 8000. Settings are `DATA_DIR`, `YOLO_WEIGHTS`, `MAX_UPLOAD_BYTES`, and torch's cache / `TORCH_HOME`. Use one Uvicorn worker. Interrupted jobs become failed, not automatically resumed. Successful extraction remains available if appearance inference/review fails.

## Matching methodology

YOLO detects motorcycles; each camera has independent ByteTrack state. Camera-local IDs are not physical cross-camera identities. Tracks retain actual bounding boxes, timestamps, confidence and trajectories.

Multiple unique observations per track are sampled and saved as crops. Genuine ImageNet-pretrained MobileNetV3-Small features, with the classifier removed, produce 576-dimensional vectors. Confidence/size/blur-weighted aggregation of L2-normalized crop features produces an L2-normalized track descriptor.

Cosine similarity ranks cross-camera candidates. Forward and reciprocal margins consider the full gallery before top-K export. A negative reciprocal margin means another source track scores higher for the destination; it is not a negative similarity or proof of different identity. One-item galleries provide a weak discrimination test. Fragmentation hints remain advisory: similar local tracklets are never automatically merged or removed from competition.

### Timing and movement

Add B's signed offset to B timestamps to align with A, including clip-start and clock differences. Unknown offset is null, not zero. For A→B, travel is aligned B start minus A end; reverse routes are supported. Clock and observation-boundary uncertainty expand the possible interval:

- Entire interval within route limits supports timing.
- A reliable interval entirely outside limits contradicts the route.
- Unknown timing, partial overlap or FPS-derived timestamps remain missing/uncertain evidence.

Unknown timing blocks acceptance. Image-space direction is configured independently per camera; it is not world-space camera calibration. Track endpoints are only proxies for route entry/exit.

## Independent evidence review

A separate structured-rules component rechecks crop-level agreement, track duration/count/confidence, within-track consistency, gaps, timing, movement, competing candidates and missing information. It shares crops and the encoder with retrieval; independence is in decision logic, not an independent visual sensor or separately trained model.

| Outcome | Meaning |
| --- | --- |
| **ACCEPTED** | Sufficient support without a reliable contradiction or blocking omission |
| **UNCERTAIN** | Weak/missing observations, unknown timing or unresolved competition |
| **REJECTED** | Reliable contradictory evidence under the policy |
| **NO RELIABLE MATCH** | No candidate is accepted for the query |

Rank alone never forces acceptance. Experimental defaults remain crop support ≥ 0.90, rejection < 0.55 with adequate evidence, separation ≥ 0.04 and explicit quality safeguards. Every setting remains in `backend/app/config.py`. Similarity and detector confidence are not identity probabilities; calibrated identity probability is null. Identical inputs cannot produce accepted pairs or validate external-camera accuracy.

## API and exports

| Method/path | Purpose |
| --- | --- |
| `GET /api/health` | HTTP liveness, not model readiness/accuracy |
| `POST /api/jobs` | Multipart `camera_a`, `camera_b`, optional JSON-string `matching_config` |
| `GET /api/jobs/{id}` | Status, progress and errors |
| `GET /api/jobs/{id}/results` | Full evidence JSON |
| `GET /api/jobs/{id}/results.csv` | Candidate/evidence CSV, including abstention rows |
| `GET /api/jobs/{id}/media/{camera}/{filename}` | Best or representative crop |

JSON schema version 2 retains observations, descriptors, frame references, scores, competition, supporting/contradictory/missing evidence, outcomes, uncertainty, fingerprints, policy and processing time. Results return 409 before available; invalid configuration returns 422. CLI outputs refuse to overwrite existing files.

## Automated checks and observed results

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-py311/bin/python -m pytest -c backend/pytest.ini backend/tests -q
npm test --prefix frontend
npm run build --prefix frontend
.venv-py311/bin/python -m pip check
docker compose config --quiet
```

| Check | Observed result |
| --- | --- |
| Backend tests | **92 passed, 3 footage-dependent skips** |
| Frontend tests | **13 passed** |
| Production build, compilation, dependency and Compose checks | **Passed** |
| Current live Docker/browser operation and overlay alignment | **Not verified here** |
| Private-video smoke tests | **Not run: inputs/runtime unavailable** |
| Held-out real-camera accuracy and recorded demonstration | **Not supplied** |

Tests include real cached pretrained inference on generated frame crops and explicitly synthetic rule/metric fixtures. Neither is real-camera accuracy. Frontend tests exercise rendering/logic, not live browser interaction. A clean environment without cached appearance weights may skip the pretrained integrity test too. Full test logs remain locally under ignored `reports/`; no evidence has been deleted.

For actual private clips and local genuine YOLO weights:

```sh
TEST_VIDEO_A='/absolute/camera-a.mp4' \
TEST_VIDEO_B='/absolute/camera-b.mp4' \
YOLO_WEIGHTS='/absolute/yolo11n.pt' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv-py311/bin/python -m pytest -c backend/pytest.ini backend/tests/test_inference.py -v
```

## Ground truth and reproducible evaluation

Use genuinely different camera views, independently confirmed identities, lookalike negatives and confirmed unmatched queries. Re-encoded copies are still duplicates even if hashes differ. Keep development/calibration and held-out test footage/identities separate where feasible; freeze thresholds before evaluation.

Save the actual JSON export under `evaluation/exports/latest-analysis.json`, or obtain it with the real UI job ID without overwriting an existing file:

```sh
mkdir -p evaluation/exports
JOB_ID='replace-with-the-real-job-UUID'
(set -C; curl --fail "http://localhost:8080/api/jobs/${JOB_ID}/results" > evaluation/exports/latest-analysis.json)
```

Inspect existing results and create a label draft without rerunning inference:

```sh
.venv-py311/bin/python -m backend.app.evaluation diagnose --results evaluation/exports/latest-analysis.json --output evaluation/diagnosis-latest.json
.venv-py311/bin/python -m backend.app.evaluation template --results evaluation/exports/latest-analysis.json --output evaluation/labels-latest.json
```

The UI can also create browser-local annotation drafts and download labels JSON. Labels **never change model predictions**. Independently annotate:

- `footage_relationship`: `independent_cameras` only after verification; duplicates/unconfirmed footage are refused.
- `annotation_basis`: evidence establishing identity; the model's score/outcome cannot be its own label.
- `label_status`: `confirmed`, `unknown`, or `excluded_mixed` for an ID-switched/mixed track.
- `identity_id`: the same case-sensitive string across cameras only for confirmed physical correspondence.
- `query_status`: `has_counterpart`, `no_counterpart`, or `unknown`; unknown is never silently a negative.
- `fragment_group`: optional manually confirmed same-camera continuity group, not an automatic merge. Repeat passage is not necessarily fragmentation.

Keep job IDs/fingerprints intact and include each exported track exactly once. Conflicting labels are refused. Run:

```sh
.venv-py311/bin/python -m backend.app.evaluation evaluate --results evaluation/exports/latest-analysis.json --labels evaluation/labels-latest.json --ks 1 3 5 --output evaluation/metrics-latest.json
```

The evaluator preserves runtime decisions, recomputes full-gallery ranking from real descriptors rather than truncated UI results, and records checksums, denominators, exclusions, warnings and per-query details. Output paths must be new. Unknown labels are refused by default; `--allow-partial` is exploratory and explicitly provisional.

| Metric | Definition |
| --- | --- |
| Rank-1 / Recall@K | Fraction of known-matchable detected queries with at least one correct gallery track in top K |
| Gallery Recall@K / mAP | Fraction of relevant gallery tracklets retrieved / full-gallery average precision |
| Accepted precision | Correct accepted labeled pairs / evaluable accepted pairs |
| Accepted recall | Matchable queries with a correct acceptance / known-matchable queries |
| False acceptance | Confirmed unmatched queries receiving acceptance / confirmed unmatched queries; wrong acceptances on matchable queries are counted separately |
| Abstention | Evaluated queries with no accepted pair / evaluated queries |
| Processing speed | Both cameras' total frames / actual exported wall time; test duration is not inference speed |

Undefined ratios are null, not perfect accuracy. Multiple fragments count as multiple track-level items; disclose this weighting and group/exclusion counts. These metrics do not measure missed detector events that never became query tracks. Report false matches/rejections and excessive abstention, not just successes. Retain raw labels, exports, policy, code/model versions, hardware and timing. No real accuracy numbers have been invented or supplied here.

## Material limitations

- Generic classification features may encode riders/backgrounds and confuse lookalikes, especially under occlusion, viewpoint/light changes and low resolution. Correlated crops do not guarantee independent evidence.
- ByteTrack can fragment/switch IDs. Blur, confidence, consistency and gap checks are proxies, not proof of visibility or tracker correctness. Several fragments can compete and cause conservative abstention.
- No plate OCR or automatic clock synchronization is provided. Clock alignment is manual; image displacement is not a complete world-space trajectory. The route model assumes meaningful, non-overlapping entry/exit observations.
- The reviewer shares visual evidence with retrieval and can share its mistakes. Thresholds are uncalibrated; high precision with zero useful acceptance is not a successful identity system.
- Hashes detect byte-identical files only. Independent-camera provenance and physical identity still require human verification.
- Browser codecs, frame alignment, seeking, resizing/letterboxing and overlay interaction require live checks. Local URLs do not provide persisted replay/history.
- This is a trusted local, single-worker service without authentication, resumable/distributed jobs or scale testing. Long videos and per-frame/vector exports may consume substantial time, memory and disk.
- Docker configuration validation does not prove runtime success. The sandbox cannot access the host daemon/service. Real labeled metrics and a demonstration recording remain required evidence; scripts/tests do not replace them.

## Demonstration — 4 minutes 30 seconds

Preprocess genuine clips and keep completed tabs/files selected before recording. Prepare independently labeled examples; do not change timing/thresholds to manufacture outcomes.

| Time | Demonstrate |
| --- | --- |
| 0:00–0:25 | localhost:8080, two genuine camera sources, job ID and actual Compose status |
| 0:25–0:55 | Independent playback, synchronized observed boxes/IDs and representative crops |
| 0:55–1:35 | A real correct match: distinguish independent ground truth, candidate rank and reviewer outcome |
| 1:35–2:10 | A genuine lookalike negative and competing margins; disclose any false acceptance |
| 2:10–2:45 | A rejected pair and its actual reliable contradictory evidence |
| 2:45–3:20 | An uncertain/no-reliable-match query and blocking missing evidence |
| 3:20–3:55 | JSON/CSV, source-frame traceability, actual labeled metrics/denominators and measured hardware/speed |
| 3:55–4:30 | Automated checks, shared-encoder/clock/tracking limitations and remaining gaps |

This is a script, not a recording. If no real accepted/correct example or measured metric exists, disclose the missing evidence. Synthetic state tests and identical-video self-similarity do not fulfill the real-case demonstration. Explain why a cosine is not a probability, why reciprocal competition can arise from fragments, and what makes verification independent.

## Submission and attribution

The exact intended public set is **40 files** in `SUBMISSION_FILES.txt`. Local environments, dependencies, caches, builds, logs/reports, original docs/history, weights, private recordings/analyses and checkpoints remain on disk and Git-ignored. No deletion is required for a minimal source submission.

```sh
git init
git add --pathspec-from-file=SUBMISSION_FILES.txt
git diff --cached --name-only
git diff --cached --check
git status --short
# Review the full staged diff before committing/publishing.
```

Git is uninitialized in this workspace; sandbox permissions prevent project metadata writes. No project files were staged or published here. On the host, verify Docker startup, `/api/health`, browser playback/overlays and exports; complete real evaluation and the recording. Submit reviewed technical results/demo separately when private material is involved. No project `LICENSE` currently exists: make an owner decision and update the allowlist for any newly approved license/public artifact.

Development/review used **Codex-assisted tooling**. Original audits, reports and documentation remain locally recoverable; exclusion from Git does not conceal assistance or material limitations. Source history must not be rewritten to disguise provenance.

Detection/tracking uses [Ultralytics](https://docs.ultralytics.com/modes/track/) ([AGPL-3.0 package license](https://github.com/ultralytics/ultralytics/blob/v8.3.230/LICENSE)). Features use [torchvision MobileNetV3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), genuine upstream weights and exported provenance. [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE), FastAPI, OpenCV, React and other dependencies retain their upstream licenses/notices. Weights/private footage are not bundled. This README grants no repository-wide license and does not replace upstream license files.

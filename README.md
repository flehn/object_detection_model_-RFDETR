# Object Tracker

Football player detection, tracking, and team classification.

Built on the [RFDETR-Soccernet](https://huggingface.co/julianzu9612/RFDETR-Soccernet)
checkpoint, with on-field filtering (so crowd / bench detections don't pollute
the tracks) and unsupervised team assignment from jersey colour clusters.

## Components

```
.
├── tracker_core.py    Detection → tracking → on-field gate → team classifier
├── onnx_detector.py   ONNX Runtime inference (drop-in for RFDETRLarge.predict)
├── api.py             FastAPI service: POST /track → SSE progress + JSON result
├── infer.py           CLI: render an annotated video + per-frame counts CSV
├── export_onnx.py     Convert the PyTorch checkpoint to ONNX
├── debug_single_step.py
│                      Frame-by-frame debug tool (jersey samples, k-means snapshots)
├── Dockerfile         Backend container (CUDA libs included for L4 on Cloud Run)
├── cloudbuild.yaml    Backend Cloud Build pipeline
└── frontend/          React + Vite SPA that streams from /track and renders boxes
```

## Quickstart

### 1. Install

```bash
uv sync
```

### 2. Get the ONNX model

Either export it from the upstream checkpoint:

```bash
uv run python export_onnx.py --resolution 384
# writes weights/inference_model.onnx
```

…or wire up your own DVC remote and `dvc pull` (see [DVC setup](#dvc-setup)).

### 3. Run the CLI on a video

```bash
uv run python infer.py path/to/video.mp4 --backend onnx
# writes path/to/video_tracked.mp4 + path/to/video_tracked.csv
```

PyTorch backend (downloads the checkpoint from Hugging Face on first run):

```bash
uv run python infer.py path/to/video.mp4 --backend pytorch
```

### 4. Run the API locally

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8080
```

Then `POST /track` with a `video` form field. The response is an SSE stream of
`progress` events followed by a single `result` event containing per-frame
detections, team assignments, and counts.

### 5. Run the frontend

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8080 npm run dev
```

## Deploy

The repo ships Cloud Build configs that build, push, and deploy both services
to Cloud Run (backend on an L4 GPU, frontend on the smallest instance).

See the headers of [`cloudbuild.yaml`](cloudbuild.yaml) and
[`frontend/cloudbuild.yaml`](frontend/cloudbuild.yaml) for the one-time IAM /
Artifact Registry setup, then:

```bash
gcloud builds submit . --config=cloudbuild.yaml --project=<YOUR_PROJECT_ID>

BACKEND_URL=$(gcloud run services describe object-tracker \
  --region=europe-west4 --format="value(status.url)" \
  --project=<YOUR_PROJECT_ID>)

gcloud builds submit . \
  --config=frontend/cloudbuild.yaml \
  --substitutions=_API_BASE_URL=$BACKEND_URL \
  --project=<YOUR_PROJECT_ID>
```

## DVC setup

The ONNX weights are tracked with DVC and pulled from a GCS bucket during
Cloud Build. To wire up your own remote:

```bash
# .dvc/config has placeholder values that are committed; override with your
# real bucket in .dvc/config.local (gitignored):
cat > .dvc/config.local <<EOF
[core]
    remote = gcs_remote
['remote "gcs_remote"']
    url = gs://<YOUR_WEIGHTS_BUCKET>
    projectname = <YOUR_PROJECT_ID>
EOF

dvc pull weights
```

## Configuration

Backend env vars:

| Variable            | Default                          | Notes                                     |
|---------------------|----------------------------------|-------------------------------------------|
| `ONNX_MODEL_PATH`   | `weights/inference_model.onnx`   | Path to the ONNX weights inside container |
| `ORT_INTRA_THREADS` | unset (ORT picks)                | Intra-op thread count for onnxruntime     |
| `CORS_ORIGINS`      | `*`                              | Comma-separated origin allowlist          |

Frontend build args:

| Variable             | Default                  | Notes                       |
|----------------------|--------------------------|-----------------------------|
| `VITE_API_BASE_URL`  | `http://localhost:8080`  | Backend URL baked in at build |

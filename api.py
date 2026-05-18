"""FastAPI application — single POST /track endpoint.

Accepts a video file upload, runs ONNX-backed detection + tracking, and
returns per-frame bounding boxes with team assignments and counts as JSON.
The frontend is responsible for rendering overlays on top of the original video.

Environment variables
---------------------
ONNX_MODEL_PATH   Path to inference_model.onnx  (default: weights/inference_model.onnx)
ORT_INTRA_THREADS Number of intra-op threads for ONNX Runtime  (default: unset → ORT picks)
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import iterate_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from onnx_detector import OnnxDetector
from tracker_core import TRACKERS, track_video

# ── configuration ──────────────────────────────────────────────────────────────
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "weights/inference_model.onnx")
_ort_threads_env = os.getenv("ORT_INTRA_THREADS")
ORT_INTRA_THREADS: int | None = int(_ort_threads_env) if _ort_threads_env else None

# Comma-separated list of origins, or "*" for any. Set in Cloud Run to the
# frontend's URL (e.g. https://object-tracker-frontend-xxx.run.app).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# Module-level model — loaded once at startup, reused across requests.
_model: OnnxDetector | None = None


# ── lifespan (startup / shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _model
    model_path = Path(ONNX_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"ONNX model not found at {model_path.resolve()}")
    print(f"Loading ONNX model from {model_path}…", flush=True)
    _model = OnnxDetector(str(model_path), intra_op_threads=ORT_INTRA_THREADS)
    print(f"Model ready. ORT providers: {_model.providers}", flush=True)
    yield
    _model = None


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Object Tracker API",
    description=(
        "Football player detection and tracking. "
        "Returns per-frame bounding boxes, team assignments, and counts."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post(
    "/track",
    summary="Track players in a video (streaming)",
    response_description=(
        "Server-Sent Events stream. Yields `progress` events as the video is "
        "processed and a single final `result` event with all detections."
    ),
)
async def track(
    video: UploadFile = File(..., description="Video file (mp4, avi, …)"),
    confidence: float = Query(0.5, ge=0.01, le=1.0, description="Detection confidence threshold"),
    tracker: str = Query("bytetrack", description=f"Tracking algorithm: {list(TRACKERS)}"),
    detect_every: int = Query(
        2,
        ge=1,
        le=10,
        description=(
            "Run detection every Nth frame; skipped frames are not in the output. "
            "1 = every frame (most accurate, slowest). 2 = every other frame "
            "(~2x faster, fine for player tracking)."
        ),
    ),
) -> StreamingResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — service unavailable")

    if tracker not in TRACKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown tracker '{tracker}'. Valid options: {list(TRACKERS)}",
        )

    # Write the upload to a temp file so OpenCV can open it by path.
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    def producer():
        """Sync generator: runs track_video and ensures the temp file is removed."""
        try:
            yield from track_video(
                _model,
                tmp_path,
                confidence=confidence,
                tracker_name=tracker,
                detect_every=detect_every,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def sse_stream() -> AsyncIterator[str]:
        """Wrap the sync generator in SSE framing, running it in a threadpool
        so the event loop stays responsive for other connections (CORS preflights,
        /health probes, etc.)."""
        try:
            async for event in iterate_in_threadpool(producer()):
                yield f"data: {json.dumps(event)}\n\n"
        except (RuntimeError, ValueError) as exc:
            err = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            # Tell any proxies (Cloud Run's load balancer / nginx) NOT to buffer
            # the response — we want each event to reach the client immediately.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

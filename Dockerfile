# ── Object Tracker API ────────────────────────────────────────────────────────
#
# Intentionally avoids PyTorch / rfdetr — inference runs purely through
# ONNX Runtime, keeping the image lean and the container CPU-only friendly.
#
# The model weights (weights/inference_model.onnx) are baked in.
# In Cloud Build the file is downloaded from GCS via `dvc pull` BEFORE
# `docker build` runs, so it is available as a regular COPY source.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System libraries required by OpenCV headless and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV headless runtime
        libglib2.0-0 \
        libgl1 \
        # OpenMP (used by onnxruntime for parallelism)
        libgomp1 \
        # C compiler needed by some tracker packages (lap, filterpy)
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# onnxruntime-gpu >=1.19 needs CUDA 12 + cuDNN 9 at runtime, but the wheel
# only ships the provider .so — not the CUDA/cuDNN libs themselves. Cloud
# Run supplies the NVIDIA driver; we pull the full userspace lib set from
# PyPI (same set PyTorch ships) and register each wheel's lib dir with
# ldconfig so the dynamic loader finds libcublasLt.so.12, libcudnn.so.9,
# libcurand.so.10, libcufft.so.11, etc. at session-creation time.
RUN pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9" \
    "opencv-python-headless>=4.9" \
    "onnxruntime-gpu>=1.19" \
    "nvidia-cuda-runtime-cu12" \
    "nvidia-cuda-nvrtc-cu12" \
    "nvidia-cublas-cu12" \
    "nvidia-cudnn-cu12>=9,<10" \
    "nvidia-curand-cu12" \
    "nvidia-cufft-cu12" \
    "nvidia-cusparse-cu12" \
    "nvidia-cusolver-cu12" \
    "nvidia-nvjitlink-cu12" \
    "supervision>=0.21" \
    "trackers>=2.3.0" \
    "numpy>=1.26" \
    "tqdm>=4.67" \
 && find /usr/local/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib \
      > /etc/ld.so.conf.d/nvidia-cu12.conf \
 && ldconfig

# ── model weights (baked in, downloaded by Cloud Build before docker build) ───
COPY weights/inference_model.onnx ./weights/inference_model.onnx

# ── application source ────────────────────────────────────────────────────────
COPY api.py tracker_core.py onnx_detector.py ./

# Cloud Run injects PORT; uvicorn should listen on 8080 by default.
ENV ONNX_MODEL_PATH=weights/inference_model.onnx \
    ORT_INTRA_THREADS=4

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

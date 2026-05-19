"""ONNX Runtime backed RFDETR detector, drop-in replacement for ``RFDETRLarge.predict``.

Mirrors the preprocessing (resize + ImageNet normalize) and post-processing
(sigmoid + top-K over flattened query×class probs + cxcywh→xyxy scaling) that
:class:`rfdetr.detr.RFDETR` applies around the underlying graph, so callers can
use it interchangeably without touching the rest of the pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

# rfdetr's PostProcess default; matches RFDETRLarge inference behaviour.
NUM_SELECT = 300
# ImageNet normalization, identical to rfdetr/detr.py:373.
_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _default_providers() -> list[str]:
    """Pick a safe default execution provider list.

    CUDA when available (Linux GPU runtimes, including future GCP GPU pools); otherwise CPU.
    CoreML is intentionally *not* auto-selected — it fails to compile this DETR graph in
    some Mac sandboxes and the diagnostic is opaque. Opt in via ``providers=`` if needed.
    """
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxDetector:
    """RFDETR inference via onnxruntime. Exposes the same ``predict`` API as the torch model."""

    def __init__(
        self,
        onnx_path: str,
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads is not None:
            sess_options.intra_op_num_threads = intra_op_threads
        self.providers = providers or _default_providers()
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=self.providers)

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        _, _, h, w = input_meta.shape
        if not isinstance(h, int) or not isinstance(w, int):
            raise RuntimeError(
                f"Expected static H,W in ONNX input shape, got {input_meta.shape}. "
                "Re-export without dynamic spatial axes."
            )
        self.input_h, self.input_w = h, w

        out_names = {o.name for o in self.session.get_outputs()}
        missing = {"dets", "labels"} - out_names
        if missing:
            raise RuntimeError(f"ONNX model missing expected outputs {missing}; found {out_names}")

    def predict(self, image_rgb: np.ndarray, threshold: float = 0.5) -> sv.Detections:
        """Run detection on a single HxWx3 RGB uint8 image and return supervision Detections."""
        orig_h, orig_w = image_rgb.shape[:2]
        x = self._preprocess(image_rgb)
        dets, logits = self.session.run(["dets", "labels"], {self.input_name: x})
        return self._postprocess(dets[0], logits[0], orig_w, orig_h, threshold)

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        # rfdetr's PyTorch predict path uses torchvision.F.resize on a [0,1]
        # tensor with antialias=True (its tensor-input default since
        # torchvision 0.17). cv2.INTER_LINEAR does not antialias and can shift
        # class confidence near the decision boundary on heavy downsamples;
        # cv2.INTER_AREA averages over the source area mapped to each output
        # pixel and is a close functional match for the antialiased bilinear
        # used by torchvision. Use it whenever we're downsampling.
        orig_h, orig_w = image_rgb.shape[:2]
        downsampling = self.input_w < orig_w or self.input_h < orig_h
        interp = cv2.INTER_AREA if downsampling else cv2.INTER_LINEAR
        resized = cv2.resize(image_rgb, (self.input_w, self.input_h), interpolation=interp)
        x = resized.astype(np.float32) * (1.0 / 255.0)
        x = np.transpose(x, (2, 0, 1))[None, ...]
        x = (x - _MEANS) / _STDS
        return np.ascontiguousarray(x, dtype=np.float32)

    @staticmethod
    def _postprocess(
        dets: np.ndarray,
        logits: np.ndarray,
        orig_w: int,
        orig_h: int,
        threshold: float,
    ) -> sv.Detections:
        # dets: (N, 4) cxcywh in [0,1]; logits: (N, C)
        probs = 1.0 / (1.0 + np.exp(-logits))
        flat = probs.reshape(-1)
        k = min(NUM_SELECT, flat.size)

        # Partial sort: argpartition pulls the top-k unordered, then we sort that slice.
        topk_unsorted = np.argpartition(-flat, k - 1)[:k]
        topk_scores_unsorted = flat[topk_unsorted]
        order = np.argsort(-topk_scores_unsorted)
        topk_idx = topk_unsorted[order]
        scores = topk_scores_unsorted[order]

        num_classes = logits.shape[1]
        query_idx = topk_idx // num_classes
        class_idx = (topk_idx % num_classes).astype(int)

        boxes = dets[query_idx]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack(
            [
                (cx - 0.5 * w) * orig_w,
                (cy - 0.5 * h) * orig_h,
                (cx + 0.5 * w) * orig_w,
                (cy + 0.5 * h) * orig_h,
            ],
            axis=1,
        )

        keep = scores > threshold
        return sv.Detections(
            xyxy=xyxy[keep].astype(np.float32),
            confidence=scores[keep].astype(np.float32),
            class_id=class_idx[keep],
        )

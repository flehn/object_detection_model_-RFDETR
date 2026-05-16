#!/usr/bin/env python3
"""Export the RFDETR-Soccernet checkpoint to ONNX for deployment.

Outputs ``weights/<basename>.onnx`` (plus the rfdetr export sidecar files) that the
:mod:`onnx_detector` module can load via onnxruntime. By default we keep the model's
native square resolution (704 for ``RFDETRLarge``); pass ``--resolution`` to shrink
the input for faster CPU inference on Cloud Run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from rfdetr.detr import RFDETRLarge

MODEL_REPO = "julianzu9612/RFDETR-Soccernet"
CHECKPOINT_FILE = "weights/checkpoint_best_regular.pth"
DEFAULT_OUTPUT_DIR = "weights"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RFDETR-Soccernet to ONNX")
    parser.add_argument(
        "--checkpoint",
        help="Local checkpoint path. Downloads from HuggingFace if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX file (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Square input resolution in pixels. Must be divisible by patch_size * num_windows "
            "(32 for RFDETRLarge). Defaults to the model's native resolution (704)."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch dimension (default: static batch_size=1)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)
    print(f"Loading checkpoint: {checkpoint_path}")

    # Export runs on CPU regardless of device; we set "cpu" to skip MPS/CUDA placement.
    kwargs = {"pretrain_weights": checkpoint_path, "num_classes": 3, "device": "cpu"}
    if args.resolution is not None:
        kwargs["resolution"] = args.resolution
    model = RFDETRLarge(**kwargs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX to {output_dir.resolve()} (opset={args.opset}, dynamic_batch={args.dynamic_batch})...")
    model.export(
        output_dir=str(output_dir),
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
    )
    print("Done.")


if __name__ == "__main__":
    main()

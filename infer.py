#!/usr/bin/env python3
"""Football player detection, tracking, and team classification using RFDETR-Soccernet.

Thin video-rendering shell around the shared pipeline in tracker_core.py — the
same field gate, tracker filter, and team classifier that the API uses, so CLI
output matches what the API produces frame-for-frame.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import supervision as sv
import torch
from huggingface_hub import hf_hub_download
from rfdetr.detr import RFDETRLarge
from tqdm import tqdm

from onnx_detector import OnnxDetector
from tracker_core import (
    BALL_CID,
    PLAYER_CID,
    REFEREE_CID,
    TRACKERS,
    FieldColorEstimator,
    TeamClassifier,
    _filter_on_field,
    _filter_tracked,
)

DEFAULT_ONNX_PATH = "weights/inference_model.onnx"


class Detector(Protocol):
    def predict(self, image_rgb: np.ndarray, threshold: float = ...) -> sv.Detections: ...

MODEL_REPO = "julianzu9612/RFDETR-Soccernet"
CHECKPOINT_FILE = "weights/checkpoint_best_regular.pth"

# Display palette indices for the renderer. Slots 0/1 intentionally coincide
# with the team labels returned by TeamClassifier.team_of() (0 = Team A,
# 1 = Team B) so we can assign display_cids[i] = team directly.
TEAM_A, TEAM_B, REFEREE, BALL, UNASSIGNED = 0, 1, 2, 3, 4
DISPLAY_PALETTE = sv.ColorPalette.from_hex(
    [
        "#e74c3c",  # Team A — red
        "#3498db",  # Team B — blue
        "#f1c40f",  # referee — yellow
        "#ecf0f1",  # ball — white
        "#7f8c8d",  # player not yet assigned — grey
    ]
)


def get_checkpoint_path() -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILE)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str | None, device: str) -> RFDETRLarge:
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path()
    return RFDETRLarge(pretrain_weights=checkpoint_path, num_classes=3, device=device)


def _build_display(
    detections: sv.Detections, team_classifier: TeamClassifier
) -> tuple[sv.Detections, list[str], dict[int, set[int]]]:
    display_cids = np.empty(len(detections), dtype=int)
    labels: list[str] = []
    seen: dict[int, set[int]] = {TEAM_A: set(), TEAM_B: set(), REFEREE: set(), BALL: set()}
    for i, (cid, tid) in enumerate(zip(detections.class_id, detections.tracker_id)):
        tid_int = int(tid)
        if cid == PLAYER_CID:
            team = team_classifier.team_of(tid_int)
            if team is None:
                display_cids[i] = UNASSIGNED
                labels.append(f"#{tid_int} ?")
            else:
                display_cids[i] = team
                seen[team].add(tid_int)
                labels.append(f"#{tid_int} T{'AB'[team]}")
        elif cid == REFEREE_CID:
            display_cids[i] = REFEREE
            seen[REFEREE].add(tid_int)
            labels.append(f"#{tid_int} ref")
        elif cid == BALL_CID:
            display_cids[i] = BALL
            seen[BALL].add(tid_int)
            labels.append(f"#{tid_int} ball")
        else:
            display_cids[i] = UNASSIGNED
            labels.append(f"#{tid_int}")
    display = sv.Detections(
        xyxy=detections.xyxy,
        class_id=display_cids,
        confidence=detections.confidence,
        tracker_id=detections.tracker_id,
    )
    return display, labels, seen


def _draw_overlay(frame: np.ndarray, visible: dict[str, int], unique: dict[str, int]) -> None:
    lines = [
        f"Team A   {visible['A']:2d}  (seen {unique['A']})",
        f"Team B   {visible['B']:2d}  (seen {unique['B']})",
        f"Unknown  {visible['U']:2d}",
        f"Referee  {visible['R']:2d}  (seen {unique['R']})",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def process_video(
    model: Detector,
    video_path: str,
    output_path: str,
    *,
    confidence: float = 0.5,
    tracker_name: str = "bytetrack",
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracker = TRACKERS[tracker_name]()
    field_color = FieldColorEstimator()
    team_classifier = TeamClassifier(field_color=field_color)
    box_annotator = sv.BoxAnnotator(color=DISPLAY_PALETTE, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=DISPLAY_PALETTE, color_lookup=sv.ColorLookup.CLASS)

    seen_a: set[int] = set()
    seen_b: set[int] = set()
    seen_ref: set[int] = set()
    frame_log: list[tuple[int, float, int, int, int, int]] = []

    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            field_color.maybe_update(frame)
            detections = model.predict(rgb, threshold=confidence)
            detections = tracker.update(detections)
            detections = _filter_tracked(detections)
            detections = _filter_on_field(detections, frame, field_color)

            visible = {"A": 0, "B": 0, "U": 0, "R": 0}
            annotated = frame
            if len(detections) > 0:
                players = detections[detections.class_id == PLAYER_CID]
                if len(players) > 0 and players.tracker_id is not None:
                    team_classifier.update(frame, players)
                display, labels, seen = _build_display(detections, team_classifier)
                seen_a |= seen[TEAM_A]
                seen_b |= seen[TEAM_B]
                seen_ref |= seen[REFEREE]
                visible["A"] = int((display.class_id == TEAM_A).sum())
                visible["B"] = int((display.class_id == TEAM_B).sum())
                visible["U"] = int((display.class_id == UNASSIGNED).sum())
                visible["R"] = int((display.class_id == REFEREE).sum())
                annotated = box_annotator.annotate(annotated, display)
                annotated = label_annotator.annotate(annotated, display, labels=labels)

            frame_log.append(
                (
                    frame_idx,
                    frame_idx / fps if fps else 0.0,
                    visible["A"],
                    visible["B"],
                    visible["U"],
                    visible["R"],
                )
            )

            _draw_overlay(
                annotated,
                visible=visible,
                unique={"A": len(seen_a), "B": len(seen_b), "R": len(seen_ref)},
            )

            out.write(annotated)
            pbar.update(1)
            frame_idx += 1

    cap.release()
    out.release()

    counts_path = Path(output_path).with_suffix(".csv")
    with counts_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "time_s", "team_a_visible", "team_b_visible", "unassigned_visible", "referee_visible"]
        )
        writer.writerows(frame_log)

    peak_a = max((row[2] for row in frame_log), default=0)
    peak_b = max((row[3] for row in frame_log), default=0)
    print(f"Output saved to {output_path}")
    print(f"Per-frame counts saved to {counts_path}")
    print(
        "Unique tracker IDs seen — "
        f"Team A: {len(seen_a)} | Team B: {len(seen_b)} | Referees: {len(seen_ref)}"
    )
    print(f"Peak simultaneous visible — Team A: {peak_a} | Team B: {peak_b}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect, track, and team-classify football players using RFDETR-Soccernet"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "-o", "--output", help="Output video path (default: <input_path>_tracked.mp4 next to input)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--tracker",
        choices=list(TRACKERS),
        default="bytetrack",
        help="Tracking algorithm (default: bytetrack)",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx"],
        default="pytorch",
        help="Detection backend (default: pytorch)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='PyTorch device: "auto" (default), "cuda", "cuda:N", "mps", or "cpu" (ignored for onnx)',
    )
    parser.add_argument(
        "--checkpoint",
        help="PyTorch checkpoint path (downloads from HuggingFace if omitted; ignored for onnx)",
    )
    parser.add_argument(
        "--onnx-path",
        default=DEFAULT_ONNX_PATH,
        help=f"ONNX model path (default: {DEFAULT_ONNX_PATH})",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output = args.output or str(video_path.with_name(video_path.stem + "_tracked.mp4"))

    model: Detector
    if args.backend == "onnx":
        print(f"Loading ONNX model from {args.onnx_path}...")
        model = OnnxDetector(args.onnx_path)
        print(f"ORT providers: {model.providers}")
    else:
        device = resolve_device(args.device)
        source = args.checkpoint if args.checkpoint else MODEL_REPO
        print(f"Loading PyTorch model from {source} onto {device}...")
        model = load_model(args.checkpoint, device)

    process_video(
        model,
        args.video,
        output,
        confidence=args.confidence,
        tracker_name=args.tracker,
    )


if __name__ == "__main__":
    main()

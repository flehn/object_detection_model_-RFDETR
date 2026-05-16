#!/usr/bin/env python3
"""Football player detection, tracking, and team classification using RFDETR-Soccernet."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from huggingface_hub import hf_hub_download
from rfdetr.detr import RFDETRLarge
from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker
from tqdm import tqdm

MODEL_REPO = "julianzu9612/RFDETR-Soccernet"
CHECKPOINT_FILE = "weights/checkpoint_best_regular.pth"
# Checkpoint was trained with 4 classes (ball/player/referee/goalkeeper) but the newer
# rfdetr loader treats the 4-dim head as 3 foreground + 1 background, so only 3 are active.
CLASS_NAMES = ["ball", "player", "referee"]
BALL_CID = CLASS_NAMES.index("ball")
PLAYER_CID = CLASS_NAMES.index("player")
REFEREE_CID = CLASS_NAMES.index("referee")

TRACKERS = {
    "bytetrack": ByteTrackTracker,
    "sort": SORTTracker,
    "ocsort": OCSORTTracker,
}

# Display class IDs used to drive box/label/trace colors. Distinct from detector class IDs.
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


class TeamClassifier:
    """Assign each player tracker_id to one of two teams by clustering jersey colors.

    Per detection we sample the torso strip, mask out green field and very dark pixels,
    and take the median LAB color. Per-track medians are clustered with 2-means on a
    fixed cadence; once a track has a team it keeps it (sticky) until the next refit.
    """

    def __init__(
        self,
        refit_interval: int = 30,
        min_track_samples: int = 3,
        min_fit_tracks: int = 8,
        max_samples_per_track: int = 50,
    ) -> None:
        self._samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._team_by_tid: dict[int, int] = {}
        self._centroids: np.ndarray | None = None
        self._frames_since_fit = 0
        self.refit_interval = refit_interval
        self.min_track_samples = min_track_samples
        self.min_fit_tracks = min_fit_tracks
        self.max_samples_per_track = max_samples_per_track

    @staticmethod
    def _jersey_color(frame_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 8:
            return None
        # Torso strip: skip head (top 15%) and legs (below 55%) to bias toward the kit.
        ty1 = y1 + int(bh * 0.15)
        ty2 = y1 + int(bh * 0.55)
        crop = frame_bgr[ty1:ty2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        green = (H >= 35) & (H <= 85) & (S >= 40) & (V >= 30)
        too_dark = V < 30
        valid = ~(green | too_dark)
        if int(valid.sum()) < 30:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        return np.median(lab[valid], axis=0).astype(np.float32)

    def update(self, frame_bgr: np.ndarray, players: sv.Detections) -> None:
        for tid, xyxy in zip(players.tracker_id, players.xyxy):
            tid = int(tid)
            if len(self._samples[tid]) >= self.max_samples_per_track:
                continue
            color = self._jersey_color(frame_bgr, xyxy)
            if color is not None:
                self._samples[tid].append(color)
        self._frames_since_fit += 1
        if self._frames_since_fit >= self.refit_interval:
            self._frames_since_fit = 0
            self._fit()

    def _fit(self) -> None:
        tids, medians = [], []
        for tid, samples in self._samples.items():
            if len(samples) >= self.min_track_samples:
                tids.append(tid)
                medians.append(np.median(np.stack(samples), axis=0))
        if len(tids) < self.min_fit_tracks:
            return
        X = np.stack(medians).astype(np.float32)
        centroids = self._kmeans2(X)
        # Pin label order: darker kit (lower L*) is always Team A, so labels stay stable across refits.
        if centroids[0, 0] > centroids[1, 0]:
            centroids = centroids[::-1]
        self._centroids = centroids
        d0 = np.linalg.norm(X - centroids[0], axis=1)
        d1 = np.linalg.norm(X - centroids[1], axis=1)
        labels = (d1 < d0).astype(int)
        for tid, label in zip(tids, labels):
            self._team_by_tid[tid] = int(label)

    @staticmethod
    def _kmeans2(X: np.ndarray, n_iter: int = 25) -> np.ndarray:
        rng = np.random.default_rng(0)
        i = int(rng.integers(0, len(X)))
        j = int(np.argmax(np.linalg.norm(X - X[i], axis=1)))
        c = np.stack([X[i], X[j]]).astype(np.float32)
        for _ in range(n_iter):
            d0 = np.linalg.norm(X - c[0], axis=1)
            d1 = np.linalg.norm(X - c[1], axis=1)
            labels = (d1 < d0).astype(int)
            new_c = c.copy()
            for k in (0, 1):
                mask = labels == k
                if mask.any():
                    new_c[k] = X[mask].mean(axis=0)
            if np.allclose(new_c, c):
                break
            c = new_c
        return c

    def team_of(self, tid: int) -> int | None:
        return self._team_by_tid.get(int(tid))


def _filter_tracked(detections: sv.Detections) -> sv.Detections:
    """Drop detections whose tracker_id is None (unconfirmed tracks)."""
    tids = detections.tracker_id
    if tids is None or len(detections) == 0:
        return detections[:0]
    valid = np.array([t is not None for t in tids], dtype=bool)
    return detections[valid]


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
    model: RFDETRLarge,
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
    team_classifier = TeamClassifier()
    box_annotator = sv.BoxAnnotator(color=DISPLAY_PALETTE, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=DISPLAY_PALETTE, color_lookup=sv.ColorLookup.CLASS)
    trace_annotator = sv.TraceAnnotator(
        color=DISPLAY_PALETTE, color_lookup=sv.ColorLookup.CLASS, trace_length=60
    )

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
            detections = model.predict(rgb, threshold=confidence)
            detections = tracker.update(detections)
            detections = _filter_tracked(detections)

            visible = {"A": 0, "B": 0, "U": 0, "R": 0}
            annotated = frame
            if len(detections) > 0:
                players = detections[detections.class_id == PLAYER_CID]
                team_classifier.update(frame, players)
                display, labels, seen = _build_display(detections, team_classifier)
                seen_a |= seen[TEAM_A]
                seen_b |= seen[TEAM_B]
                seen_ref |= seen[REFEREE]
                visible["A"] = int((display.class_id == TEAM_A).sum())
                visible["B"] = int((display.class_id == TEAM_B).sum())
                visible["U"] = int((display.class_id == UNASSIGNED).sum())
                visible["R"] = int((display.class_id == REFEREE).sum())
                annotated = trace_annotator.annotate(annotated, display)
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
        "--device",
        default="auto",
        help='Inference device: "auto" (default), "cuda", "cuda:N", "mps", or "cpu"',
    )
    parser.add_argument(
        "--checkpoint", help="Local model checkpoint path (downloads from HuggingFace if omitted)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output = args.output or str(video_path.with_name(video_path.stem + "_tracked.mp4"))

    device = resolve_device(args.device)
    source = args.checkpoint if args.checkpoint else MODEL_REPO
    print(f"Loading model from {source} onto {device}...")
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

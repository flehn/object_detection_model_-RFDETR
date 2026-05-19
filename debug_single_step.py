#!/usr/bin/env python3
"""Step through a video frame by frame for debugging detection + team assignment.

For each frame this script runs the same detect → track → field-gate → team-classify
pipeline as the API, then for every surviving detection it prints the bounding box,
class, tracker id, current team assignment, and (for players) the sampled jersey
LAB colour that feeds the k-means clusterer. Annotated screenshots are written to
disk and the per-detection samples are exported to JSONL so the team-assignment
clustering can be iterated on offline without re-running detection.

Output layout (under --out-dir):
  frame_NNNNNN.jpg     annotated frame (bbox in team colour + jersey swatch)
  detections.jsonl     one JSON record per detection per processed frame
  fits.jsonl           one record per k-means refit (centroids + per-track assignment)

Example:
  uv run python debug_single_step.py sample.mp4 --end 200 --out-dir debug_run
  uv run python debug_single_step.py sample.mp4 --interactive --start 100 --end 110
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from onnx_detector import OnnxDetector
from tracker_core import (
    BALL_CID,
    CLASS_NAMES,
    PLAYER_CID,
    REFEREE_CID,
    TEAM_A,
    TEAM_B,
    TRACKERS,
    FieldColorEstimator,
    TeamClassifier,
    _filter_on_field,
    _filter_tracked,
)

DEFAULT_ONNX_PATH = "weights/inference_model.onnx"

# BGR display colours matching infer.py's palette.
COLOR_TEAM_A = (60, 76, 231)        # red
COLOR_TEAM_B = (219, 152, 52)       # blue
COLOR_REFEREE = (15, 196, 241)      # yellow
COLOR_BALL = (241, 240, 236)        # white
COLOR_UNASSIGNED = (141, 140, 127)  # grey


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    """Convert a single LAB sample (as produced by TeamClassifier) back to BGR."""
    px = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def class_color(cid: int, team: int | None) -> tuple[int, int, int]:
    if cid == BALL_CID:
        return COLOR_BALL
    if cid == REFEREE_CID:
        return COLOR_REFEREE
    if cid == PLAYER_CID:
        if team == TEAM_A:
            return COLOR_TEAM_A
        if team == TEAM_B:
            return COLOR_TEAM_B
    return COLOR_UNASSIGNED


def annotate(
    frame: np.ndarray,
    detections: sv.Detections,
    teams: list[int | None],
    jersey_lab: list[np.ndarray | None],
) -> np.ndarray:
    out = frame.copy()
    for i in range(len(detections)):
        x1, y1, x2, y2 = (int(v) for v in detections.xyxy[i])
        cid = int(detections.class_id[i])
        tid = int(detections.tracker_id[i])
        conf = float(detections.confidence[i])
        color = class_color(cid, teams[i])

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if cid == PLAYER_CID:
            team_str = "?" if teams[i] is None else f"T{'AB'[teams[i]]}"
            label = f"#{tid} {team_str} {conf:.2f}"
        elif cid == REFEREE_CID:
            label = f"#{tid} ref {conf:.2f}"
        elif cid == BALL_CID:
            label = f"#{tid} ball {conf:.2f}"
        else:
            label = f"#{tid} {CLASS_NAMES[cid]} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Jersey-colour swatch: shows the LAB value the classifier saw, mapped
        # back to BGR. A grey box at the top of the bbox means no sample.
        jc = jersey_lab[i]
        if jc is not None:
            sw = lab_to_bgr(jc)
            cv2.rectangle(out, (x2 - 16, y1), (x2, y1 + 16), sw, -1)
            cv2.rectangle(out, (x2 - 16, y1), (x2, y1 + 16), (0, 0, 0), 1)
    return out


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24


def parse_times(spec: str, fps: float) -> list[tuple[int, int]]:
    """Parse '13-15,54-55' (seconds) into [(start_frame, end_frame_exclusive), ...]."""
    if fps <= 0:
        raise SystemExit("--times needs a positive fps from the video metadata")
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise SystemExit(f"--times entry '{part}' is not a range like '13-15'")
        a, b = part.split("-", 1)
        a_f = int(float(a) * fps)
        b_f = int(float(b) * fps) + 1
        if b_f <= a_f:
            raise SystemExit(f"--times entry '{part}' is empty after fps conversion")
        out.append((a_f, b_f))
    return out


def snapshot_fit(team_clf: TeamClassifier, frame_idx: int) -> dict | None:
    """Capture the current k-means state for offline analysis. None if no fit yet."""
    if team_clf._centroids is None:
        return None
    return {
        "frame": frame_idx,
        "centroids_lab": team_clf._centroids.tolist(),
        "assignments": {str(tid): team for tid, team in team_clf._team_by_tid.items()},
        "n_samples_per_track": {str(tid): len(s) for tid, s in team_clf._samples.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="input video path")
    ap.add_argument("--onnx", default=DEFAULT_ONNX_PATH, help=f"ONNX model path (default: {DEFAULT_ONNX_PATH})")
    ap.add_argument("-o", "--out-dir", default="debug_run", help="output directory (default: debug_run)")
    ap.add_argument("--start", type=int, default=0, help="start frame index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end frame index (exclusive, default: until EOF)")
    ap.add_argument("--times", default=None,
                    help="comma-separated time ranges in seconds, e.g. '13-15,54-55'. "
                         "Earlier frames are still processed (so the team classifier warms up), "
                         "but console/JSONL/screenshot output is only written for frames inside "
                         "one of these ranges. Implies --end = end of last range unless --end is set.")
    ap.add_argument("--save-every", type=int, default=1,
                    help="save annotated frame every Nth processed frame (default: 1 = save all)")
    ap.add_argument("--confidence", type=float, default=0.5, help="detection confidence threshold")
    ap.add_argument("--tracker", default="bytetrack", choices=list(TRACKERS), help="tracker (default: bytetrack)")
    ap.add_argument("--interactive", action="store_true",
                    help="pause between frames — ENTER advances, 'q' + ENTER quits")
    ap.add_argument("--no-save-frames", action="store_true", help="skip writing annotated frame images")
    args = ap.parse_args()

    if args.save_every < 1:
        ap.error("--save-every must be >= 1")
    if args.end is not None and args.end <= args.start:
        ap.error("--end must be > --start")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "detections.jsonl"
    fit_path = out_dir / "fits.jsonl"

    print(f"Loading ONNX model from {args.onnx}...")
    model = OnnxDetector(args.onnx)
    print(f"  providers: {model.providers}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video} — {total_frames} frames {width}x{height} @ {fps:.2f} fps")

    save_ranges: list[tuple[int, int]] | None = None
    if args.times:
        save_ranges = parse_times(args.times, fps)
        if args.end is None:
            args.end = max(b for _, b in save_ranges)
        print(f"Save ranges (frames): {save_ranges} — output only inside these; "
              f"earlier frames still processed for warmup.")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    tracker = TRACKERS[args.tracker]()
    field_color = FieldColorEstimator()
    team_clf = TeamClassifier(field_color=field_color)
    last_fit_centroids: np.ndarray | None = None

    frame_idx = args.start
    processed = 0

    with det_path.open("w") as det_out, fit_path.open("w") as fit_out:
        while True:
            if args.end is not None and frame_idx >= args.end:
                break
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            field_color.maybe_update(frame)
            detections = model.predict(rgb, threshold=args.confidence)
            detections = tracker.update(detections)
            detections = _filter_tracked(detections)
            detections = _filter_on_field(detections, frame, field_color)

            if len(detections) > 0:
                players = detections[detections.class_id == PLAYER_CID]
                if len(players) > 0 and players.tracker_id is not None:
                    team_clf.update(frame, players)

            # Detect that a refit just landed (centroids object identity changed).
            if team_clf._centroids is not None and team_clf._centroids is not last_fit_centroids:
                snap = snapshot_fit(team_clf, frame_idx)
                if snap is not None:
                    fit_out.write(json.dumps(snap) + "\n")
                    fit_out.flush()
                last_fit_centroids = team_clf._centroids

            in_save_range = save_ranges is None or any(
                a <= frame_idx < b for a, b in save_ranges
            )

            if not in_save_range:
                # Warmup frame — pipeline state updated above, skip all output.
                processed += 1
                frame_idx += 1
                continue

            # Sample jersey colour + look up team for each detection. Calls into
            # TeamClassifier._jersey_color directly so the LAB sample matches
            # exactly what the production clusterer sees.
            teams: list[int | None] = []
            jersey_lab: list[np.ndarray | None] = []
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                tid = int(detections.tracker_id[i])
                if cid == PLAYER_CID:
                    lab = team_clf._jersey_color(frame, detections.xyxy[i])
                    teams.append(team_clf.team_of(tid))
                    jersey_lab.append(lab)
                else:
                    teams.append(None)
                    jersey_lab.append(None)

            # ── console ──
            print(f"\n── frame {frame_idx} ({len(detections)} det) "
                  f"field_hue=[{field_color.hue_lo},{field_color.hue_hi}] ──")
            counts = {"A": 0, "B": 0, "?": 0, "ref": 0, "ball": 0}
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                tid = int(detections.tracker_id[i])
                bbox = [round(float(v), 1) for v in detections.xyxy[i]]
                conf = float(detections.confidence[i])
                cls = CLASS_NAMES[cid]
                line = f"  #{tid:>3} {cls:8} conf={conf:.2f} bbox={bbox}"
                if cid == PLAYER_CID:
                    team = teams[i]
                    team_str = "?" if team is None else f"T{'AB'[team]}"
                    counts["?" if team is None else "AB"[team]] += 1
                    jc = jersey_lab[i]
                    if jc is not None:
                        bgr = lab_to_bgr(jc)
                        line += (f"  team={team_str}"
                                 f"  jersey_LAB=[{jc[0]:5.1f},{jc[1]:5.1f},{jc[2]:5.1f}]"
                                 f"  bgr={bgr}")
                    else:
                        line += f"  team={team_str}  jersey=<no sample>"
                elif cid == REFEREE_CID:
                    counts["ref"] += 1
                elif cid == BALL_CID:
                    counts["ball"] += 1
                print(line)
            print(f"  counts: TA={counts['A']} TB={counts['B']} ?={counts['?']} "
                  f"ref={counts['ref']} ball={counts['ball']}")
            if team_clf._centroids is not None:
                ca, cb = team_clf._centroids
                print(f"  centroids LAB: A=[{ca[0]:5.1f},{ca[1]:5.1f},{ca[2]:5.1f}] "
                      f"B=[{cb[0]:5.1f},{cb[1]:5.1f},{cb[2]:5.1f}]")

            # ── jsonl ──
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                tid = int(detections.tracker_id[i])
                jc = jersey_lab[i]
                det_out.write(json.dumps({
                    "frame": frame_idx,
                    "track_id": tid,
                    "class": CLASS_NAMES[cid],
                    "bbox": [float(v) for v in detections.xyxy[i]],
                    "confidence": float(detections.confidence[i]),
                    "team": teams[i],
                    "jersey_lab": [float(v) for v in jc] if jc is not None else None,
                }) + "\n")
            det_out.flush()

            # ── screenshot ──
            if not args.no_save_frames and (processed % args.save_every == 0):
                annotated = annotate(frame, detections, teams, jersey_lab)
                hud = [
                    f"frame {frame_idx}/{total_frames or '?'}  det={len(detections)}",
                    f"TA={counts['A']}  TB={counts['B']}  ?={counts['?']}  "
                    f"ref={counts['ref']}  ball={counts['ball']}",
                ]
                draw_hud(annotated, hud)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), annotated)

            processed += 1
            frame_idx += 1

            if args.interactive:
                resp = input("[enter]=next  [q]=quit > ").strip().lower()
                if resp == "q":
                    break

    cap.release()
    print(f"\nDone — processed {processed} frames "
          f"(input idx {args.start}..{frame_idx - 1})")
    print(f"  detections   → {det_path}")
    print(f"  k-means fits → {fit_path}")
    if not args.no_save_frames:
        print(f"  frames       → {out_dir}/frame_*.jpg")


if __name__ == "__main__":
    main()

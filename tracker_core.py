"""Core tracking logic used by the API — zero PyTorch / rfdetr dependency.

All heavy detection is delegated to an OnnxDetector instance; this module
owns only the tracking loop, team classification, and the JSON serialisation
of per-frame results.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Iterator, Protocol

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker

# ── class constants ────────────────────────────────────────────────────────────
CLASS_NAMES = ["ball", "player", "referee"]
BALL_CID = CLASS_NAMES.index("ball")
PLAYER_CID = CLASS_NAMES.index("player")
REFEREE_CID = CLASS_NAMES.index("referee")

TEAM_A, TEAM_B = 0, 1

TRACKERS: dict[str, type] = {
    "bytetrack": ByteTrackTracker,
    "sort": SORTTracker,
    "ocsort": OCSORTTracker,
}


# ── detector protocol (accepts OnnxDetector or any compatible object) ──────────
class Detector(Protocol):
    def predict(self, image_rgb: np.ndarray, threshold: float = ...) -> sv.Detections: ...


# ── field color estimator ──────────────────────────────────────────────────────
class FieldColorEstimator:
    """Learn the dominant turf hue from the lower part of the frame.

    Used for two purposes: (1) gating detections so only on-field tracks survive
    (kills crowd false-positives), and (2) masking field pixels out of jersey
    colour sampling so a team in green jerseys doesn't get filtered to nothing.
    Refreshes on a fixed cadence — the camera may pan or lighting may shift, but
    not frame-by-frame.
    """

    def __init__(self, refresh_every: int = 30) -> None:
        self.refresh_every = refresh_every
        # Force estimation on the very first frame.
        self._frames_since_refresh = refresh_every
        self.hue_lo: int | None = None
        self.hue_hi: int | None = None
        # Topmost image row where field pixels start to dominate. Detections
        # whose feet sit above this line are rejected as off-pitch (stands,
        # bench, technical area). None until the first successful estimate.
        self.field_top_y: int | None = None

    def maybe_update(self, frame_bgr: np.ndarray) -> None:
        self._frames_since_refresh += 1
        if self._frames_since_refresh < self.refresh_every:
            return
        result = self._estimate(frame_bgr)
        if result is not None:
            self.hue_lo, self.hue_hi, self.field_top_y = result
            self._frames_since_refresh = 0
        # If estimation fails (close-up, weird lighting), keep the cached
        # range and try again next frame instead of disabling filtering.

    @staticmethod
    def _estimate(frame_bgr: np.ndarray) -> tuple[int, int, int | None] | None:
        h = frame_bgr.shape[0]
        region = frame_bgr[int(h * 0.4):]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # Ignore unsaturated (lines, white kit, shadow on grass) and very bright pixels.
        valid = (S >= 40) & (V >= 30) & (V <= 230)
        if int(valid.sum()) < 5000:
            return None
        H_valid = H[valid].astype(np.int32)
        # 18 bins of 10° each over [0, 180) — OpenCV hue scale.
        hist = np.bincount(H_valid // 10, minlength=18)
        peak = int(np.argmax(hist))
        # Require the peak to dominate; otherwise the frame is probably a
        # close-up where the field isn't the dominant hue.
        if hist[peak] / hist.sum() < 0.20:
            return None
        lo = max(0, (peak - 2) * 10)
        hi = min(180, (peak + 3) * 10)

        # Locate the field horizon. The pitch is the deep contiguous block
        # of high-field-fraction rows at the bottom of the frame, so we look
        # for the topmost y where a long 100-row window stays ≥80 % field on
        # average — strict enough that isolated green/yellow advertising
        # bands above the pitch don't count.
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H_f, S_f, V_f = hsv_full[..., 0], hsv_full[..., 1], hsv_full[..., 2]
        field_mask = (H_f >= lo) & (H_f <= hi) & (S_f >= 30) & (V_f >= 30)
        row_fraction = field_mask.mean(axis=1)
        window = 100
        field_top_y: int | None = None
        if h >= window:
            smoothed = np.convolve(row_fraction, np.ones(window) / window, mode="valid")
            above = smoothed >= 0.8
            if above.any():
                field_top_y = int(np.argmax(above))
        return lo, hi, field_top_y

    def field_mask(self, hsv_block: np.ndarray) -> np.ndarray | None:
        """Boolean mask of field-coloured pixels. None if not yet initialised."""
        if self.hue_lo is None:
            return None
        H, S, V = hsv_block[..., 0], hsv_block[..., 1], hsv_block[..., 2]
        return (H >= self.hue_lo) & (H <= self.hue_hi) & (S >= 30) & (V >= 30)

    def is_on_field(self, frame_bgr: np.ndarray, xyxy: np.ndarray, min_frac: float = 0.35) -> bool:
        """Check whether the strip just below the bbox is mostly field-coloured."""
        if self.hue_lo is None:
            return True  # Not yet initialised — don't filter.
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy.astype(int)
        # Horizon gate: a detection's feet (y2) must land at or below the
        # learned field horizon. No amount of green near the boundary
        # can put a person in the stands onto the pitch.
        if self.field_top_y is not None and y2 < self.field_top_y:
            return False
        x1, x2 = max(0, x1), min(w, x2)
        bw = x2 - x1
        bh = max(1, y2 - y1)
        if bw < 4:
            return False
        strip_h = max(6, bh // 8)
        sy1 = min(h, y2)
        sy2 = min(h, y2 + strip_h)
        if sy2 - sy1 < 4:
            # bbox sits at the bottom edge of the frame — can't sample below
            # the feet, so don't filter.
            return True
        crop = frame_bgr[sy1:sy2, x1:x2]
        if crop.size == 0:
            return True
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = self.field_mask(hsv)
        if mask is None:
            return True
        return float(mask.mean()) >= min_frac


# ── team classifier ────────────────────────────────────────────────────────────
class TeamClassifier:
    """Assign each player tracker_id to one of two teams by clustering jersey colours.

    Samples the torso strip of each detection, masks out field-coloured and very
    dark pixels (using the learned field hue when a FieldColorEstimator is wired
    in), and takes the median LAB colour. Per-track medians are clustered with
    2-means on a fixed cadence; tracks whose residual is far above the typical
    within-cluster spread are left unassigned rather than forced into TA/TB.
    """

    def __init__(
        self,
        refit_interval: int = 10,
        min_track_samples: int = 3,
        min_fit_tracks: int = 5,
        sample_window: int = 5,
        field_color: FieldColorEstimator | None = None,
        outlier_mad_mult: float = 3.0,
    ) -> None:
        self.sample_window = sample_window
        self._samples: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=sample_window)
        )
        self._team_by_tid: dict[int, int] = {}
        self._centroids: np.ndarray | None = None
        self._frames_since_fit = 0
        self.refit_interval = refit_interval
        self.min_track_samples = min_track_samples
        self.min_fit_tracks = min_fit_tracks
        self.field_color = field_color
        self.outlier_mad_mult = outlier_mad_mult

    def _jersey_color(self, frame_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 6:
            return None
        # Small box centered on the bbox centre instead of the full torso
        # strip — keeps the sample away from background pixels at the bbox
        # edges (neighbouring players, crowd, grass) and from the head/leg
        # regions where colour is unreliable.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w = max(2, int(bw * 0.15))   # 30% of bbox width
        half_h = max(2, int(bh * 0.15))   # 30% of bbox height
        bx1, by1 = max(0, cx - half_w), max(0, cy - half_h)
        bx2, by2 = min(w, cx + half_w), min(h, cy + half_h)
        crop = frame_bgr[by1:by2, bx1:bx2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        too_dark = hsv[..., 2] < 30
        field_pixels = self.field_color.field_mask(hsv) if self.field_color is not None else None
        if field_pixels is not None:
            valid = ~(field_pixels | too_dark)
        else:
            # Bootstrap fallback before the field estimator has converged.
            H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            green = (H >= 35) & (H <= 85) & (S >= 40) & (V >= 30)
            valid = ~(green | too_dark)
        # Smaller sample window → lower pixel floor than the old torso strip.
        if int(valid.sum()) < 10:
            # Box is almost entirely field/dark — most likely a team whose
            # kit clashes with the field. Drop the field mask so we still
            # get *some* signal.
            valid = ~too_dark
            if int(valid.sum()) < 10:
                return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        return np.median(lab[valid], axis=0).astype(np.float32)

    def update(self, frame_bgr: np.ndarray, players: sv.Detections) -> None:
        for tid, xyxy in zip(players.tracker_id, players.xyxy):
            tid = int(tid)
            if tid < 0:
                # Defensive — _filter_tracked should have dropped these already.
                continue
            color = self._jersey_color(frame_bgr, xyxy)
            if color is not None:
                # If the new sample is far enough from the deque's median to
                # cross the inter-cluster gap, the tracker has almost
                # certainly been revived onto a different physical player
                # (scene cut, heavy occlusion, dormant track resurrected by
                # ByteTrack's track buffer). The old samples no longer
                # describe this track — clear them so the next classify
                # uses the fresh sample only.
                if self._centroids is not None and len(self._samples[tid]) > 0:
                    current_median = np.median(np.stack(self._samples[tid]), axis=0)
                    inter_centroid = float(
                        np.linalg.norm(self._centroids[0] - self._centroids[1])
                    )
                    if np.linalg.norm(color - current_median) > 0.5 * inter_centroid:
                        self._samples[tid].clear()
                # deque(maxlen=sample_window) auto-evicts the oldest sample,
                # so we always cluster on the last N frames per track.
                self._samples[tid].append(color)
            # Once centroids exist, classify every frame from the rolling
            # median. Without this a fresh track sits as `?` for up to
            # refit_interval frames, and any track present at fit time but
            # not in the current frame loses its team at the next refit.
            if self._centroids is not None and len(self._samples[tid]) > 0:
                median = np.median(np.stack(self._samples[tid]), axis=0)
                d0 = np.linalg.norm(median - self._centroids[0])
                d1 = np.linalg.norm(median - self._centroids[1])
                self._team_by_tid[tid] = 0 if d0 < d1 else 1
        self._frames_since_fit += 1
        if self._frames_since_fit >= self.refit_interval:
            self._frames_since_fit = 0
            self._fit()

    def _fit(self) -> None:
        tids, medians = [], []
        for tid, samples in self._samples.items():
            if tid < 0:
                continue
            if len(samples) >= self.min_track_samples:
                tids.append(tid)
                medians.append(np.median(np.stack(samples), axis=0))
        if len(tids) < self.min_fit_tracks:
            return
        X = np.stack(medians).astype(np.float32)
        centroids = self._kmeans2(X)
        # Pin label order: darker kit (lower L*) is always Team A for stable labels.
        if centroids[0, 0] > centroids[1, 0]:
            centroids = centroids[::-1]
        self._centroids = centroids
        d0 = np.linalg.norm(X - centroids[0], axis=1)
        d1 = np.linalg.norm(X - centroids[1], axis=1)
        labels = (d1 < d0).astype(int)
        chosen_dist = np.minimum(d0, d1)

        # Outlier rejection: a track whose residual is far above the typical
        # within-cluster spread is probably not actually on either team
        # (referee, goalkeeper in off-colour kit, a crowd detection that
        # slipped past the field gate). Leave such tracks unassigned rather
        # than forcing TA/TB.
        tids_arr = np.array(tids)
        new_assignments: dict[int, int] = {}
        for k in (0, 1):
            mask = labels == k
            if not mask.any():
                continue
            d_k = chosen_dist[mask]
            if len(d_k) < 3:
                # Cluster too small to estimate spread reliably; trust k-means.
                for tid in tids_arr[mask]:
                    new_assignments[int(tid)] = int(k)
                continue
            med = float(np.median(d_k))
            mad = float(np.median(np.abs(d_k - med))) + 1e-6
            threshold = med + self.outlier_mad_mult * mad
            for tid, dist in zip(tids_arr[mask], d_k):
                if dist <= threshold:
                    new_assignments[int(tid)] = int(k)
        # Merge instead of replace: MAD-rejected tracks and tracks not in
        # this fit window keep their existing (nearest-centroid) assignment
        # from update() instead of evaporating until the next refit.
        self._team_by_tid.update(new_assignments)

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


# ── helpers ────────────────────────────────────────────────────────────────────
def _filter_tracked(detections: sv.Detections) -> sv.Detections:
    """Drop detections without a confirmed tracker_id (None or negative).

    Tentative ByteTrack ids (-1) are excluded so they can't pool into a single
    fake "track -1" in the team classifier or appear as flickering boxes in
    the renderer — the same physical player will reappear with a real id
    within a frame or two.
    """
    tids = detections.tracker_id
    if tids is None or len(detections) == 0:
        return detections[:0]
    valid = np.array([t is not None and t >= 0 for t in tids], dtype=bool)
    return detections[valid]


def _filter_on_field(
    detections: sv.Detections,
    frame_bgr: np.ndarray,
    field_color: FieldColorEstimator,
) -> sv.Detections:
    """Drop player detections whose feet aren't planted on field-coloured pixels.

    Ball and referee detections are exempt: the ball legitimately leaves the
    ground, and assistant referees stand on the touchline where the strip
    below their feet is white line / track / advertising rather than grass —
    they'd be filtered out by the field-fraction check despite being on-pitch.
    """
    if len(detections) == 0:
        return detections
    keep = np.ones(len(detections), dtype=bool)
    for i in range(len(detections)):
        cid = int(detections.class_id[i])
        if cid == BALL_CID or cid == REFEREE_CID:
            continue
        if not field_color.is_on_field(frame_bgr, detections.xyxy[i]):
            keep[i] = False
    return detections[keep]


# ── main API entry point ───────────────────────────────────────────────────────
def track_video(
    model: Detector,
    video_path: str,
    *,
    confidence: float = 0.5,
    tracker_name: str = "bytetrack",
    detect_every: int = 2,
    progress_every_frames: int = 30,
) -> Iterator[dict]:
    """Iterator yielding tracking events as the video is processed.

    Two event types are produced:

      ``{"type": "progress",
         "frame": int,              # 1-indexed frame number just processed
         "total": int,              # total frames in the video (0 if unknown)
         "fps_processing": float,   # frames-per-second we're processing at
         "eta_seconds": float|None  # estimated seconds until completion
        }``

      ``{"type": "result",
         "video_info": {...},
         "frames": [...],
         "summary": {...}
        }``  (yielded exactly once, as the final event)

    Progress events are emitted every ``progress_every_frames`` frames. The
    result event matches the dict shape the API used to return synchronously.

    ``detect_every`` controls detection cadence: 1 = every frame (most accurate
    and most expensive); 2 = every other frame (roughly halves compute, fine
    for player tracking since people move slowly relative to typical 25–30 fps
    capture); higher values trade more accuracy for more speed. Frames where
    detection is skipped do not appear in the output's ``frames`` list — the
    ``frame_id`` field tells consumers which input frame each entry came from.
    """
    if tracker_name not in TRACKERS:
        raise ValueError(f"Unknown tracker '{tracker_name}'. Choose from: {list(TRACKERS)}")
    if detect_every < 1:
        raise ValueError(f"detect_every must be >= 1, got {detect_every}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    tracker = TRACKERS[tracker_name]()
    field_color = FieldColorEstimator()
    team_clf = TeamClassifier(field_color=field_color)

    frames_data: list[dict] = []
    seen_team0: set[int] = set()
    seen_team1: set[int] = set()
    seen_ref: set[int] = set()
    frame_idx = 0
    last_progress_at = 0
    start_time = time.monotonic()

    print(
        f"track_video: starting — {total_frames or '?'} frames, "
        f"{width}x{height} @ {fps:.2f} fps, detect_every={detect_every}",
        flush=True,
    )

    try:
        while True:
            is_detect_frame = (frame_idx % detect_every == 0)

            if is_detect_frame:
                ret, frame = cap.read()
            else:
                # Skip decode entirely on non-detect frames — grab() just
                # advances the demuxer, which is dramatically cheaper than
                # a full decode at HD resolutions.
                ret = cap.grab()

            if not ret:
                break

            if not is_detect_frame:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            field_color.maybe_update(frame)
            detections = model.predict(rgb, threshold=confidence)
            detections = tracker.update(detections)
            detections = _filter_tracked(detections)
            detections = _filter_on_field(detections, frame, field_color)

            frame_detections: list[dict] = []

            if len(detections) > 0:
                players = detections[detections.class_id == PLAYER_CID]
                if len(players) > 0 and players.tracker_id is not None:
                    team_clf.update(frame, players)

                for i in range(len(detections)):
                    cid = int(detections.class_id[i])
                    tid = int(detections.tracker_id[i])
                    x1, y1, x2, y2 = (float(v) for v in detections.xyxy[i])
                    conf = float(detections.confidence[i])
                    class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "unknown"

                    team: int | None = None
                    if cid == PLAYER_CID:
                        team = team_clf.team_of(tid)
                        if team == TEAM_A:
                            seen_team0.add(tid)
                        elif team == TEAM_B:
                            seen_team1.add(tid)
                    elif cid == REFEREE_CID:
                        seen_ref.add(tid)

                    frame_detections.append(
                        {
                            "track_id": tid,
                            "class": class_name,
                            "team": team,
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "confidence": round(conf, 3),
                        }
                    )

            counts = {
                "team_0": sum(
                    1 for d in frame_detections if d["class"] == "player" and d["team"] == 0
                ),
                "team_1": sum(
                    1 for d in frame_detections if d["class"] == "player" and d["team"] == 1
                ),
                "unassigned_players": sum(
                    1 for d in frame_detections if d["class"] == "player" and d["team"] is None
                ),
                "referees": sum(1 for d in frame_detections if d["class"] == "referee"),
                "ball": sum(1 for d in frame_detections if d["class"] == "ball"),
            }

            frames_data.append(
                {
                    "frame_id": frame_idx,
                    "timestamp_ms": round((frame_idx / fps) * 1000) if fps else 0,
                    "detections": frame_detections,
                    "counts": counts,
                }
            )

            frame_idx += 1

            # ── progress event ────────────────────────────────────────────
            if frame_idx - last_progress_at >= progress_every_frames:
                elapsed = time.monotonic() - start_time
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_frames - frame_idx) if total_frames else 0
                eta = remaining / fps_proc if fps_proc > 0 and total_frames else None
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                print(
                    f"track_video: frame {frame_idx}/{total_frames or '?'} "
                    f"({pct:.1f}%) — {fps_proc:.1f} fps proc, "
                    f"ETA {eta:.0f}s" if eta is not None else
                    f"track_video: frame {frame_idx} — {fps_proc:.1f} fps proc",
                    flush=True,
                )
                yield {
                    "type": "progress",
                    "frame": frame_idx,
                    "total": total_frames,
                    "fps_processing": round(fps_proc, 2),
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                }
                last_progress_at = frame_idx

    finally:
        cap.release()

    elapsed = time.monotonic() - start_time
    processed_frames = len(frames_data)
    print(
        f"track_video: done — {frame_idx} input frames "
        f"({processed_frames} detected) in {elapsed:.1f}s "
        f"({frame_idx / elapsed if elapsed > 0 else 0:.1f} fps avg)",
        flush=True,
    )

    yield {
        "type": "result",
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "detect_every": detect_every,
            "processed_frames": processed_frames,
        },
        "frames": frames_data,
        "summary": {
            "total_frames": frame_idx,
            "unique_ids": {
                "team_0": len(seen_team0),
                "team_1": len(seen_team1),
                "referees": len(seen_ref),
            },
            "peak_counts": {
                "team_0": max((f["counts"]["team_0"] for f in frames_data), default=0),
                "team_1": max((f["counts"]["team_1"] for f in frames_data), default=0),
            },
        },
    }

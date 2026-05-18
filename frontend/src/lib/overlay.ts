/** Canvas overlay rendering. Boxes come in original-video pixels, the canvas
 *  is sized to the rendered <video> element — so every box gets scaled by
 *  (canvas.width / video.videoWidth) etc. before drawing. */

import type { Detection, FrameData } from "./api";

const TEAM_COLORS = {
  team0: "#ef4444", // red
  team1: "#3b82f6", // blue
  referee: "#facc15", // yellow
  ball: "#f8fafc", // white-ish
  unassigned: "#94a3b8", // grey
} as const;

function colorFor(d: Detection): string {
  if (d.class === "player") {
    if (d.team === 0) return TEAM_COLORS.team0;
    if (d.team === 1) return TEAM_COLORS.team1;
    return TEAM_COLORS.unassigned;
  }
  if (d.class === "referee") return TEAM_COLORS.referee;
  if (d.class === "ball") return TEAM_COLORS.ball;
  return TEAM_COLORS.unassigned;
}

function labelFor(d: Detection): string {
  if (d.class === "player") {
    if (d.team === 0) return `#${d.track_id} TA`;
    if (d.team === 1) return `#${d.track_id} TB`;
    return `#${d.track_id} ?`;
  }
  if (d.class === "referee") return `#${d.track_id} ref`;
  if (d.class === "ball") return "ball";
  return `#${d.track_id}`;
}

/** Binary search for the frame whose timestamp_ms is closest to `targetMs`.
 *  Frames are assumed to be sorted by timestamp_ms ascending (which they are
 *  by construction in the API response). */
export function findClosestFrame(frames: FrameData[], targetMs: number): FrameData | null {
  if (frames.length === 0) return null;

  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].timestamp_ms < targetMs) lo = mid + 1;
    else hi = mid;
  }
  // Compare lo against lo-1 to find the closer of the two.
  if (lo > 0 && Math.abs(frames[lo - 1].timestamp_ms - targetMs) < Math.abs(frames[lo].timestamp_ms - targetMs)) {
    return frames[lo - 1];
  }
  return frames[lo];
}

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  frame: FrameData,
  videoWidth: number,
  videoHeight: number,
  // Set false when compositing on top of an already-painted video frame (export path).
  clear: boolean = true,
): void {
  const { canvas } = ctx;
  const scaleX = canvas.width / videoWidth;
  const scaleY = canvas.height / videoHeight;

  if (clear) ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 480));
  ctx.font = `${Math.max(11, Math.round(canvas.width / 96))}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  for (const det of frame.detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;

    const color = colorFor(det);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // Label background
    const label = labelFor(det);
    const padding = 3;
    const textMetrics = ctx.measureText(label);
    const textHeight = parseInt(ctx.font, 10);
    const labelW = textMetrics.width + padding * 2;
    const labelH = textHeight + padding * 2;
    const labelY = y - labelH > 0 ? y - labelH : y;

    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, labelW, labelH);
    ctx.fillStyle = "#0f1115";
    ctx.fillText(label, x + padding, labelY + padding);
  }
}

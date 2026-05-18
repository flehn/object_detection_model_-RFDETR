/** Types and helpers for talking to the FastAPI backend on Cloud Run. */

export type DetectionClass = "ball" | "player" | "referee" | "unknown";

export interface Detection {
  track_id: number;
  class: DetectionClass;
  team: 0 | 1 | null;
  bbox: [number, number, number, number]; // x1, y1, x2, y2 in original video pixels
  confidence: number;
}

export interface FrameCounts {
  team_0: number;
  team_1: number;
  unassigned_players: number;
  referees: number;
  ball: number;
}

export interface FrameData {
  frame_id: number;
  timestamp_ms: number;
  detections: Detection[];
  counts: FrameCounts;
}

export interface TrackingResult {
  video_info: {
    fps: number;
    width: number;
    height: number;
    total_frames: number;
  };
  frames: FrameData[];
  summary: {
    total_frames: number;
    unique_ids: { team_0: number; team_1: number; referees: number };
    peak_counts: { team_0: number; team_1: number };
  };
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";

export interface ProgressEvent {
  frame: number;
  total: number;
  fps_processing: number;
  eta_seconds: number | null;
}

/** Streams from POST /track. Calls `onProgress` for each progress event and
 *  resolves with the final result when the stream ends.
 *
 *  The endpoint returns text/event-stream with SSE framing (`data: {...}\n\n`).
 *  We parse it manually because EventSource doesn't support POST. */
export async function trackVideo(
  file: File,
  { confidence = 0.5, tracker = "bytetrack" }: { confidence?: number; tracker?: string } = {},
  onProgress?: (event: ProgressEvent) => void,
): Promise<TrackingResult> {
  const form = new FormData();
  form.append("video", file);

  const url = new URL(`${API_BASE}/track`);
  url.searchParams.set("confidence", String(confidence));
  url.searchParams.set("tracker", tracker);

  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  if (!res.body) {
    throw new Error("Response has no body to stream");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: TrackingResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line ("\n\n"). Split, keep the last
    // (possibly partial) chunk in the buffer for the next iteration.
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      // A chunk may contain multiple "data: …" lines; we only have one per event.
      const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice("data: ".length));

      if (payload.type === "progress") {
        onProgress?.({
          frame: payload.frame,
          total: payload.total,
          fps_processing: payload.fps_processing,
          eta_seconds: payload.eta_seconds,
        });
      } else if (payload.type === "result") {
        // Drop the "type" tag — the rest matches TrackingResult.
        const { type: _t, ...rest } = payload;
        result = rest as TrackingResult;
      } else if (payload.type === "error") {
        throw new Error(payload.detail || "Server error during tracking");
      }
    }
  }

  if (!result) {
    throw new Error("Stream ended without a result event");
  }
  return result;
}

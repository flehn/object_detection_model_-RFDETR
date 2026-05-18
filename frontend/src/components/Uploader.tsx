import { useState } from "react";
import { trackVideo, type ProgressEvent, type TrackingResult } from "../lib/api";

interface UploaderProps {
  onResult: (file: File, result: TrackingResult) => void;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || !isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function Uploader({ onResult }: UploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.5);
  const [tracker, setTracker] = useState("bytetrack");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProgress(null);
    try {
      const result = await trackVideo(
        file,
        { confidence, tracker },
        (p) => setProgress(p),
      );
      onResult(file, result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 16,
        padding: 24,
        background: "#1a1d24",
        borderRadius: 12,
        border: "1px solid #2a2d36",
      }}
    >
      <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <span style={{ color: "#94a3b8", fontSize: 13 }}>Video file</span>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          disabled={loading}
        />
      </label>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 6, flex: 1, minWidth: 160 }}>
          <span style={{ color: "#94a3b8", fontSize: 13 }}>
            Confidence: {confidence.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            disabled={loading}
          />
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: 6, flex: 1, minWidth: 160 }}>
          <span style={{ color: "#94a3b8", fontSize: 13 }}>Tracker</span>
          <select
            value={tracker}
            onChange={(e) => setTracker(e.target.value)}
            disabled={loading}
            style={{
              padding: 8,
              borderRadius: 6,
              background: "#0f1115",
              color: "inherit",
              border: "1px solid #2a2d36",
            }}
          >
            <option value="bytetrack">ByteTrack</option>
            <option value="sort">SORT</option>
            <option value="ocsort">OC-SORT</option>
          </select>
        </label>
      </div>

      <button onClick={handleSubmit} disabled={!file || loading}>
        {loading ? "Processing…" : "Run tracking"}
      </button>

      {loading && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            padding: 12,
            background: "#0f1115",
            borderRadius: 8,
            border: "1px solid #2a2d36",
          }}
        >
          {progress ? (
            <>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 13,
                  color: "#cbd5e1",
                }}
              >
                <span>
                  Frame {progress.frame.toLocaleString()}
                  {progress.total > 0 && ` / ${progress.total.toLocaleString()}`}
                  {" · "}
                  {progress.fps_processing.toFixed(1)} fps
                </span>
                <span style={{ color: "#94a3b8" }}>
                  ~{formatEta(progress.eta_seconds)} remaining
                </span>
              </div>
              <div
                style={{
                  width: "100%",
                  height: 6,
                  background: "#1f2330",
                  borderRadius: 3,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width:
                      progress.total > 0
                        ? `${Math.min(100, (progress.frame / progress.total) * 100)}%`
                        : "20%",
                    height: "100%",
                    background: "#3b82f6",
                    transition: "width 200ms ease",
                  }}
                />
              </div>
            </>
          ) : (
            <div style={{ fontSize: 13, color: "#94a3b8" }}>
              Uploading and warming up the model…
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{ color: "#f87171", fontSize: 13, whiteSpace: "pre-wrap" }}>{error}</div>
      )}
    </div>
  );
}

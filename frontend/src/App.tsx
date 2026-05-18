import { useEffect, useState } from "react";
import { Uploader } from "./components/Uploader";
import { VideoPlayer } from "./components/VideoPlayer";
import { StatsPanel } from "./components/StatsPanel";
import { DownloadButton } from "./components/DownloadButton";
import type { FrameData, TrackingResult } from "./lib/api";

export function App() {
  const [result, setResult] = useState<TrackingResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState<FrameData | null>(null);

  // Revoke object URLs when they're swapped out, otherwise the browser
  // hangs on to the underlying file memory.
  useEffect(() => {
    if (!videoUrl) return;
    return () => URL.revokeObjectURL(videoUrl);
  }, [videoUrl]);

  function handleResult(file: File, r: TrackingResult) {
    setVideoUrl(URL.createObjectURL(file));
    setVideoName(file.name);
    setResult(r);
    setCurrentFrame(null);
  }

  function handleReset() {
    setResult(null);
    setVideoUrl(null);
    setVideoName(null);
    setCurrentFrame(null);
  }

  return (
    <div
      style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: "32px 24px",
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 12,
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: 28 }}>Object Tracker</h1>
          <p style={{ margin: "4px 0 0", color: "#94a3b8", fontSize: 14 }}>
            Upload a football clip — get per-frame player tracking and team counts.
          </p>
        </div>
        {result && videoUrl && (
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <DownloadButton videoUrl={videoUrl} videoName={videoName} result={result} />
            <button onClick={handleReset} style={{ background: "#475569" }}>
              New video
            </button>
          </div>
        )}
      </header>

      {!result || !videoUrl ? (
        <Uploader onResult={handleResult} />
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 2fr) minmax(280px, 1fr)",
            gap: 24,
            alignItems: "start",
          }}
        >
          <VideoPlayer videoUrl={videoUrl} result={result} onFrameChange={setCurrentFrame} />
          <StatsPanel result={result} currentFrame={currentFrame} />
        </div>
      )}
    </div>
  );
}

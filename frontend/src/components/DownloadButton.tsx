import { useState } from "react";
import { drawFrame, findClosestFrame } from "../lib/overlay";
import type { TrackingResult } from "../lib/api";

interface DownloadButtonProps {
  videoUrl: string;
  videoName: string | null;
  result: TrackingResult;
}

export function DownloadButton({ videoUrl, videoName, result }: DownloadButtonProps) {
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleClick() {
    setError(null);
    setProgress(0);
    try {
      const blob = await recordAnnotated(videoUrl, result, setProgress);
      triggerDownload(blob, outputName(videoName));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setProgress(null);
    }
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <button
        onClick={handleClick}
        disabled={progress !== null}
        style={{ background: progress !== null ? "#475569" : "#0ea5e9" }}
        title="Records the video with bounding-box overlay at playback speed"
      >
        {progress !== null
          ? `Recording… ${Math.round(progress * 100)}%`
          : "Download annotated"}
      </button>
      {error && (
        <span style={{ color: "#f87171", fontSize: 12, maxWidth: 280 }}>{error}</span>
      )}
    </div>
  );
}

function outputName(videoName: string | null): string {
  const base = videoName?.replace(/\.[^.]+$/, "") ?? "tracked";
  return `${base}_tracked.webm`;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Give the browser a tick to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

type AnyVideo = HTMLVideoElement & {
  captureStream?: () => MediaStream;
  mozCaptureStream?: () => MediaStream;
  requestVideoFrameCallback?: (cb: (now: number, metadata: object) => void) => number;
};

async function recordAnnotated(
  videoUrl: string,
  result: TrackingResult,
  onProgress: (p: number) => void,
): Promise<Blob> {
  if (typeof MediaRecorder === "undefined") {
    throw new Error("Browser does not support MediaRecorder.");
  }

  const video = document.createElement("video") as AnyVideo;
  video.src = videoUrl;
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video for recording."));
  });

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || result.video_info.width;
  canvas.height = video.videoHeight || result.video_info.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context not available.");

  const fps = result.video_info.fps || 30;
  const canvasStream = canvas.captureStream(fps);

  // Best-effort: pull audio off the video element. Falls back silently if the
  // browser doesn't expose captureStream() on HTMLMediaElement (e.g. Safari).
  try {
    const elementStream = video.captureStream?.() ?? video.mozCaptureStream?.();
    if (elementStream) {
      for (const track of elementStream.getAudioTracks()) {
        canvasStream.addTrack(track);
      }
    }
  } catch {
    // ignore — visual recording works without audio
  }

  const mimeCandidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";

  const recorder = new MediaRecorder(canvasStream, mimeType ? { mimeType } : undefined);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  const recordingDone = new Promise<Blob>((resolve, reject) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: mimeType || "video/webm" }));
    recorder.onerror = () => reject(new Error("MediaRecorder error."));
  });

  recorder.start();

  function paint() {
    ctx!.drawImage(video, 0, 0, canvas.width, canvas.height);
    const ms = video.currentTime * 1000;
    const frame = findClosestFrame(result.frames, ms);
    if (frame) {
      drawFrame(ctx!, frame, result.video_info.width, result.video_info.height, false);
    }
    const duration = video.duration || 1;
    onProgress(Math.min(1, video.currentTime / duration));
  }

  const rvfc = video.requestVideoFrameCallback;

  await new Promise<void>((resolve, reject) => {
    let cancelled = false;

    function tick() {
      if (cancelled || video.ended) return;
      paint();
      if (rvfc) {
        rvfc.call(video, () => tick());
      } else {
        requestAnimationFrame(tick);
      }
    }

    video.onended = () => {
      paint();
      resolve();
    };
    video.onerror = () => {
      cancelled = true;
      reject(new Error("Video playback error during recording."));
    };

    video.play().then(() => tick()).catch(reject);
  });

  recorder.stop();
  const blob = await recordingDone;

  for (const track of canvasStream.getTracks()) {
    track.stop();
  }
  video.src = "";
  return blob;
}

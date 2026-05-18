import { useEffect, useRef, useState } from "react";
import { drawFrame, findClosestFrame } from "../lib/overlay";
import type { FrameData, TrackingResult } from "../lib/api";

interface VideoPlayerProps {
  videoUrl: string;
  result: TrackingResult;
  onFrameChange: (frame: FrameData | null) => void;
}

export function VideoPlayer({ videoUrl, result, onFrameChange }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const onFrameChangeRef = useRef(onFrameChange);
  const [dims, setDims] = useState({ width: 0, height: 0 });

  // Keep latest callback in a ref so the rAF loop doesn't restart on every render.
  useEffect(() => {
    onFrameChangeRef.current = onFrameChange;
  }, [onFrameChange]);

  // Size the canvas to match the rendered <video> element, both initially
  // and whenever the window resizes.
  useEffect(() => {
    function syncSize() {
      const video = videoRef.current;
      const container = containerRef.current;
      if (!video || !container) return;
      const rect = video.getBoundingClientRect();
      setDims({ width: rect.width, height: rect.height });
    }

    const video = videoRef.current;
    if (!video) return;

    video.addEventListener("loadedmetadata", syncSize);
    window.addEventListener("resize", syncSize);
    return () => {
      video.removeEventListener("loadedmetadata", syncSize);
      window.removeEventListener("resize", syncSize);
    };
  }, []);

  // The rAF render loop: on every animation frame, look up the closest
  // detection frame by timestamp and redraw the canvas.
  useEffect(() => {
    let frameId = 0;
    let lastFrameId = -1;

    function render() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) {
        frameId = requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        frameId = requestAnimationFrame(render);
        return;
      }

      const currentMs = video.currentTime * 1000;
      const frame = findClosestFrame(result.frames, currentMs);

      if (frame && frame.frame_id !== lastFrameId) {
        drawFrame(ctx, frame, result.video_info.width, result.video_info.height);
        onFrameChangeRef.current(frame);
        lastFrameId = frame.frame_id;
      }

      frameId = requestAnimationFrame(render);
    }

    frameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frameId);
  }, [result]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        background: "#000",
        borderRadius: 12,
        overflow: "hidden",
      }}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        style={{ display: "block", width: "100%", height: "auto" }}
      />
      <canvas
        ref={canvasRef}
        width={dims.width}
        height={dims.height}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: dims.width,
          height: dims.height,
          pointerEvents: "none",
        }}
      />
    </div>
  );
}

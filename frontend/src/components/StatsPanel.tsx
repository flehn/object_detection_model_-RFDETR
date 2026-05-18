import type { CSSProperties } from "react";
import type { FrameData, TrackingResult } from "../lib/api";

interface StatsPanelProps {
  result: TrackingResult;
  currentFrame: FrameData | null;
}

const cellStyle: CSSProperties = {
  padding: 12,
  background: "#1a1d24",
  borderRadius: 8,
  border: "1px solid #2a2d36",
};

function Stat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={cellStyle}>
      <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: color ?? "#e7e9ee" }}>{value}</div>
    </div>
  );
}

export function StatsPanel({ result, currentFrame }: StatsPanelProps) {
  const counts = currentFrame?.counts ?? {
    team_0: 0,
    team_1: 0,
    unassigned_players: 0,
    referees: 0,
    ball: 0,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div>
        <h3 style={{ margin: "0 0 8px", color: "#94a3b8", fontSize: 13, fontWeight: 600 }}>
          Current frame
        </h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: 12,
          }}
        >
          <Stat label="Team A" value={counts.team_0} color="#ef4444" />
          <Stat label="Team B" value={counts.team_1} color="#3b82f6" />
          <Stat label="Unassigned" value={counts.unassigned_players} color="#94a3b8" />
          <Stat label="Referees" value={counts.referees} color="#facc15" />
        </div>
      </div>

      <div>
        <h3 style={{ margin: "0 0 8px", color: "#94a3b8", fontSize: 13, fontWeight: 600 }}>
          Whole video
        </h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: 12,
          }}
        >
          <Stat label="Frames" value={result.summary.total_frames} />
          <Stat label="Peak Team A" value={result.summary.peak_counts.team_0} color="#ef4444" />
          <Stat label="Peak Team B" value={result.summary.peak_counts.team_1} color="#3b82f6" />
          <Stat
            label="Unique IDs"
            value={result.summary.unique_ids.team_0 + result.summary.unique_ids.team_1}
          />
        </div>
      </div>
    </div>
  );
}

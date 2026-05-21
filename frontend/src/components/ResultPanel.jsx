import { useState } from "react";
import FrameChart from "./FrameChart";

export default function ResultPanel({ result, pipelineResult, loading }) {
  const [selectedSeg, setSelectedSeg] = useState(0);

  if (loading) {
    return (
      <div className="result-panel empty">
        <div className="spinner" />
        <p>Running inference...</p>
      </div>
    );
  }

  // Pipeline modu
  if (pipelineResult) {
    const { segments, video_is_anomaly, anomaly_ratio, threshold } = pipelineResult;
    const seg = segments[selectedSeg];

    return (
      <div className="result-panel">
        <div className={`decision-badge ${video_is_anomaly ? "anomaly" : "normal"}`}>
          {video_is_anomaly ? "⚠ ANOMALY DETECTED" : "✓ NORMAL VIDEO"}
        </div>

        <div className="metrics-row">
          <div className="metric">
            <span className="metric-label">Segments</span>
            <span className="metric-value">{segments.length}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Anomaly ratio</span>
            <span className="metric-value">{(anomaly_ratio * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Threshold</span>
            <span className="metric-value">{threshold.toFixed(3)}</span>
          </div>
        </div>

        {/* Timeline */}
        <p className="prompt-label">TIMELINE — segment seç</p>
        <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", marginBottom: "12px" }}>
          {segments.map((s, i) => (
            <div
              key={i}
              onClick={() => setSelectedSeg(i)}
              title={`${s.start_sec}s–${s.end_sec}s | score=${s.score.toFixed(3)}`}
              style={{
                width: "36px",
                height: "36px",
                borderRadius: "6px",
                background: s.is_anomaly ? "#f87171" : "#4ade80",
                color: s.is_anomaly ? "#fff" : "#000",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                fontSize: "11px",
                fontWeight: "600",
                outline: i === selectedSeg ? "3px solid #fff" : "none",
              }}
            >
              {i}
            </div>
          ))}
        </div>

        {/* Seçili segment */}
        {seg && (
          <div style={{ borderTop: "1px solid #222", paddingTop: "12px" }}>
            <p className="prompt-label">
              SEGMENT {seg.segment_idx} — {seg.start_sec}s – {seg.end_sec}s
            </p>
            <div className={`decision-badge ${seg.is_anomaly ? "anomaly" : "normal"}`}
              style={{ fontSize: "12px", padding: "4px 10px", marginBottom: "8px" }}>
              {seg.is_anomaly ? `⚠ ${seg.anomaly_type}` : "✓ Normal"}
            </div>
            <p style={{ color: "#666", fontSize: "12px" }}>Score: {seg.score.toFixed(4)}</p>

            <FrameChart scores={seg.frame_scores} threshold={threshold} />

            {seg.all_frames && seg.all_frames.length > 0 && (
              <div className="anomaly-frames">
                <p className="prompt-label">ANALYZED FRAMES</p>
                <div className="frames-grid">
                  {seg.all_frames.map((b64, i) => (
                    <div key={i} className="frame-wrapper">
                      <img
                        src={`data:image/jpeg;base64,${b64}`}
                        alt={`frame ${i}`}
                        className={`anomaly-frame-img ${seg.frame_scores[i] > threshold ? "anom" : "norm"}`}
                      />
                      <span className="frame-score">{seg.frame_scores[i].toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // Single modu
  if (!result) {
    return (
      <div className="result-panel empty">
        <p>Results will appear here</p>
      </div>
    );
  }

  const { is_anomaly, anomaly_type, score_diff, threshold, top_prompt, frame_scores, all_frames } = result;

  return (
    <div className="result-panel">
      <div className={`decision-badge ${is_anomaly ? "anomaly" : "normal"}`}>
        {is_anomaly ? "⚠ ANOMALY DETECTED" : "✓ NORMAL VIDEO"}
      </div>

      {is_anomaly && <p className="anomaly-type">{anomaly_type}</p>}

      <div className="metrics-row">
        <div className="metric">
          <span className="metric-label">Score diff</span>
          <span className={`metric-value ${is_anomaly ? "danger" : "ok"}`}>
            {score_diff.toFixed(3)}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Threshold</span>
          <span className="metric-value">{threshold.toFixed(3)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Frames</span>
          <span className="metric-value">{frame_scores.length}</span>
        </div>
      </div>

      <FrameChart scores={frame_scores} threshold={threshold} />

      {all_frames && all_frames.length > 0 && (
        <div className="anomaly-frames">
          <p className="prompt-label">ANALYZED FRAMES</p>
          <div className="frames-grid">
            {all_frames.map((b64, i) => (
              <div key={i} className="frame-wrapper">
                <img
                  src={`data:image/jpeg;base64,${b64}`}
                  alt={`frame ${i}`}
                  className={`anomaly-frame-img ${frame_scores[i] > threshold ? "anom" : "norm"}`}
                />
                <span className="frame-score">{frame_scores[i].toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
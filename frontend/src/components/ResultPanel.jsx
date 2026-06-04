import { useState } from "react";
import FrameChart from "./FrameChart";

export default function ResultPanel({ result, pipelineResult, loading, mode }) {
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
    const { segments, video_is_anomaly, threshold, score_diff } = pipelineResult;
    const seg = segments[selectedSeg];

    return (
      <div className="result-panel">
        <div style={{
          borderLeft: "3px solid #f59e0b",
          background: "#161616",
          borderRadius: "4px",
          padding: "6px 10px",
          marginBottom: "10px",
          display: "flex",
          flexDirection: "column",
          gap: "2px",
        }}>
          <span style={{ color: "#f59e0b", fontSize: "11px", fontWeight: 700, letterSpacing: "0.5px" }}>
            PIPELINE — Lokalizasyon
          </span>
          <span style={{ color: "#888", fontSize: "11px" }}>
            Segment-segment zaman çizelgesi; karar video-seviyesi top-K ile verilir.
          </span>
        </div>

        <div className={`decision-badge ${video_is_anomaly ? "anomaly" : "normal"}`}>
          {video_is_anomaly ? "⚠ ANOMALY DETECTED" : "✓ NORMAL VIDEO"}
        </div>

        <div className="metrics-row">
          <div className="metric">
            <span className="metric-label">Segments</span>
            <span className="metric-value">{segments.length}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Score diff</span>
            <span className={`metric-value ${video_is_anomaly ? "danger" : "ok"}`}>
              {(score_diff ?? 0).toFixed(3)}
            </span>
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
              {seg.is_anomaly ? "⚠ Anomali" : "✓ Normal"}
            </div>
            <p style={{ color: "#666", fontSize: "12px" }}>Score: {seg.score.toFixed(4)}</p>

            {seg.all_frames && seg.all_frames.length > 0 && (
              <div className="anomaly-frames">
                <p className="prompt-label">SEGMENT FRAME</p>
                <div className="frames-grid">
                  {seg.all_frames.map((b64, i) => (
                    <div key={i} className="frame-wrapper">
                      <img
                        src={`data:image/jpeg;base64,${b64}`}
                        alt={`segment ${seg.segment_idx} frame`}
                        className={`anomaly-frame-img ${seg.is_anomaly ? "anom" : "norm"}`}
                      />
                      <span className="frame-score">{seg.score.toFixed(3)}</span>
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

  // Henüz sonuç yok
  if (!result) {
    const EMPTY_INFO = {
      single: {
        tag: "SINGLE — Hızlı triyaj",
        desc: "Ham video yükle → tek bir video-seviyesi evet/hayır kararı.",
        color: "#0ea5e9",
      },
      pipeline: {
        tag: "PIPELINE — Lokalizasyon",
        desc: "Ham video yükle → segment-segment zaman çizelgesi (anomali nerede?).",
        color: "#f59e0b",
      },
      test: {
        tag: "TEST — Doğruluk kanıtı",
        desc: "Datasetteki segment klasörü → score_diff notebook tablosuyla eşleşir.",
        color: "#a855f7",
      },
    };
    const info = EMPTY_INFO[mode] || EMPTY_INFO.single;
    return (
      <div className="result-panel empty">
        <div style={{
          borderLeft: `3px solid ${info.color}`,
          background: "#161616",
          borderRadius: "4px",
          padding: "10px 12px",
          maxWidth: "320px",
          textAlign: "left",
        }}>
          <p style={{ color: info.color, fontSize: "12px", fontWeight: 700, margin: "0 0 4px", letterSpacing: "0.5px" }}>
            {info.tag}
          </p>
          <p style={{ color: "#888", fontSize: "12px", margin: 0 }}>{info.desc}</p>
        </div>
      </div>
    );
  }

  const { is_anomaly, score_diff, threshold, frame_scores, all_frames } = result;

  const MODE_BANNER = {
    single: {
      tag: "SINGLE — Hızlı triyaj",
      desc: "Ham video → tek bir video-seviyesi evet/hayır kararı.",
      color: "#0ea5e9",
    },
    test: {
      tag: "TEST — Doğruluk kanıtı",
      desc: "Datasetteki hazır segmentler → score_diff notebook tablosuyla birebir eşleşir.",
      color: "#a855f7",
    },
  };
  const banner = MODE_BANNER[mode] || MODE_BANNER.single;

  return (
    <div className="result-panel">
      <div style={{
        borderLeft: `3px solid ${banner.color}`,
        background: "#161616",
        borderRadius: "4px",
        padding: "6px 10px",
        marginBottom: "10px",
        display: "flex",
        flexDirection: "column",
        gap: "2px",
      }}>
        <span style={{ color: banner.color, fontSize: "11px", fontWeight: 700, letterSpacing: "0.5px" }}>
          {banner.tag}
        </span>
        <span style={{ color: "#888", fontSize: "11px" }}>{banner.desc}</span>
      </div>

      <div className={`decision-badge ${is_anomaly ? "anomaly" : "normal"}`}>
        {is_anomaly ? "⚠ ANOMALY DETECTED" : "✓ NORMAL VIDEO"}
      </div>

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
          <span className="metric-label">Segments</span>
          <span className="metric-value">{frame_scores.length}</span>
        </div>
      </div>

      <FrameChart scores={frame_scores} threshold={threshold} />

      {all_frames && all_frames.length > 0 && (
        <div className="anomaly-frames">
          <p className="prompt-label">ANALYZED SEGMENTS</p>
          <div className="frames-grid">
            {all_frames.map((b64, i) => (
              <div key={i} className="frame-wrapper">
                <img
                  src={`data:image/jpeg;base64,${b64}`}
                  alt={`segment ${i}`}
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
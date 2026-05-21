import "./FrameChart.css";

export default function FrameChart({ scores, threshold }) {
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const range = maxScore - minScore || 1;

  return (
    <div className="frame-chart">
      <p className="chart-label">FRAME-LEVEL SIMILARITY SCORES</p>
      <div className="chart-area">
        {scores.map((s, i) => {
          const heightPct = ((s - minScore) / range) * 100;
          const isAnom = s > threshold;
          return (
            <div key={i} className="bar-col" title={`Frame ${i}: ${s.toFixed(3)}`}>
              <div
                className={`bar ${isAnom ? "anom" : "norm"}`}
                style={{ height: `${heightPct}%` }}
              />
              {i % 4 === 0 && <span className="bar-tick">{i}</span>}
            </div>
          );
        })}

        {/* Threshold line */}
        <div
          className="threshold-line"
          style={{ bottom: `${((threshold - minScore) / range) * 100}%` }}
        >
          <span className="threshold-tag">thr={threshold.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

export default function ProgressBar({ pct, label }) {
  return (
    <div className="progress-wrap">
      <p className="progress-label">{label}</p>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

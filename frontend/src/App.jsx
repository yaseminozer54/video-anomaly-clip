import { useState } from "react";
import UploadZone from "./components/UploadZone";
import ResultPanel from "./components/ResultPanel";
import ProgressBar from "./components/ProgressBar";
import { predictVideo, predictFromFrames, predictPipeline } from "./services/api";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ pct: 0, label: "" });
  const [result, setResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("single"); // "single" | "pipeline" | "test"
  const [segmentPath, setSegmentPath] = useState("");

  const STEPS = [
    [15, "Extracting frames..."],
    [35, "Creating segments..."],
    [60, "Running CLIP inference..."],
    [80, "Aggregating scores..."],
    [95, "Applying threshold..."],
    [100, "Done."],
  ];

  async function handlePredict() {
    setLoading(true);
    setResult(null);
    setPipelineResult(null);
    setError(null);

    for (const [pct, label] of STEPS) {
      setProgress({ pct, label });
      await new Promise((r) => setTimeout(r, 400));
    }

    try {
      if (mode === "single") {
        const data = await predictVideo(file);
        setResult(data);
      } else if (mode === "pipeline") {
        const data = await predictPipeline(file);
        setPipelineResult(data);
      } else if (mode === "test") {
        const data = await predictFromFrames(segmentPath);
        setResult(data);
      }
    } catch (e) {
      setError("Backend bağlantısı kurulamadı.");
    } finally {
      setLoading(false);
      setProgress({ pct: 0, label: "" });
    }
  }

  const canRun = mode === "test" ? !!segmentPath : !!file;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Zero-Shot Video Anomaly Detection</h1>
          <p>CLIP-based · No training required · Frame-level analysis</p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {["single", "pipeline", "test"].map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setResult(null); setPipelineResult(null); setError(null); }}
              style={{
                background: mode === m ? "#7c3aed" : "#333",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
                padding: "4px 12px",
                cursor: "pointer",
                fontSize: "12px",
              }}
            >
              {m === "single" ? "Single" : m === "pipeline" ? "Pipeline" : "Test"}
            </button>
          ))}
          <span className="model-badge">CLIP ViT-L/14</span>
        </div>
      </header>

      <main className="app-main">
        <section className="upload-section">
          {mode === "test" ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <p style={{ color: "#aaa", fontSize: "13px" }}>
                Segment klasörü path'i gir
              </p>
              <input
                type="text"
                value={segmentPath}
                onChange={(e) => setSegmentPath(e.target.value)}
                placeholder="/home/yasemin/.../seg_0000"
                style={{
                  background: "#1a1a1a",
                  border: "1px solid #444",
                  borderRadius: "8px",
                  padding: "10px 14px",
                  color: "#fff",
                  fontSize: "13px",
                  width: "100%",
                }}
              />
            </div>
          ) : (
            <UploadZone file={file} onFile={setFile} />
          )}

          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={!canRun || loading}
          >
            {loading ? "Processing..." : mode === "pipeline" ? "Run Pipeline →" : "Run Inference →"}
          </button>

          {loading && <ProgressBar pct={progress.pct} label={progress.label} />}
          {error && <p className="error-msg">{error}</p>}
        </section>

        <section className="result-section">
          <ResultPanel result={result} pipelineResult={pipelineResult} loading={loading} />
        </section>
      </main>
    </div>
  );
}
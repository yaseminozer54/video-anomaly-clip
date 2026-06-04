import { useState } from "react";
import UploadZone from "./components/UploadZone";
import ResultPanel from "./components/ResultPanel";
import ProgressBar from "./components/ProgressBar";
import {
  predictVideo,
  predictFromFrames,
  predictPipeline,
  predictPipelineFromFrames,
} from "./services/api";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ pct: 0, label: "" });
  const [result, setResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("single"); // "single" | "pipeline" | "test"
  const [pipelineSource, setPipelineSource] = useState("frames"); // "frames" | "video"
  const [segmentPath, setSegmentPath] = useState("");

  const STEPS = [
    [15, "Extracting frames..."],
    [35, "Creating segments..."],
    [60, "Running CLIP inference..."],
    [80, "Aggregating segment scores..."],
    [95, "Applying threshold..."],
    [100, "Done."],
  ];

  function clearOutputs() {
    setResult(null);
    setPipelineResult(null);
    setError(null);
  }

  async function handlePredict() {
    setLoading(true);
    clearOutputs();

    for (const [pct, label] of STEPS) {
      setProgress({ pct, label });
      await new Promise((r) => setTimeout(r, 400));
    }

    try {
      if (mode === "single") {
        setResult(await predictVideo(file));
      } else if (mode === "pipeline") {
        if (pipelineSource === "frames") {
          setPipelineResult(await predictPipelineFromFrames(segmentPath));
        } else {
          setPipelineResult(await predictPipeline(file));
        }
      } else if (mode === "test") {
        setResult(await predictFromFrames(segmentPath));
      }
    } catch (e) {
      setError("Backend bağlantısı kurulamadı.");
    } finally {
      setLoading(false);
      setProgress({ pct: 0, label: "" });
    }
  }

  // Path kutusu mu, dosya yükleme mi gösterilecek?
  const showPathInput =
    mode === "test" || (mode === "pipeline" && pipelineSource === "frames");

  const canRun = showPathInput ? !!segmentPath : !!file;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Zero-Shot Video Anomaly Detection</h1>
          <p>CLIP-based · No training required · Segment-level analysis</p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {["single", "pipeline", "test"].map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); clearOutputs(); }}
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
          {/* Pipeline modunda kaynak seçimi */}
          {mode === "pipeline" && (
            <div style={{ display: "flex", gap: "6px", marginBottom: "10px" }}>
              {[
                ["frames", "Segment klasörü"],
                ["video", "Ham video"],
              ].map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => { setPipelineSource(key); clearOutputs(); }}
                  style={{
                    background: pipelineSource === key ? "#f59e0b" : "#2a2a2a",
                    color: pipelineSource === key ? "#000" : "#aaa",
                    border: "1px solid #444",
                    borderRadius: "6px",
                    padding: "4px 10px",
                    cursor: "pointer",
                    fontSize: "12px",
                    fontWeight: 600,
                  }}
                >
                  {label}
                </button>
              ))}
              <span style={{ color: "#666", fontSize: "11px", alignSelf: "center" }}>
                {pipelineSource === "frames"
                  ? "kayıpsız · kalibrasyonla tutarlı"
                  : "mp4 · sayılar tabloyla birebir olmayabilir"}
              </span>
            </div>
          )}

          {showPathInput ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <p style={{ color: "#aaa", fontSize: "13px" }}>
                {mode === "test"
                  ? "Segment klasörü path'i gir"
                  : "Video klasörü path'i gir (içinde seg_xxxx alt-klasörleri)"}
              </p>
              <input
                type="text"
                value={segmentPath}
                onChange={(e) => setSegmentPath(e.target.value)}
                placeholder="/home/yasemin/.../Normal_Videos065_x264"
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
          <ResultPanel result={result} pipelineResult={pipelineResult} loading={loading} mode={mode} />
        </section>
      </main>
    </div>
  );
}
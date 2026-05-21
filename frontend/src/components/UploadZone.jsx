import { useRef, useState } from "react";

export default function UploadZone({ file, onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFile(f) {
    if (f && f.type.startsWith("video/")) onFile(f);
  }

  return (
    <div
      className={`upload-zone ${dragging ? "drag-over" : ""} ${file ? "has-file" : ""}`}
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />

      {!file ? (
        <div className="upload-placeholder">
          <div className="upload-icon">▶</div>
          <p className="upload-title">Drop video or click</p>
          <p className="upload-sub">.mp4 · .avi · .mov</p>
        </div>
      ) : (
        <div className="file-info">
          <div className="file-icon">✓</div>
          <p className="file-name">{file.name}</p>
          <p className="file-size">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
        </div>
      )}
    </div>
  );
}

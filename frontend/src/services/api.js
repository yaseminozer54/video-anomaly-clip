const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function predictFromFrames(segmentPath) {
  const response = await fetch(
    `${API_URL}/predict/frames?segment_path=${encodeURIComponent(segmentPath)}`,
    { method: "POST" }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function predictPipeline(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict/pipeline`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function predictPipelineFromFrames(segmentPath) {
  const response = await fetch(
    `${API_URL}/predict/pipeline/frames?segment_path=${encodeURIComponent(segmentPath)}`,
    { method: "POST" }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
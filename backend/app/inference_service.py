"""
inference_service.py — B sistemi backend.
Kirmizi (anomali) segment isareti TUM modlarda TEK anlama gelir:
'karari suren top-K segment'. Tek segment esige gore degil; video anomali ise
ve segment top-K icindeyse kirmizi. Normal videoda hic kirmizi olmaz.
"""

import sys
import json
import base64
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = "/home/yasemin/video-anomaly-clip"
sys.path.append(PROJECT_ROOT)

from src.inference import CLIPInference

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEGMENT_FRAMES = 16
ARTIFACTS      = f"{PROJECT_ROOT}/artifacts/threshold.json"

ANOMALY_TYPE_LABELS = [
    "Explosion", "Fighting", "Burglary", "Arrest",
    "RoadAccidents", "Vandalism", "Assault", "Robbery",
]

print(f"[inference_service] CLIP yukleniyor ({DEVICE})...")
_clip = CLIPInference()
_clip.set_text_prompts()

with open(ARTIFACTS) as _f:
    _cal = json.load(_f)
THRESHOLD  = float(_cal["threshold"])
TOPK_RATIO = float(_cal["topk_ratio"])
print(f"[inference_service] threshold={THRESHOLD:.6f}  topk_ratio={TOPK_RATIO}  (threshold.json'dan)")


# ---------------------------------------------------------------------
# Cekirdek
# ---------------------------------------------------------------------
def _segment_sims(frames_bgr):
    embs = []
    with torch.no_grad():
        for f in frames_bgr:
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            inp = _clip.preprocess(img).unsqueeze(0).to(DEVICE)
            embs.append(_clip.model.encode_image(inp))
    seg = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
    seg = F.normalize(seg, dim=-1)
    a = (seg @ _clip.anomaly_embeddings.T).squeeze(0).cpu().numpy()
    n = (seg @ _clip.normal_embeddings.T).squeeze(0).cpu().numpy()
    return a, n


def _topk_mean_over_segments(values, ratio):
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(len(arr) * ratio))
    return float(arr[:k].mean())


def _topk_anomaly_indices(seg_scores, ratio):
    n = len(seg_scores)
    k = max(1, int(n * ratio))
    order = sorted(range(n), key=lambda i: seg_scores[i], reverse=True)
    return set(order[:k])


def _video_decision(seg_anomaly, seg_normal):
    score_diff = (
        _topk_mean_over_segments(seg_anomaly, TOPK_RATIO)
        - _topk_mean_over_segments(seg_normal, TOPK_RATIO)
    )
    return score_diff, bool(score_diff > THRESHOLD)


def _segment_flags(seg_scores, video_is_anomaly):
    """Segment kirmizi mi? video anomali ise top-K segmentler True, digerleri False.
    Video normalse hicbiri True degil. Tum modlarda AYNI kural."""
    hot = _topk_anomaly_indices(seg_scores, TOPK_RATIO) if video_is_anomaly else set()
    return [i in hot for i in range(len(seg_scores))]


def _b64(frame_bgr):
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode("utf-8")


def _split_into_segments(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    segments, current = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current.append(frame)
        if len(current) == SEGMENT_FRAMES:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    cap.release()
    return segments, fps


def _read_segment_frames(seg_dir):
    fp = sorted(seg_dir.glob("*.jpg")) or sorted(seg_dir.glob("*.png"))
    frames = [cv2.imread(str(f)) for f in fp]
    return [f for f in frames if f is not None]


def _build_timeline(seg_anomaly, seg_normal, seg_scores, seg_a_idx, seg_n_idx, seg_b64, fps=None):
    score_diff, video_is_anomaly = _video_decision(seg_anomaly, seg_normal)
    flags = _segment_flags(seg_scores, video_is_anomaly)

    results = []
    for idx in range(len(seg_scores)):
        seg_is_anom = flags[idx]
        if fps is not None:
            start = round(idx * SEGMENT_FRAMES / fps, 2)
            end   = round((idx + 1) * SEGMENT_FRAMES / fps, 2)
        else:
            start, end = float(idx), float(idx + 1)

        if seg_is_anom:
            anomaly_type = ANOMALY_TYPE_LABELS[seg_a_idx[idx]]
            top_prompt   = _clip.anomaly_texts[seg_a_idx[idx]]
        else:
            anomaly_type = None
            top_prompt   = _clip.normal_texts[seg_n_idx[idx]]

        results.append({
            "segment_idx": idx,
            "start_sec": start,
            "end_sec": end,
            "score": round(seg_scores[idx], 6),
            "is_anomaly": seg_is_anom,
            "anomaly_type": anomaly_type,
            "top_prompt": top_prompt,
            "frame_scores": [round(seg_scores[idx], 6)],
            "all_frames": [seg_b64[idx]],
        })

    return {
        "segments": results,
        "video_is_anomaly": video_is_anomaly,
        "anomaly_ratio": round(sum(flags) / max(len(results), 1), 3),
        "n_segments": len(results),
        "threshold": THRESHOLD,
        "score_diff": round(score_diff, 6),
    }


# ---------------------------------------------------------------------
# Endpoint mantiklari
# ---------------------------------------------------------------------
def predict_from_frames(segment_path: str) -> dict:
    """Test Mode — hazir kare klasoru."""
    p = Path(segment_path)
    sub = sorted([d for d in p.iterdir() if d.is_dir()]) if p.is_dir() else []
    seg_anomaly, seg_normal, frame_diffs, all_frames = [], [], [], []
    anomaly_vecs, normal_vecs = [], []
    for sd in (sub if sub else [p]):
        frames = _read_segment_frames(sd)
        if not frames:
            continue
        a, n = _segment_sims(frames)
        seg_anomaly.append(float(a.max()))
        seg_normal.append(float(n.max()))
        anomaly_vecs.append(a)
        normal_vecs.append(n)
        frame_diffs.append(round(float(a.max() - n.max()), 6))
        all_frames.append(_b64(frames[len(frames) // 2]))

    if not seg_anomaly:
        return _empty_prediction()

    score_diff, is_anomaly = _video_decision(seg_anomaly, seg_normal)
    flags = _segment_flags(frame_diffs, is_anomaly)
    if is_anomaly:
        ti = int(np.argmax(np.mean(anomaly_vecs, axis=0)))
        anomaly_type, top_prompt = ANOMALY_TYPE_LABELS[ti], _clip.anomaly_texts[ti]
    else:
        ni = int(np.argmax(np.mean(normal_vecs, axis=0)))
        anomaly_type, top_prompt = None, _clip.normal_texts[ni]

    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "score_diff": round(score_diff, 6),
        "threshold": THRESHOLD,
        "top_prompt": top_prompt,
        "frame_scores": frame_diffs,
        "segment_flags": flags,
        "all_frames": all_frames,
    }


def predict_video(video_path: str) -> dict:
    """Single Mode — ham video."""
    segments, _ = _split_into_segments(video_path)
    if not segments:
        return _empty_prediction()
    seg_anomaly, seg_normal, seg_scores, all_frames = [], [], [], []
    anomaly_vecs, normal_vecs = [], []
    for frames in segments:
        a, n = _segment_sims(frames)
        seg_anomaly.append(float(a.max()))
        seg_normal.append(float(n.max()))
        anomaly_vecs.append(a)
        normal_vecs.append(n)
        seg_scores.append(round(float(a.max() - n.max()), 6))
        all_frames.append(_b64(frames[len(frames) // 2]))

    score_diff, is_anomaly = _video_decision(seg_anomaly, seg_normal)
    flags = _segment_flags(seg_scores, is_anomaly)
    if is_anomaly:
        ti = int(np.argmax(np.mean(anomaly_vecs, axis=0)))
        anomaly_type, top_prompt = ANOMALY_TYPE_LABELS[ti], _clip.anomaly_texts[ti]
    else:
        ni = int(np.argmax(np.mean(normal_vecs, axis=0)))
        anomaly_type, top_prompt = None, _clip.normal_texts[ni]

    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "score_diff": round(score_diff, 6),
        "threshold": THRESHOLD,
        "top_prompt": top_prompt,
        "frame_scores": seg_scores,
        "segment_flags": flags,
        "all_frames": all_frames,
    }


def predict_pipeline(video_path: str) -> dict:
    """Pipeline Mode (ham video)."""
    segments, fps = _split_into_segments(video_path)
    if not segments:
        return {"segments": [], "video_is_anomaly": False,
                "anomaly_ratio": 0.0, "n_segments": 0, "threshold": THRESHOLD, "score_diff": 0.0}
    seg_anomaly, seg_normal, seg_scores, seg_a_idx, seg_n_idx, seg_b64 = [], [], [], [], [], []
    for frames in segments:
        a, n = _segment_sims(frames)
        seg_anomaly.append(float(a.max()))
        seg_normal.append(float(n.max()))
        seg_scores.append(float(a.max()) - float(n.max()))
        seg_a_idx.append(int(np.argmax(a)))
        seg_n_idx.append(int(np.argmax(n)))
        seg_b64.append(_b64(frames[len(frames) // 2]))
    return _build_timeline(seg_anomaly, seg_normal, seg_scores, seg_a_idx, seg_n_idx, seg_b64, fps=fps)


def predict_pipeline_from_frames(segment_path: str) -> dict:
    """Pipeline Mode (kayipsiz segment klasoru)."""
    p = Path(segment_path)
    sub = sorted([d for d in p.iterdir() if d.is_dir()]) if p.is_dir() else []
    seg_anomaly, seg_normal, seg_scores, seg_a_idx, seg_n_idx, seg_b64 = [], [], [], [], [], []
    for sd in (sub if sub else [p]):
        frames = _read_segment_frames(sd)
        if not frames:
            continue
        a, n = _segment_sims(frames)
        seg_anomaly.append(float(a.max()))
        seg_normal.append(float(n.max()))
        seg_scores.append(float(a.max()) - float(n.max()))
        seg_a_idx.append(int(np.argmax(a)))
        seg_n_idx.append(int(np.argmax(n)))
        seg_b64.append(_b64(frames[len(frames) // 2]))
    if not seg_scores:
        return {"segments": [], "video_is_anomaly": False,
                "anomaly_ratio": 0.0, "n_segments": 0, "threshold": THRESHOLD, "score_diff": 0.0}
    return _build_timeline(seg_anomaly, seg_normal, seg_scores, seg_a_idx, seg_n_idx, seg_b64, fps=None)


def _empty_prediction():
    return {
        "is_anomaly": False,
        "anomaly_type": None,
        "score_diff": 0.0,
        "threshold": THRESHOLD,
        "top_prompt": _clip.normal_texts[0],
        "frame_scores": [],
        "segment_flags": [],
        "all_frames": [],
    }
"""
inference_service.py — Gerçek CLIP inference
"""

import sys
import base64
import numpy as np
from pathlib import Path
import cv2
import torch
import clip
from PIL import Image

PROJECT_ROOT = "/home/yasemin/video-anomaly-clip"
sys.path.append(PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANOMALY_CLASSES = [
    "flames and smoke from a fire or arson",
    "people punching and kicking each other in a violent fight",
    "a person breaking into a building or stealing property",
    "police handcuffing or forcibly restraining a person",
    "a serious car crash or road accident",
    "someone vandalizing or destroying property",
    "a person being physically attacked or assaulted",
    "armed robbery or shooting with a weapon",
]

NORMAL_CLASSES = [
    "people walking normally in a public area",
    "security camera footage of a calm street",
    "ordinary daily activity with no danger",
    "surveillance footage of a peaceful environment",
    "normal crowd movement in a safe location",
]

ANOMALY_TYPE_LABELS = [
    "Explosion", "Fighting", "Burglary", "Arrest",
    "RoadAccidents", "Vandalism", "Assault", "Robbery",
]

THRESHOLD      = -0.000795
TOPK_RATIO     = 0.30
SEGMENT_FRAMES = 16

print(f"[inference_service] Loading CLIP on {DEVICE}...")
model, preprocess = clip.load("ViT-L/14", device=DEVICE)
model.eval()

anomaly_tokens = clip.tokenize(ANOMALY_CLASSES).to(DEVICE)
normal_tokens  = clip.tokenize(NORMAL_CLASSES).to(DEVICE)

with torch.no_grad():
    anomaly_text_feats = model.encode_text(anomaly_tokens)
    anomaly_text_feats = anomaly_text_feats / anomaly_text_feats.norm(dim=-1, keepdim=True)
    normal_text_feats  = model.encode_text(normal_tokens)
    normal_text_feats  = normal_text_feats / normal_text_feats.norm(dim=-1, keepdim=True)

print("[inference_service] CLIP ready.")


def _topk_mean(arr: np.ndarray, ratio: float) -> float:
    k = max(1, int(len(arr) * ratio))
    return float(np.sort(arr)[::-1][:k].mean())


def _encode_frames(frames_bgr: list) -> tuple:
    images = torch.stack([
        preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        for f in frames_bgr
    ]).to(DEVICE)

    with torch.no_grad():
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        anomaly_sims = (img_feats @ anomaly_text_feats.T)
        normal_sims  = (img_feats @ normal_text_feats.T)

    anomaly_per_frame = anomaly_sims.mean(dim=1).cpu().numpy()
    normal_per_frame  = normal_sims.mean(dim=1).cpu().numpy()
    diff_per_frame    = anomaly_per_frame - normal_per_frame
    anomaly_sims_mean = anomaly_sims.mean(dim=0).cpu().numpy()

    base64_frames = []
    for f in frames_bgr:
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode("utf-8")
        base64_frames.append(b64)

    return anomaly_sims_mean, diff_per_frame, base64_frames


def preprocess_video(video_path: str) -> list:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, SEGMENT_FRAMES, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def run_clip_inference(frames: list) -> list:
    if not frames:
        return []
    _, diff_per_frame, _ = _encode_frames(frames)
    return diff_per_frame.tolist()


def predict_video(video_path: str) -> dict:
    frames = preprocess_video(video_path)
    if not frames:
        return {
            "is_anomaly"  : False,
            "anomaly_type": None,
            "score_diff"  : 0.0,
            "threshold"   : THRESHOLD,
            "top_prompt"  : NORMAL_CLASSES[0],
            "frame_scores": [],
            "all_frames"  : [],
        }

    anomaly_sims_mean, diff_per_frame, base64_frames = _encode_frames(frames)
    score_diff = _topk_mean(diff_per_frame, TOPK_RATIO)
    is_anomaly = score_diff > THRESHOLD

    top_idx      = int(np.argmax(anomaly_sims_mean))
    top_prompt   = ANOMALY_CLASSES[top_idx]
    anomaly_type = ANOMALY_TYPE_LABELS[top_idx] if is_anomaly else None

    return {
        "is_anomaly"  : bool(is_anomaly),
        "anomaly_type": anomaly_type,
        "score_diff"  : round(float(score_diff), 6),
        "threshold"   : THRESHOLD,
        "top_prompt"  : top_prompt,
        "frame_scores": [round(s, 6) for s in diff_per_frame.tolist()],
        "all_frames"  : base64_frames,
    }


def predict_from_frames(segment_path: str) -> dict:
    p = Path(segment_path)
    frame_paths = sorted(p.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(p.glob("*.png"))
    if not frame_paths:
        return {
            "is_anomaly"  : False,
            "anomaly_type": None,
            "score_diff"  : 0.0,
            "threshold"   : THRESHOLD,
            "top_prompt"  : NORMAL_CLASSES[0],
            "frame_scores": [],
            "all_frames"  : [],
        }

    frames = [cv2.imread(str(f)) for f in frame_paths]
    frames = [f for f in frames if f is not None]

    anomaly_sims_mean, diff_per_frame, base64_frames = _encode_frames(frames)
    score_diff = _topk_mean(diff_per_frame, TOPK_RATIO)
    is_anomaly = score_diff > THRESHOLD

    top_idx      = int(np.argmax(anomaly_sims_mean))
    top_prompt   = ANOMALY_CLASSES[top_idx]
    anomaly_type = ANOMALY_TYPE_LABELS[top_idx] if is_anomaly else None

    return {
        "is_anomaly"  : bool(is_anomaly),
        "anomaly_type": anomaly_type,
        "score_diff"  : round(float(score_diff), 6),
        "threshold"   : THRESHOLD,
        "top_prompt"  : top_prompt,
        "frame_scores": [round(s, 6) for s in diff_per_frame.tolist()],
        "all_frames"  : base64_frames,
    }


def predict_pipeline(video_path: str) -> dict:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total <= 0:
        cap.release()
        return {"segments": [], "video_is_anomaly": False, "anomaly_ratio": 0.0, "n_segments": 0, "threshold": THRESHOLD}

    segments = []
    current  = []

    for i in range(total):
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

    results = []
    for idx, frames in enumerate(segments):
        anomaly_sims_mean, diff_per_frame, base64_frames = _encode_frames(frames)
        score_diff = _topk_mean(diff_per_frame, TOPK_RATIO)
        is_anomaly = score_diff > THRESHOLD

        top_idx      = int(np.argmax(anomaly_sims_mean))
        top_prompt   = ANOMALY_CLASSES[top_idx]
        anomaly_type = ANOMALY_TYPE_LABELS[top_idx] if is_anomaly else None

        start_sec = (idx * SEGMENT_FRAMES) / fps
        end_sec   = ((idx + 1) * SEGMENT_FRAMES) / fps

        results.append({
            "segment_idx" : idx,
            "start_sec"   : round(start_sec, 2),
            "end_sec"     : round(end_sec, 2),
            "score"       : round(float(score_diff), 6),
            "is_anomaly"  : bool(is_anomaly),
            "anomaly_type": anomaly_type,
            "top_prompt"  : top_prompt,
            "frame_scores": [round(s, 6) for s in diff_per_frame.tolist()],
            "all_frames"  : base64_frames,
        })

    anomaly_count    = sum(1 for r in results if r["is_anomaly"])
    video_is_anomaly = anomaly_count > 0

    return {
        "segments"        : results,
        "video_is_anomaly": video_is_anomaly,
        "anomaly_ratio"   : round(anomaly_count / max(len(results), 1), 3),
        "n_segments"      : len(results),
        "threshold"       : THRESHOLD,
    }
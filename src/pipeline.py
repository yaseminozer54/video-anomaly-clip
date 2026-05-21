"""
src/pipeline.py

run_realtime_pipeline — video dosyasını segment segment işler
measure_fps           — kaç segment/saniye işlendiğini ölçer
"""

import time
import numpy as np
import cv2
import torch
import clip
from PIL import Image
from pathlib import Path

# ── Config ──────────────────────────────────────────────
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
    "Explosion",      # flames and smoke from a fire or arson
    "Fighting",       # people punching and kicking
    "Burglary",       # breaking into a building
    "Arrest",         # police handcuffing
    "RoadAccidents",  # car crash
    "Vandalism",      # vandalizing property
    "Assault",        # physically attacked
    "Robbery",        # armed robbery
]

SEGMENT_FRAMES = 16   # segment başına frame sayısı
TOPK_RATIO     = 0.30
DEFAULT_THRESHOLD = -0.000795

# ── Model yükle ─────────────────────────────────────────
print(f"[pipeline] Loading CLIP on {DEVICE}...")
model, preprocess = clip.load("ViT-L/14", device=DEVICE)
model.eval()

anomaly_tokens = clip.tokenize(ANOMALY_CLASSES).to(DEVICE)
normal_tokens  = clip.tokenize(NORMAL_CLASSES).to(DEVICE)

with torch.no_grad():
    anomaly_text_feats = model.encode_text(anomaly_tokens)
    anomaly_text_feats = anomaly_text_feats / anomaly_text_feats.norm(dim=-1, keepdim=True)
    normal_text_feats  = model.encode_text(normal_tokens)
    normal_text_feats  = normal_text_feats / normal_text_feats.norm(dim=-1, keepdim=True)

print("[pipeline] CLIP ready.")


# ── Yardımcı ────────────────────────────────────────────
def _topk_mean(arr: np.ndarray, ratio: float) -> float:
    k = max(1, int(len(arr) * ratio))
    return float(np.sort(arr)[::-1][:k].mean())


def _score_segment(frames: list) -> dict:
    """
    Frame listesi → segment skoru ve anomaly type döndürür.
    """
    images = torch.stack([
        preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        for f in frames
    ]).to(DEVICE)

    with torch.no_grad():
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        anomaly_sims = (img_feats @ anomaly_text_feats.T)  # [N, 8]
        normal_sims  = (img_feats @ normal_text_feats.T)   # [N, 5]

    anomaly_per_frame = anomaly_sims.mean(dim=1).cpu().numpy()
    normal_per_frame  = normal_sims.mean(dim=1).cpu().numpy()
    diff_per_frame    = anomaly_per_frame - normal_per_frame

    score = _topk_mean(diff_per_frame, TOPK_RATIO)

    # En yüksek anomaly prompt
    top_idx      = int(anomaly_sims.mean(dim=0).argmax().item())
    anomaly_type = ANOMALY_TYPE_LABELS[top_idx]
    top_prompt   = ANOMALY_CLASSES[top_idx]

    return {
        "score"       : round(score, 6),
        "frame_scores": [round(s, 6) for s in diff_per_frame.tolist()],
        "anomaly_type": anomaly_type,
        "top_prompt"  : top_prompt,
    }


def _extract_segments(segment_dir: str, segment_frames: int) -> tuple:
    p = Path(segment_dir)

    # Direkt frame klasörü mü (seg_0000 gibi)?
    frames_direct = sorted(p.glob("*.jpg")) or sorted(p.glob("*.png"))
    if frames_direct:
        frames = []
        for f in frames_direct:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
        return [frames], 25.0

    # Video klasörü — alt seg klasörlerini paralel oku
    from concurrent.futures import ThreadPoolExecutor

    seg_dirs = sorted([d for d in p.iterdir() if d.is_dir()])

    def load_seg(seg_dir):
        files = sorted(seg_dir.glob("*.jpg")) or sorted(seg_dir.glob("*.png"))
        frames = []
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
        return frames

    with ThreadPoolExecutor(max_workers=8) as ex:
        segments = list(ex.map(load_seg, seg_dirs))

    segments = [s for s in segments if s]
    return segments, 25.0


def measure_fps(segment_dir: str, n_segments: int = 10) -> dict:
    """
    İlk n_segments segmenti işleyip FPS ölçer.
    """
    segments_raw, _ = _extract_segments(segment_dir, SEGMENT_FRAMES)
    test_segments   = segments_raw[:n_segments]

    if not test_segments:
        print("Segment bulunamadı.")
        return {"n_segments": 0, "elapsed_sec": 0, "fps": 0, "target_met": False}

    start = time.time()
    for frames in test_segments:
        _score_segment(frames)
    elapsed = time.time() - start

    fps = len(test_segments) / elapsed

    print(f"İşlenen segment : {len(test_segments)}")
    print(f"Süre            : {elapsed:.2f}s")
    print(f"FPS             : {fps:.2f}")
    print(f"Hedef (>=15)    : {'✅' if fps >= 15 else '❌'}")

    return {
        "n_segments" : len(test_segments),
        "elapsed_sec": round(elapsed, 3),
        "fps"        : round(fps, 2),
        "target_met" : fps >= 15,
    }


def run_realtime_pipeline(
    segment_dir: str,
    anomaly_score_fn=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    segments_raw, fps_vid = _extract_segments(segment_dir, SEGMENT_FRAMES)

    results = []
    for idx, frames in enumerate(segments_raw):
        if not frames:
            continue
        if anomaly_score_fn is not None:
            score = anomaly_score_fn(frames)
            seg_result = {
                "segment_idx" : idx,
                "score"       : round(float(score), 6),
                "is_anomaly"  : score > threshold,
                "anomaly_type": None,
                "top_prompt"  : None,
                "frame_scores": [],
            }
        else:
            seg = _score_segment(frames)
            seg_result = {
                "segment_idx" : idx,
                "score"       : seg["score"],
                "is_anomaly"  : seg["score"] > threshold,
                "anomaly_type": seg["anomaly_type"] if seg["score"] > threshold else None,
                "top_prompt"  : seg["top_prompt"],
                "frame_scores": seg["frame_scores"],
            }
        results.append(seg_result)

    anomaly_count    = sum(1 for r in results if r["is_anomaly"])
    video_is_anomaly = anomaly_count > 0

    return {
        "segments"        : results,
        "video_is_anomaly": video_is_anomaly,
        "anomaly_ratio"   : round(anomaly_count / max(len(results), 1), 3),
        "threshold"       : threshold,
        "fps_video"       : fps_vid,
        "n_segments"      : len(results),
    }
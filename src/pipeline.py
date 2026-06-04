"""
src/pipeline.py  (duzeltilmis)

Demo / gercek-zamanli cikarim — DEGERLENDIRME boru hattiyla BIREBIR AYNI:
  - Ayni promptlar      : src.config (ANOMALY_CLASSES = 8, NORMAL_CLASSES = 8)
  - Ayni skorlama       : segment = kare-embedding ortalamasi -> prompt-max,
                          video = segmentler uzerinde top-K mean
  - Ayni esik/oran      : artifacts/threshold.json'dan OKUNUR (sabit yazilmaz)

Boylece demo sonucu, notebook'taki tablo ile tutarli olur.
"""

import os
import json
import time
import numpy as np
from pathlib import Path

from src.inference import CLIPInference

# UCF-Crime anomali sinif etiketleri — src.config.ANOMALY_CLASSES SIRASIYLA AYNI olmali
ANOMALY_TYPE_LABELS = [
    "Explosion",      # flames and smoke from a fire or arson
    "Fighting",       # people punching and kicking
    "Burglary",       # breaking into a building / stealing
    "Arrest",         # police handcuffing
    "RoadAccidents",  # car crash
    "Vandalism",      # vandalizing property
    "Assault",        # physically attacked
    "Robbery",        # armed robbery / shooting
]


def _topk_mean_over_segments(values, ratio):
    """En yuksek %ratio segmentin ortalamasi (aggregation.py ile ayni mantik)."""
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(len(arr) * ratio))
    return float(arr[:k].mean())


class AnomalyPipeline:
    def __init__(self, artifacts_path):
        # CLIP + promptlar: degerlendirmedeki ile AYNI kaynak (config.py)
        self.clip = CLIPInference()
        self.clip.set_text_prompts()

        # Kalibrasyon degerlerini artifact'tan oku (sabit yazma!)
        with open(artifacts_path) as f:
            cal = json.load(f)
        self.threshold  = float(cal["threshold"])
        self.topk_ratio = float(cal["topk_ratio"])
        print(f"[pipeline] threshold={self.threshold:.6f}  topk_ratio={self.topk_ratio}")

    def _segment_dirs(self, video_dir):
        """video_dir altinda segment alt-klasorleri; yoksa video_dir'in kendisi tek segment."""
        p = Path(video_dir)
        sub = sorted([d for d in p.iterdir() if d.is_dir()])
        return sub if sub else [p]

    def score_video(self, video_dir):
        """Bir videoyu (segment kare-klasorleri) puanlar ve karar dondurur."""
        seg_anomaly, seg_normal, segments = [], [], []

        for sd in self._segment_dirs(video_dir):
            try:
                pred = self.clip.predict_segment(str(sd))   # inference.py — AYNI kod yolu
            except Exception as e:
                print(f"[pipeline] segment atlandi: {sd} | {e}")
                continue

            a = float(np.max(pred["anomaly_sim"]))   # segment basina: en iyi anomali promptu
            n = float(np.max(pred["normal_sim"]))    # en iyi normal promptu
            seg_anomaly.append(a)
            seg_normal.append(n)

            top_idx = int(np.argmax(pred["anomaly_sim"]))
            segments.append({
                "segment"     : sd.name,
                "seg_score"   : round(a - n, 6),
                "anomaly_type": ANOMALY_TYPE_LABELS[top_idx],
            })

        if not seg_anomaly:
            raise ValueError(f"Puanlanabilir segment yok: {video_dir}")

        # Video skoru — notebook ile AYNI: segmentler uzerinde top-K mean farki
        score_diff = (
            _topk_mean_over_segments(seg_anomaly, self.topk_ratio)
            - _topk_mean_over_segments(seg_normal, self.topk_ratio)
        )
        video_is_anomaly = bool(score_diff > self.threshold)

        return {
            "video_is_anomaly": video_is_anomaly,
            "score_diff"      : round(score_diff, 6),
            "threshold"       : self.threshold,
            "topk_ratio"      : self.topk_ratio,
            "n_segments"      : len(seg_anomaly),
            "segments"        : segments,   # segment-bazli skor + olasi anomali turu (lokalizasyon/UI icin)
        }

    def measure_fps(self, video_dir, n_segments=10):
        """Ilk n_segments segmenti puanlayip saniyede kac segment islendigini olcer."""
        seg_dirs = self._segment_dirs(video_dir)[:n_segments]
        if not seg_dirs:
            return {"n_segments": 0, "elapsed_sec": 0.0, "fps": 0.0, "target_met": False}

        start = time.time()
        ok = 0
        for sd in seg_dirs:
            try:
                self.clip.predict_segment(str(sd))
                ok += 1
            except Exception:
                pass
        elapsed = time.time() - start
        fps = ok / elapsed if elapsed > 0 else 0.0

        print(f"Islenen segment : {ok}")
        print(f"Sure            : {elapsed:.2f}s")
        print(f"FPS             : {fps:.2f}")
        print(f"Hedef (>=15)    : {'OK' if fps >= 15 else 'NO'}")

        return {
            "n_segments" : ok,
            "elapsed_sec": round(elapsed, 3),
            "fps"        : round(fps, 2),
            "target_met" : fps >= 15,
        }


if __name__ == "__main__":
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/yasemin/video-anomaly-clip")
    ARTIFACTS    = f"{PROJECT_ROOT}/artifacts/threshold.json"

    pipe = AnomalyPipeline(ARTIFACTS)
    # ornek: bir video klasoru ver (icinde segment alt-klasorleri)
    # result = pipe.score_video(f"{PROJECT_ROOT}/data/segments_new/<video_id>")
    # print(json.dumps(result, indent=2, ensure_ascii=False))

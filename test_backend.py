"""
test_backend.py — backend (inference_service) notebook ile tutarli mi?

Calistirma (proje koku, CLIP kurulu ortam):
    cd /home/yasemin/video-anomaly-clip
    python test_backend.py

Mantik: predict_from_frames bir videonun segment klasorunu puanlar; cikan
score_diff, notebook'un urettigi val_video_predictions.csv'deki ayni videonun
score_diff'i ile eslesmeli (yuvarlamadan kucuk fark olabilir).
"""

import os
from pathlib import Path
import pandas as pd

# backend modulu: dosya backend/app/ icinde -> oradan import
import sys
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/yasemin/video-anomaly-clip")
sys.path.append(f"{PROJECT_ROOT}/backend")

from app.inference_service import predict_from_frames, THRESHOLD, TOPK_RATIO

MANIFEST = f"{PROJECT_ROOT}/data/manifests_new/manifest.csv"
SEGMENTS = f"{PROJECT_ROOT}/data/segments_new"
VAL_CSV  = f"{PROJECT_ROOT}/outputs/val_video_predictions.csv"

N_CHECK = 5
TOL     = 1e-4

print("=" * 55)
print(f"Backend kalibrasyonu: threshold={THRESHOLD:.6f}  topk_ratio={TOPK_RATIO}")
print("=" * 55)

df = pd.read_csv(MANIFEST)
df["path"] = df["path"].str.replace(
    "/content/drive/MyDrive/clip_delivery_new/segments", SEGMENTS, regex=False
)
val = pd.read_csv(VAL_CSV)

n_ok, n_diff, n_skip = 0, 0, 0
for vid in val["video_id"].head(N_CHECK):
    seg_paths = df[df["video_id"] == vid]["path"].tolist()
    if not seg_paths:
        print(f"  {vid}: manifest'te segment yok"); n_skip += 1; continue
    parents = {str(Path(p).parent) for p in seg_paths}
    if len(parents) != 1:
        print(f"  {vid}: segmentler tek klasorde degil"); n_skip += 1; continue
    video_dir = parents.pop()

    res = predict_from_frames(video_dir)
    csv_score = float(val.loc[val["video_id"] == vid, "score_diff"].iloc[0])
    diff = abs(res["score_diff"] - csv_score)
    status = "OK" if diff < TOL else "FARKLI"
    n_ok += diff < TOL
    n_diff += diff >= TOL
    print(f"  {vid:>22} | backend={res['score_diff']:+.6f} | "
          f"csv={csv_score:+.6f} | fark={diff:.1e} | {status}")

print("\n" + "-" * 55)
print(f"Eslesen: {n_ok}  |  Farkli: {n_diff}  |  Atlanan: {n_skip}")
if n_diff == 0 and n_ok > 0:
    print("SONUC: Backend, notebook ile TUTARLI. Demo = tablo, kanitlandi.")
elif n_skip and not n_ok:
    print("SONUC: Klasor yapisi farkli; seg_paths ornegini gonder, uyarlayayim.")
else:
    print("SONUC: Farkli cikti; o videonun backend cikisini ve CSV satirini gonder.")

"""
test_pipeline.py — pipeline.py dogru calisiyor mu kontrol eder.

Calistirma (kendi makinende, CLIP kurulu ortamda):
    cd /home/yasemin/video-anomaly-clip
    python test_pipeline.py

Ne yapar:
  1) AnomalyPipeline yuklenir mi, threshold.json okunuyor mu  (SMOKE TEST)
  2) Birkac video icin pipeline'in score_diff'i, notebook'un urettigi
     val_video_predictions.csv'deki score_diff ile eslesiyor mu  (TUTARLILIK)
"""

import os
from pathlib import Path
import pandas as pd

from src.pipeline import AnomalyPipeline

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/yasemin/video-anomaly-clip")
MANIFEST     = f"{PROJECT_ROOT}/data/manifests_new/manifest.csv"
SEGMENTS     = f"{PROJECT_ROOT}/data/segments_new"
ARTIFACTS    = f"{PROJECT_ROOT}/artifacts/threshold.json"
VAL_CSV      = f"{PROJECT_ROOT}/outputs/val_video_predictions.csv"

N_CHECK = 5          # kac video kontrol edilsin
TOL     = 1e-4       # kabul edilebilir fark (yuvarlamadan)

# ---- 1) SMOKE TEST -------------------------------------------------
print("=" * 55)
print("1) SMOKE TEST — pipeline yukleniyor")
print("=" * 55)
pipe = AnomalyPipeline(ARTIFACTS)
print("Pipeline yuklendi. threshold ve topk_ratio yukarida yazdi.\n")

# ---- 2) TUTARLILIK TESTI -------------------------------------------
print("=" * 55)
print("2) TUTARLILIK — pipeline score_diff vs notebook CSV")
print("=" * 55)

# Manifest'i oku ve yollari notebook'taki gibi yerele cevir
df = pd.read_csv(MANIFEST)
df["path"] = df["path"].str.replace(
    "/content/drive/MyDrive/clip_delivery_new/segments",
    SEGMENTS, regex=False
)

val = pd.read_csv(VAL_CSV)   # notebook'un urettigi video-seviyesi tahminler

n_ok, n_diff, n_skip = 0, 0, 0
for vid in val["video_id"].head(N_CHECK):
    seg_paths = df[df["video_id"] == vid]["path"].tolist()
    if not seg_paths:
        print(f"  {vid}: manifest'te segment yok, atlandi"); n_skip += 1; continue

    # Bu videonun segmentleri tek bir ust klasorde mi? (score_video bunu bekliyor)
    parents = {str(Path(p).parent) for p in seg_paths}
    if len(parents) != 1:
        print(f"  {vid}: segmentler tek klasorde DEGIL -> score_video uyarlanmali"); n_skip += 1; continue
    video_dir = parents.pop()

    res = pipe.score_video(video_dir)
    csv_score = float(val.loc[val["video_id"] == vid, "score_diff"].iloc[0])
    diff = abs(res["score_diff"] - csv_score)
    status = "OK" if diff < TOL else "FARKLI"
    if diff < TOL: n_ok += 1
    else: n_diff += 1
    print(f"  {vid:>22} | pipeline={res['score_diff']:+.6f} | "
          f"csv={csv_score:+.6f} | fark={diff:.1e} | {status} | "
          f"n_seg={res['n_segments']}")

print("\n" + "-" * 55)
print(f"Eslesen: {n_ok}  |  Farkli: {n_diff}  |  Atlanan: {n_skip}")
if n_diff == 0 and n_ok > 0:
    print("SONUC: Demo ile notebook TUTARLI. pipeline.py dogru calisiyor.")
elif n_skip > 0 and n_ok == 0:
    print("SONUC: Klasor yapisi score_video'nun bekledigi gibi degil.")
    print("       seg_paths'i bana gonder; score_video'yu yapina gore uyarlayayim.")
else:
    print("SONUC: Farkli cikanlar var; skor yolu bir yerde ayrismis.")

# ---- 3) (Istege bagli) FPS ----------------------------------------
# ilk videonun klasorunde hiz olcumu:
# pipe.measure_fps(video_dir, n_segments=10)

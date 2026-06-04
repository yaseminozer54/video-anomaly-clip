import pandas as pd
import numpy as np

# DİKKAT: Bu liste, notebook Cell 15'teki grid-search listesiyle
# BİREBİR AYNI olmak zorunda. Kolon adları buradan üretiliyor.
TOPK_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]


def _topk_mean_over_segments(values, ratio):
    """
    values : video içindeki segmentlerin skaler benzerlik dizisi
    ratio  : en yüksek kaç segmentin ortalanacağı (0-1 arası oran)

    En yüksek %ratio segmenti seçip ortalar. Anomali genelde videonun
    küçük bir bölümünde olduğu için, tüm segmentleri ortalamak yerine
    en güçlü sinyali veren segmentlere bakarız.
    """
    arr = np.sort(np.asarray(values, dtype=float))[::-1]   # büyükten küçüğe
    k = max(1, int(len(arr) * ratio))
    return float(arr[:k].mean())


def aggregate_video_scores(results_df, ratios=TOPK_RATIOS):
    """
    Segment-seviyesi sonuçları video-seviyesine indirger.

    Her segment için elimizde 8-boyutlu prompt benzerlik vektörleri var
    (anomaly_sim, normal_sim). Önce her segmenti tek bir skalere indiriyoruz
    (en iyi anomali / en iyi normal prompt benzerliği), sonra SEGMENTLER
    üzerinde top-K mean alıyoruz.

    Çıkan kolonlar Cell 15'in beklediği isimlerde: topk_a_{ratio}, topk_n_{ratio}
    """
    video_results = []

    for video_id, group in results_df.groupby("video_id"):
        true_class = group["true_class"].iloc[0]
        is_anomaly = true_class != "NormalVideos"

        # Segment başına 8-boyutlu prompt vektörleri -> [n_segment, 8]
        anomaly_vectors = np.stack(group["anomaly_sim"].values)
        normal_vectors = np.stack(group["normal_sim"].values)

        # Her segment için skaler: en iyi eşleşen prompt benzerliği
        seg_anomaly = anomaly_vectors.max(axis=1)   # [n_segment]
        seg_normal = normal_vectors.max(axis=1)     # [n_segment]

        row = {
            "video_id": video_id,
            "true_class": true_class,
            "is_anomaly": is_anomaly,
            "n_segments": int(len(seg_anomaly)),
        }

        # === Cell 15'in beklediği kolonlar: SEGMENTLER üzerinde top-K mean ===
        for r in ratios:
            row[f"topk_a_{r}"] = _topk_mean_over_segments(seg_anomaly, r)
            row[f"topk_n_{r}"] = _topk_mean_over_segments(seg_normal, r)

        # (İsteğe bağlı tanılama kolonları — karar için kullanılmıyor)
        row["max_anomaly_sim"] = float(seg_anomaly.max())
        row["max_normal_sim"] = float(seg_normal.max())
        row["mean_anomaly_sim"] = float(seg_anomaly.mean())
        row["mean_normal_sim"] = float(seg_normal.mean())

        video_results.append(row)

    return pd.DataFrame(video_results)

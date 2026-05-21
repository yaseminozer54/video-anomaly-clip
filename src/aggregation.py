import pandas as pd
import numpy as np


def aggregate_video_scores(results_df):

    video_results = []

    grouped = results_df.groupby("video_id")

    for video_id, group in grouped:

        true_class = group["true_class"].iloc[0]

        is_anomaly = true_class != "NormalVideos"

        anomaly_vectors = np.stack(
            group["anomaly_sim"].values
        )

        normal_vectors = np.stack(
            group["normal_sim"].values
        )

        # =====================================
        # MAX
        # =====================================

        max_anomaly_sim = anomaly_vectors.max(axis=0)

        max_normal_sim = normal_vectors.max(axis=0)

        # =====================================
        # MEAN
        # =====================================

        mean_anomaly_sim = anomaly_vectors.mean(axis=0)

        mean_normal_sim = normal_vectors.mean(axis=0)

        # =====================================
        # P90
        # =====================================

        p90_anomaly_sim = np.percentile(
            anomaly_vectors,
            90,
            axis=0
        )

        p90_normal_sim = np.percentile(
            normal_vectors,
            90,
            axis=0
        )

        # =====================================
        # TOP-K
        # =====================================

        k = min(3, anomaly_vectors.shape[0])

        topk_anomaly_sim = np.mean(
            np.sort(anomaly_vectors, axis=0)[-k:],
            axis=0
        )

        topk_normal_sim = np.mean(
            np.sort(normal_vectors, axis=0)[-k:],
            axis=0
        )

        video_results.append({

            "video_id": video_id,

            "true_class": true_class,

            "is_anomaly": is_anomaly,

            # MAX
            "max_anomaly_sim": max_anomaly_sim,
            "max_normal_sim": max_normal_sim,

            # MEAN
            "mean_anomaly_sim": mean_anomaly_sim,
            "mean_normal_sim": mean_normal_sim,

            # P90
            "p90_anomaly_sim": p90_anomaly_sim,
            "p90_normal_sim": p90_normal_sim,

            # TOP-K
            "topk_anomaly_sim": topk_anomaly_sim,
            "topk_normal_sim": topk_normal_sim
        })

    return pd.DataFrame(video_results)
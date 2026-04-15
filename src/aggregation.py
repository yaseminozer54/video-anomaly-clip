import pandas as pd
import numpy as np

def aggregate_video_scores(results_df):
    video_results = []

    for video_id, group in results_df.groupby("video_id"):
        true_class = group["true_class"].iloc[0]
        is_anomaly = true_class != "NormalVideos"

        all_anomaly_sims = np.vstack(group["anomaly_sim"].values)
        all_normal_sims = np.vstack(group["normal_sim"].values)

        max_anomaly_sims = np.max(all_anomaly_sims, axis=0)
        max_normal_sims = np.max(all_normal_sims, axis=0)

        video_results.append({
            "video_id": video_id,
            "true_class": true_class,
            "is_anomaly": is_anomaly,
            "anomaly_sim": max_anomaly_sims.tolist(),
            "normal_sim": max_normal_sims.tolist()
        })

    return pd.DataFrame(video_results)

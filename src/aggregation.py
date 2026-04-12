import numpy as np
import pandas as pd


# ===============================
# Aggregation Helpers
# ===============================

def max_pooling(scores):
    return np.max(scores)


def mean_pooling(scores):
    return np.mean(scores)


def topk_mean(scores, k=3):
    k = min(k, len(scores))
    topk = sorted(scores, reverse=True)[:k]
    return np.mean(topk)


def percentile_pooling(scores, percentile=90):
    return np.percentile(scores, percentile)


def softmax_pooling(scores, temperature=1.0):
    scores = np.array(scores)
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / np.sum(exp_scores)
    return np.sum(weights * scores)


# ===============================
# Main Aggregation Function
# ===============================

def aggregate_video_scores(results_df, method="max", **kwargs):
    """
    Aggregates segment-level scores into video-level score.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns:
        ["video_id", "segment_uid", "true_class", "score"]

    method : str
        Aggregation method:
        - "max"
        - "mean"
        - "topk"
        - "p90"
        - "softmax"

    kwargs :
        Optional parameters:
        - k (for topk)
        - percentile (for pXX)
        - temperature (for softmax)

    Returns
    -------
    video_df : pd.DataFrame
        Columns:
        ["video_id", "score", "true_class", "is_anomaly"]
    """

    if len(results_df) == 0:
        raise ValueError("results_df is empty")

    video_results = []

    grouped = results_df.groupby("video_id")

    for video_id, group in grouped:
        segment_scores = group["score"].values
        true_class = group["true_class"].iloc[0]

        if method == "max":
            video_score = max_pooling(segment_scores)

        elif method == "mean":
            video_score = mean_pooling(segment_scores)

        elif method == "topk":
            k = kwargs.get("k", 3)
            video_score = topk_mean(segment_scores, k=k)

        elif method == "p90":
            percentile = kwargs.get("percentile", 90)
            video_score = percentile_pooling(segment_scores, percentile)

        elif method == "softmax":
            temperature = kwargs.get("temperature", 1.0)
            video_score = softmax_pooling(segment_scores, temperature)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        is_anomaly = true_class != "NormalVideos"

        video_results.append({
            "video_id": video_id,
            "score": float(video_score),
            "true_class": true_class,
            "is_anomaly": is_anomaly
        })

    video_df = pd.DataFrame(video_results)

    return video_df

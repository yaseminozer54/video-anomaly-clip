import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
import pandas as pd

from src.dataset import SegmentDataset
import src.config as config


def load_model():
    if config.DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif config.DEVICE == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    model, preprocess = clip.load(config.MODEL_NAME, device=device)
    model.eval()

    return model, preprocess, device


def prepare_text_embeddings(model, device):
    text_tokens = clip.tokenize(config.TEXT_PROMPTS).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)

    text_embeddings = F.normalize(text_embeddings, dim=-1)

    return text_embeddings


def run_inference(model, preprocess, text_embeddings, device):
    dataset = SegmentDataset(
    config.MANIFEST_PATH,
    config.SEGMENTS_ROOT,
    return_meta=True
)

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    all_results = []

    with torch.no_grad():
        for frames, true_class, meta in dataloader:

            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W).to(device)

            image_features = model.encode_image(frames)
            image_features = image_features.view(B, T, -1).mean(dim=1)
            image_features = F.normalize(image_features, dim=-1)

            similarity = image_features @ text_embeddings.T
            max_scores, best_idx = similarity.max(dim=1)

            for i in range(B):

                segment_uid = meta["segment_uid"][i]
            
                video_id = segment_uid.split("_seg_")[0]
                split = segment_uid.split("_")[0]
            
                pred_class = config.TEXT_PROMPTS[best_idx[i].item()]
                score = max_scores[i].item()
            
                is_anomaly = (
                    pred_class in config.ANOMALY_CLASSES
                    and score > config.THRESHOLD
                )
            
                all_results.append({
                    "segment_uid": segment_uid,
                    "video_id": video_id,
                    "split": split,
                    "true_class": true_class[i],
                    "pred_class": pred_class,
                    "score": score,
                    "is_anomaly": is_anomaly
                })

    return pd.DataFrame(all_results)


def aggregate_video_predictions(df):
    video_results = []

    grouped = df.groupby("video_id")

    for video_id, group in grouped:
        total_segments = len(group)
        anomaly_segments = group["is_anomaly"].sum()
        anomaly_ratio = anomaly_segments / total_segments

        video_is_anomaly = anomaly_ratio > config.VIDEO_ANOMALY_RATIO

        split = group["split"].iloc[0]
        true_class = group["true_class"].iloc[0]

        video_results.append({
            "video_id": video_id,
            "split": split,
            "true_class": true_class,
            "total_segments": total_segments,
            "anomaly_segments": anomaly_segments,
            "anomaly_ratio": anomaly_ratio,
            "video_is_anomaly": video_is_anomaly
        })

    return pd.DataFrame(video_results)


def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    model, preprocess, device = load_model()

    text_embeddings = prepare_text_embeddings(model, device)

    segment_df = run_inference(
        model=model,
        preprocess=preprocess,
        text_embeddings=text_embeddings,
        device=device
    )

    segment_df.to_csv("outputs/segment_predictions.csv", index=False)

    video_df = aggregate_video_predictions(segment_df)

    video_df.to_csv("outputs/video_predictions.csv", index=False)

    print("Inference completed.")
    print(f"Total segments: {len(segment_df)}")
    print(f"Total videos: {len(video_df)}")

if __name__ == "__main__":
    main()
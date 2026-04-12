import os
import torch
import clip
import torch.nn.functional as F
from PIL import Image

from src.config import ANOMALY_CLASSES, NORMAL_CLASSES


class CLIPInference:

    def __init__(self, model_name="ViT-B/32", scoring_mode="max_max"):
        """
        scoring_mode:
            - "max_max"
            - "max_mean"
            - "mean_mean"
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Store class groups
        self.anomaly_classes = ANOMALY_CLASSES
        self.normal_classes = NORMAL_CLASSES

        self.text_embeddings = None
        self.text_labels = None

        self.scoring_mode = scoring_mode

    # --------------------------------------------------
    # TEXT PROMPTS
    # --------------------------------------------------

    def set_text_prompts(self, texts):
        """
        texts = ANOMALY_CLASSES + NORMAL_CLASSES
        """
        self.text_labels = texts

        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            text_embeddings = self.model.encode_text(tokens)
            self.text_embeddings = F.normalize(text_embeddings, dim=-1)

    # --------------------------------------------------
    # SEGMENT EMBEDDING
    # --------------------------------------------------

    def compute_segment_embedding(self, segment_dir):
        frame_files = sorted(os.listdir(segment_dir))
        frame_embeddings = []

        with torch.no_grad():
            for frame_name in frame_files:
                frame_path = os.path.join(segment_dir, frame_name)

                image = Image.open(frame_path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                emb = self.model.encode_image(image_input)
                frame_embeddings.append(emb)

        frame_embeddings = torch.cat(frame_embeddings, dim=0)

        # Mean pooling across frames
        segment_embedding = frame_embeddings.mean(dim=0, keepdim=True)
        segment_embedding = F.normalize(segment_embedding, dim=-1)

        return segment_embedding

    # --------------------------------------------------
    # CONTRASTIVE SCORING
    # --------------------------------------------------

    def compute_contrastive_score(self, similarities):
        """
        similarities: tensor shape (num_prompts,)
        """

        num_anomaly = len(self.anomaly_classes)

        anomaly_sim = similarities[:num_anomaly]
        normal_sim = similarities[num_anomaly:]

        if self.scoring_mode == "max_max":
            final_score = anomaly_sim.max() - normal_sim.max()

        elif self.scoring_mode == "max_mean":
            final_score = anomaly_sim.max() - normal_sim.mean()

        elif self.scoring_mode == "mean_mean":
            final_score = anomaly_sim.mean() - normal_sim.mean()

        else:
            raise ValueError(f"Unknown scoring mode: {self.scoring_mode}")

        return final_score, anomaly_sim, normal_sim

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def predict_segment(self, segment_dir):
        """
        Returns:
        {
            "predicted_label": "Anomaly" or "Normal",
            "score": float,
            "anomaly_score": float,
            "normal_score": float
        }
        """

        if self.text_embeddings is None:
            raise ValueError("Text prompts not set. Call set_text_prompts() first.")

        segment_embedding = self.compute_segment_embedding(segment_dir)

        similarities = segment_embedding @ self.text_embeddings.T
        similarities = similarities.squeeze(0)

        final_score, anomaly_sim, normal_sim = self.compute_contrastive_score(similarities)

        predicted_label = "Anomaly" if final_score > 0 else "Normal"

        return {
            "predicted_label": predicted_label,
            "score": float(final_score.item()),
            "anomaly_score": float(anomaly_sim.max().item()),
            "normal_score": float(normal_sim.max().item()),
        }

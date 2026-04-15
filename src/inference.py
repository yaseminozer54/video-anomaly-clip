import os
import torch
import clip
import torch.nn.functional as F
from PIL import Image

from src.config import ANOMALY_CLASSES, NORMAL_CLASSES

class CLIPInference:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.anomaly_texts = ANOMALY_CLASSES
        self.normal_texts = NORMAL_CLASSES

        self.anomaly_embeddings = None
        self.normal_embeddings = None

    def set_text_prompts(self):
        with torch.no_grad():
            tokens = clip.tokenize(self.anomaly_texts).to(self.device)
            emb = self.model.encode_text(tokens)
            self.anomaly_embeddings = F.normalize(emb, dim=-1)

        with torch.no_grad():
            tokens = clip.tokenize(self.normal_texts).to(self.device)
            emb = self.model.encode_text(tokens)
            self.normal_embeddings = F.normalize(emb, dim=-1)

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
        
        segment_embedding = frame_embeddings.mean(dim=0, keepdim=True)
        segment_embedding = F.normalize(segment_embedding, dim=-1)

        return segment_embedding

    def predict_segment(self, segment_dir):
        if self.anomaly_embeddings is None or self.normal_embeddings is None:
            raise ValueError("Call set_text_prompts() first")

        segment_embedding = self.compute_segment_embedding(segment_dir)

        anomaly_sim = (segment_embedding @ self.anomaly_embeddings.T).squeeze(0)
        normal_sim = (segment_embedding @ self.normal_embeddings.T).squeeze(0)

        score = anomaly_sim.max() - normal_sim.max()

        return {
            "score": float(score.item()),
            "anomaly_score": float(anomaly_sim.max().item()),
            "normal_score": float(normal_sim.max().item()),
            "anomaly_sim": anomaly_sim.detach().cpu().numpy(),
            "normal_sim": normal_sim.detach().cpu().numpy()
        }

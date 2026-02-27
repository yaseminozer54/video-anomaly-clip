import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentDataset(Dataset):
    """
    Dataset for 16-frame video segments.

    Each sample returns:
        x: Tensor of shape [16, 3, 224, 224]
        label: class name (string)
        meta: dict with segment metadata
    """

    def __init__(self, manifest_csv: str, root_dir: str, return_meta: bool = True):
        self.root_dir = root_dir
        self.df = pd.read_csv(manifest_csv)
        self.return_meta = return_meta

        # CLIP normalization (OpenAI)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seg_path = os.path.join(self.root_dir, row["segment_path"])

        frame_files = sorted(
            f for f in os.listdir(seg_path) if f.lower().endswith(".png")
        )
        if len(frame_files) != 16:
            raise RuntimeError(
                f"Expected 16 frames, got {len(frame_files)} in {seg_path}"
            )

        frames = []
        for fn in frame_files:
            img = Image.open(os.path.join(seg_path, fn)).convert("RGB")
            frames.append(self.transform(img))

        x = torch.stack(frames, dim=0)  # [16, 3, 224, 224]

        label = row["class"]

        if not self.return_meta:
            return x, label

        meta = {
            "segment_uid": row["segment_uid"],
            "split": row["split"],
            "class": row["class"],
            "video_id": row["video_id"],
            "segment_id": row["segment_id"],
            "segment_path": row["segment_path"],
        }

        return x, label, meta

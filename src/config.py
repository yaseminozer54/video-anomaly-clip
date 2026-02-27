# Model
MODEL_NAME = "ViT-B/32"
DEVICE = "auto"

# Dataset
MANIFEST_PATH = "data/segments_manifest_16.csv"
SEGMENTS_ROOT = "data/segments_16"

# Inference
BATCH_SIZE = 8
NUM_WORKERS = 2

# Zero-shot prompts
TEXT_PROMPTS = [
    "a person fighting",
    "a violent action",
    "a robbery happening",
    "people walking normally",
    "a peaceful street scene"
]

ANOMALY_CLASSES = [
    "a person fighting",
    "a violent action",
    "a robbery happening"
]

# Decision rule
THRESHOLD = 0.225
VIDEO_ANOMALY_RATIO = 0.10  # video anomalous if anomaly segment ratio > this value
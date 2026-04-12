MODEL_NAME = "ViT-B/32"

DEVICE = "cuda"

SEGMENT_THRESHOLD = 0.225
VIDEO_THRESHOLD = 0.10

ANOMALY_CLASSES = [
    "a violent crime happening in a public place",
    "people physically attacking each other",
    "a robbery in progress",
    "an explosion causing chaos",
    "a dangerous situation involving violence",
    "criminal activity captured on surveillance camera",
]

NORMAL_CLASSES = [
    "people behaving peacefully in everyday life",
    "a calm and normal public environment",
    "ordinary daily activity with no violence",
    "a safe and uneventful street scene",
]

# Final threshold learned from validation
FINAL_THRESHOLD = 0.2286

AGGREGATION_METHOD = "p90"

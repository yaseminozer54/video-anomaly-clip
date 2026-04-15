MODEL_NAME = "ViT-B/32"

DEVICE = "cuda"

ANOMALY_CLASSES = [
    "CCTV video of police officers arresting a person",
    "CCTV video of a massive explosion and fire",
    "CCTV video of a thief stealing something and committing a crime",
    "CCTV video of a physical fight between people"
]

NORMAL_CLASSES = [
    "CCTV video of a completely normal, safe street with regular traffic",
    "CCTV video of normal pedestrians walking safely and peacefully",
    "CCTV video of an empty, quiet, and safe place"
]

AGGREGATION_METHOD = "max"

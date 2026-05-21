MODEL_NAME = "ViT-L/14"
DEVICE = "cuda"

ANOMALY_CLASSES = [
    "flames and smoke from a fire or arson",
    "people punching and kicking each other in a violent fight",
    "a person breaking into a building or stealing property",
    "police handcuffing or forcibly restraining a person",
    "a serious car crash or road accident",
    "someone vandalizing or destroying property",
    "a person being physically attacked or assaulted",
    "armed robbery or shooting with a weapon",
]

NORMAL_CLASSES = [
    "people walking calmly on a street with no incidents",
    "a quiet and peaceful indoor or outdoor environment",
    "normal daily activities in a safe public space",
    "ordinary traffic and pedestrian movement",
    "people going about their routine without any danger",
    "a calm surveillance scene with no criminal activity",
    "people working shopping or socializing peacefully",
    "an empty or undisturbed area captured on camera",
]
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, uuid

from app.schemas import PredictionResponse, PipelineResponse
from app.inference_service import predict_video, predict_from_frames, predict_pipeline

app = FastAPI(title="CLIP Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "CLIP Anomaly Detection API", "status": "running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    ext       = os.path.splitext(file.filename)[-1]
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict_video(save_path)
    finally:
        os.remove(save_path)

    return result


@app.post("/predict/frames", response_model=PredictionResponse)
async def predict_frames(segment_path: str):
    try:
        result = predict_from_frames(segment_path)
    finally:
        pass
    return result


@app.post("/predict/pipeline", response_model=PipelineResponse)
async def predict_pipeline_endpoint(file: UploadFile = File(...)):
    ext       = os.path.splitext(file.filename)[-1]
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict_pipeline(save_path)
    finally:
        os.remove(save_path)

    return result
from pydantic import BaseModel
from typing import Optional, List

class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_type: Optional[str]
    score_diff: float
    threshold: float
    top_prompt: str
    frame_scores: List[float]      # segment-seviyesi skorlar
    segment_flags: List[bool] = [] # segment kirmizi mi (top-K karar segmenti)
    all_frames: List[str]

class SegmentResult(BaseModel):
    segment_idx: int
    start_sec: float
    end_sec: float
    score: float
    is_anomaly: bool
    anomaly_type: Optional[str]
    top_prompt: str
    frame_scores: List[float]
    all_frames: List[str]

class PipelineResponse(BaseModel):
    segments: List[SegmentResult]
    video_is_anomaly: bool
    anomaly_ratio: float
    n_segments: int
    threshold: float
    score_diff: float = 0.0
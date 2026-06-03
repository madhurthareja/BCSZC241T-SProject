from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    message: str


class MetadataResponse(BaseModel):
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


class DetectResponse(BaseModel):
    confidence: float
    threshold: float
    present: bool
    frame_scores: list[float]


class BiasAnalysisResponse(BaseModel):
    clean_confidence: float
    watermarked_confidence: float
    bias_gap: float


class CalibrateResponse(BaseModel):
    clean_confidence: float
    watermarked_confidence: float
    suggested_threshold: float

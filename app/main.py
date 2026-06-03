from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.schemas import (
    HealthResponse,
    MetadataResponse,
    DetectResponse,
    CalibrateResponse,
    BiasAnalysisResponse,
)

from app.services import (
    save_upload_file,
    get_video_metadata,
    embed_watermark,
    detect_watermark,
    calibrate_threshold,
    analyze_bias,
)


app = FastAPI(
    title="Vidmark FastAPI Backend",
    description="Backend server for video watermark embedding, detection, metadata, calibration, and bias analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
def home():
    return {
        "status": "ok",
        "message": "Vidmark FastAPI backend is running.",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "message": "Server is healthy.",
    }


@app.post("/video/metadata", response_model=MetadataResponse)
def video_metadata(
    video: UploadFile = File(...),
):
    video_path = save_upload_file(video)
    return get_video_metadata(video_path)


@app.post("/watermark/embed")
def watermark_embed(
    video: UploadFile = File(...),
    key: str = Form(...),
    strength: str = Form("medium"),
    repeat: int = Form(1),
):
    video_path = save_upload_file(video)

    output_path = embed_watermark(
        video_path=video_path,
        key=key,
        strength=strength,
        repeat=repeat,
    )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="watermarked_video.mp4",
    )


@app.post("/watermark/detect", response_model=DetectResponse)
def watermark_detect(
    video: UploadFile = File(...),
    key: str = Form(...),
    strength: str = Form("medium"),
    repeat: int = Form(1),
    threshold: float = Form(0.2),
):
    video_path = save_upload_file(video)

    return detect_watermark(
        video_path=video_path,
        key=key,
        strength=strength,
        repeat=repeat,
        threshold=threshold,
    )


@app.post("/watermark/calibrate", response_model=CalibrateResponse)
def watermark_calibrate(
    clean_video: UploadFile = File(...),
    watermarked_video: UploadFile = File(...),
    key: str = Form(...),
    strength: str = Form("medium"),
    repeat: int = Form(1),
):
    clean_path = save_upload_file(clean_video)
    watermarked_path = save_upload_file(watermarked_video)

    return calibrate_threshold(
        clean_video_path=clean_path,
        watermarked_video_path=watermarked_path,
        key=key,
        strength=strength,
        repeat=repeat,
    )


@app.post("/watermark/analyze-bias", response_model=BiasAnalysisResponse)
def watermark_analyze_bias(
    clean_video: UploadFile = File(...),
    watermarked_video: UploadFile = File(...),
    key: str = Form(...),
    strength: str = Form("medium"),
    repeat: int = Form(1),
):
    clean_path = save_upload_file(clean_video)
    watermarked_path = save_upload_file(watermarked_video)

    return analyze_bias(
        clean_video_path=clean_path,
        watermarked_video_path=watermarked_path,
        key=key,
        strength=strength,
        repeat=repeat,
    )

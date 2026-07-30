import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, File, Form, UploadFile
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vidmark.api")


_DEFAULT_CORS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("VIDMARK_CORS_ORIGINS", _DEFAULT_CORS).split(",")
    if o.strip()
]


app = FastAPI(
    title="Vidmark FastAPI Backend",
    description="Backend server for video watermark embedding, detection, metadata, calibration, and bias analysis.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
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
def video_metadata(video: UploadFile = File(...)):
    with save_upload_file(video) as video_path:
        return get_video_metadata(video_path)


@app.post("/watermark/embed")
def watermark_embed(
    background: BackgroundTasks,
    video: UploadFile = File(...),
    key: str = Form(...),
    strength: str = Form("medium"),
    repeat: int = Form(1),
):
    """Embed a watermark and stream the result back.

    The upload lives in a private tempdir for the duration of the embed; the
    generated output is placed in a *separate* tempdir that we register a
    BackgroundTask to delete AFTER the response body has been fully streamed
    to the client. This keeps the surface area stateless — every request
    creates and destroys its own scratch space.
    """
    output_dir = Path(tempfile.mkdtemp(prefix="vidmark-out-"))
    background.add_task(shutil.rmtree, str(output_dir), True)

    with save_upload_file(video) as video_path:
        output_path = embed_watermark(
            video_path=video_path,
            key=key,
            strength=strength,
            repeat=repeat,
            output_dir=output_dir,
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
    with save_upload_file(video) as video_path:
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
    with save_upload_file(clean_video) as clean_path, \
         save_upload_file(watermarked_video) as wm_path:
        return calibrate_threshold(
            clean_video_path=clean_path,
            watermarked_video_path=wm_path,
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
    with save_upload_file(clean_video) as clean_path, \
         save_upload_file(watermarked_video) as wm_path:
        return analyze_bias(
            clean_video_path=clean_path,
            watermarked_video_path=wm_path,
            key=key,
            strength=strength,
            repeat=repeat,
        )

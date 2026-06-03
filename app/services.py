import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from vidmark import Watermarker
from vidmark.io import VideoFile


UPLOAD_DIR = Path("uploaded_files")
OUTPUT_DIR = Path("output_files")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Saves an uploaded video file to disk and returns its path.
    FastAPI gives us UploadFile, but existing vidmark code expects a file path.
    """
    suffix = Path(upload_file.filename).suffix or ".mp4"
    file_path = UPLOAD_DIR / f"{uuid4()}{suffix}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path


def get_video_metadata(video_path: Path) -> dict:
    """
    Uses existing VideoFile class to extract metadata.
    """
    video = VideoFile(video_path)

    try:
        metadata = video.metadata
        return {
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "duration_sec": metadata.duration_sec,
        }
    finally:
        video.close()


def embed_watermark(
    video_path: Path,
    key: str,
    strength: str,
    repeat: int,
) -> Path:
    """
    Uses existing Watermarker.embed() function.
    Returns path of generated watermarked video.
    """
    output_path = OUTPUT_DIR / f"watermarked_{uuid4()}.mp4"

    watermarker = Watermarker(
        key=key,
        strength=strength,
        repeat=repeat,
    )

    watermarker.embed(
        str(video_path),
        str(output_path),
    )

    return output_path


def detect_watermark(
    video_path: Path,
    key: str,
    strength: str,
    repeat: int,
    threshold: float,
) -> dict:
    """
    Uses existing Watermarker.detect() function.
    """
    watermarker = Watermarker(
        key=key,
        strength=strength,
        repeat=repeat,
    )

    result = watermarker.detect(
        str(video_path),
        threshold=threshold,
    )

    return {
        "confidence": result.confidence,
        "threshold": result.threshold,
        "present": result.present,
        "frame_scores": result.scores,
    }


def calibrate_threshold(
    clean_video_path: Path,
    watermarked_video_path: Path,
    key: str,
    strength: str,
    repeat: int,
) -> dict:
    """
    Simple calibration API.
    It detects both clean and watermarked videos.
    Then it suggests a threshold between the two confidence values.
    """
    clean_result = detect_watermark(
        video_path=clean_video_path,
        key=key,
        strength=strength,
        repeat=repeat,
        threshold=0.2,
    )

    watermarked_result = detect_watermark(
        video_path=watermarked_video_path,
        key=key,
        strength=strength,
        repeat=repeat,
        threshold=0.2,
    )

    clean_conf = clean_result["confidence"]
    watermarked_conf = watermarked_result["confidence"]

    suggested_threshold = (clean_conf + watermarked_conf) / 2

    return {
        "clean_confidence": clean_conf,
        "watermarked_confidence": watermarked_conf,
        "suggested_threshold": suggested_threshold,
    }


def analyze_bias(
    clean_video_path: Path,
    watermarked_video_path: Path,
    key: str,
    strength: str,
    repeat: int,
) -> dict:
    """
    Very simple bias analysis.
    It compares confidence on clean video vs watermarked video.
    """
    clean_result = detect_watermark(
        video_path=clean_video_path,
        key=key,
        strength=strength,
        repeat=repeat,
        threshold=0.2,
    )

    watermarked_result = detect_watermark(
        video_path=watermarked_video_path,
        key=key,
        strength=strength,
        repeat=repeat,
        threshold=0.2,
    )

    clean_conf = clean_result["confidence"]
    watermarked_conf = watermarked_result["confidence"]

    return {
        "clean_confidence": clean_conf,
        "watermarked_confidence": watermarked_conf,
        "bias_gap": watermarked_conf - clean_conf,
    }

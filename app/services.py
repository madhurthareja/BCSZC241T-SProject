"""Service layer: bridges FastAPI upload files to vidmark library calls.

Storage model is stateless: every request gets a fresh tempdir that is removed
when the request handler exits, regardless of success or failure. No files
persist between requests. This is the only safe shape for serverless
deployments (Cloud Run, App Runner, etc.) where the container can be killed
between requests.
"""
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from typing import Iterator

from fastapi import UploadFile

from vidmark import Watermarker
from vidmark.io import VideoFile


log = logging.getLogger("vidmark.services")


@contextmanager
def save_upload_file(upload_file: UploadFile) -> Iterator[Path]:
    """Save an uploaded video into a private tempdir; yield its path.

    The tempdir is created here, owned by the caller, and removed when the
    ``with`` block exits — even if the body raises. Use as::

        with save_upload_file(upload) as path:
            ...
    """
    suffix = Path(upload_file.filename or "upload.mp4").suffix or ".mp4"
    request_dir = Path(tempfile.mkdtemp(prefix="vidmark-"))
    file_path = request_dir / f"{uuid4()}{suffix}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        yield file_path
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


def get_video_metadata(video_path: Path) -> dict:
    """Extract metadata via vidmark's VideoFile."""
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
    output_dir: Path,
) -> Path:
    """Embed a watermark; return the path of the generated MP4 inside ``output_dir``.

    The caller (route handler) is responsible for streaming the returned file
    to the client and then deleting the parent directory when done. Splitting
    "where to put the file" from "save_upload_file's tempdir" lets the embed
    response outlive the request tempdir long enough to be streamed.
    """
    watermarker = Watermarker(key=key, strength=strength, repeat=repeat)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"watermarked_{uuid4()}.mp4"

    watermarker.embed(str(video_path), str(output_path))
    return output_path


def detect_watermark(
    video_path: Path,
    key: str,
    strength: str,
    repeat: int,
    threshold: float,
) -> dict:
    """Detect a watermark; return confidence / threshold / per-frame scores."""
    watermarker = Watermarker(key=key, strength=strength, repeat=repeat)
    result = watermarker.detect(str(video_path), threshold=threshold)
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
    """Suggest a threshold between the clean and watermarked confidence."""
    clean_wm = Watermarker(key=key, strength=strength, repeat=repeat)
    wm_wm = Watermarker(key=key, strength=strength, repeat=repeat)

    clean_result = clean_wm.detect(str(clean_video_path), threshold=0.0)
    wm_result = wm_wm.detect(str(watermarked_video_path), threshold=0.0)

    clean_conf = clean_result.confidence
    wm_conf = wm_result.confidence

    if clean_conf <= 0.0 < wm_conf:
        suggested = (clean_conf + wm_conf) / 2
    elif clean_conf > 0 and wm_conf > 0:
        suggested = float((clean_conf * wm_conf) ** 0.5)
    else:
        suggested = max(clean_conf, wm_conf)

    return {
        "clean_confidence": clean_conf,
        "watermarked_confidence": wm_conf,
        "suggested_threshold": float(suggested),
    }


def analyze_bias(
    clean_video_path: Path,
    watermarked_video_path: Path,
    key: str,
    strength: str,
    repeat: int,
) -> dict:
    """Compute the bias gap (watermarked - clean confidence)."""
    clean_wm = Watermarker(key=key, strength=strength, repeat=repeat)
    wm_wm = Watermarker(key=key, strength=strength, repeat=repeat)

    clean_result = clean_wm.detect(str(clean_video_path), threshold=0.0)
    wm_result = wm_wm.detect(str(watermarked_video_path), threshold=0.0)

    return {
        "clean_confidence": clean_result.confidence,
        "watermarked_confidence": wm_result.confidence,
        "bias_gap": float(wm_result.confidence - clean_result.confidence),
    }


__all__ = [
    "save_upload_file",
    "get_video_metadata",
    "embed_watermark",
    "detect_watermark",
    "calibrate_threshold",
    "analyze_bias",
]

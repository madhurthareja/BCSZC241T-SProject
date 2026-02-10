from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .base import VidmarkFile


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


class VideoFile(VidmarkFile):
    def __init__(self, path: str | Path):
        super().__init__(path)
        self.type = "video"
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def frames(self) -> Iterator[np.ndarray]:
        cap = self.load_file()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    def load_file(self) -> cv2.VideoCapture:
        if self._cap is None:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {self.path}")
            self._cap = cap
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self.load_file()

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            cap = self.load_file()

            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration_sec = frame_count / fps if fps > 0 else 0.0

            self._metadata = VideoMetadata(
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration_sec=duration_sec,
            )

        return self._metadata

    def __enter__(self):
        self.load_file()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

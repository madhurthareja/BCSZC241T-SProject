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

    def frames_with_index(self, reference_fps: Optional[float] = None) -> Iterator[tuple[np.ndarray, int]]:
        """Yields (frame, sync_index) where sync_index is recovered from the
        frame's own presentation timestamp rather than read-order position.

        A positional counter desynchronises the moment any upstream frame is
        dropped (every later frame's counter value no longer matches what it
        was at embed time). PTS survives frame drops even when a re-encoder
        renumbers the container's frame count, so re-deriving the index from
        timestamp keeps embedding and detection aligned on the frames that
        actually survived, without needing to know the drop pattern.

        ``reference_fps`` must be the *original* (pre-drop) frame rate used at
        embed time, not this file's own reported rate: a variable-frame-rate
        re-encode after a drop rewrites the container's average FPS metadata
        (e.g. 29.97 -> 15.15 after a 50% drop), and using that corrupted value
        silently recovers the wrong index. Detection therefore needs the
        original fps supplied out-of-band (alongside the key), the same way
        it already needs the key itself.
        """
        cap = self.load_file()
        fps = reference_fps if reference_fps is not None else self.metadata.fps
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            sync_index = round(pos_msec / 1000.0 * fps) if fps > 0 else 0
            yield frame, sync_index

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

"""High-level watermarking façade: ``WatermarkConfig`` (key + strength + the
selector/algorithm pair that drive behaviour) and ``Watermarker`` (the object
that performs ``embed`` and ``detect`` on full video files, including the
sync-search offset sweep used to recover from frame-drop re-encodes)."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from vidmark.core import (
    FrameSelector,
    WatermarkAlgorithm,
    AllFrames,
    DctSpreadSpectrumWatermark,
    NoOpWatermark,
    expand_sequence,
    key_to_bipolar_sequence,
)
from vidmark.io import VideoFile, VideoWriter
from vidmark.utils import DetectionResult

Strength = Literal["low", "medium", "high"]


@dataclass(slots=True)
class WatermarkConfig:
    key: str
    strength: Literal["low", "medium", "high"] = "medium"
    repeat: int = 1

    selector: Optional[FrameSelector] = None
    algorithm: Optional[WatermarkAlgorithm] = None

    alpha: float = field(init=False)

    def __post_init__(self):
        self.alpha = self._strength_to_alpha(self.strength)

        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

        if self.selector is not None and not isinstance(self.selector, FrameSelector):
            raise TypeError("selector must be a FrameSelector")

        if self.algorithm is not None and not isinstance(self.algorithm, WatermarkAlgorithm):
            raise TypeError("algorithm must be a WatermarkAlgorithm")

    @staticmethod
    def _strength_to_alpha(strength: Strength) -> float:
        # DCT-coefficient-domain units (Eq. 2: C'_k = C_k + alpha_k * w_i),
        # matching the alpha in {10, 20, 30} sweep from the paper's own
        # experimental setup — not a fraction of pixel range.
        return {
            "low": 10.0,
            "medium": 20.0,
            "high": 30.0,
        }[strength]

    def watermark_sequence(self):
        sequence = key_to_bipolar_sequence(self.key)
        return expand_sequence(sequence, self.repeat)


class Watermarker:
    def __init__(
            self,
            key: str,
            strength: Strength,
            selector: Optional[FrameSelector] = None,
            algorithm: Optional[WatermarkAlgorithm] = None,
            repeat: int = 1,
    ):
        alpha = WatermarkConfig._strength_to_alpha(strength)
        algorithm = algorithm or DctSpreadSpectrumWatermark(key=key, alpha=alpha, repeat=repeat)
        self.config = WatermarkConfig(
            key=key,
            strength=strength,
            repeat=repeat,
            selector=selector or AllFrames(),
            algorithm=algorithm,
        )

    def embed(self, input_file_path: str, output_file_path: str, crf: int = 18) -> None:
        video = VideoFile(input_file_path)
        meta = video.metadata

        writer = VideoWriter(
            output_file_path,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            crf=crf,
        )

        selector = self.config.selector or AllFrames()
        algorithm = self.config.algorithm or NoOpWatermark()

        try:
            for frame, sync_index in video.frames_with_index():
                if selector.should_watermark(sync_index):
                    frame = algorithm.apply(frame, sync_index)
                writer.write(frame)
        finally:
            writer.close()
            video.close()

    def detect(
        self,
        input_file_path: str,
        threshold: float = 0.2,
        reference_fps: float = None,
        sync_search: int = 0,
    ) -> DetectionResult:
        """``sync_search`` searches a window of constant index offsets and keeps
        whichever gives the strongest aggregate correlation.

        Timestamp-derived indexing (frames_with_index) assumes each frame's
        recovered index matches its true original position, which holds as
        long as the file's timestamp epoch is intact. A re-encoder that drops
        the very first frame of a sequence can reset its output PTS to treat
        the first *surviving* frame as t=0 (observed with ffmpeg's -vsync vfr),
        shifting every recovered index by a constant amount. Correct-key
        correlation peaks sharply at the true offset and is near zero
        elsewhere (the same separation the wrong-key null already relies on),
        so searching a small window and keeping the best-scoring offset
        recovers alignment without needing to know the drop pattern.
        """
        video = VideoFile(input_file_path)
        selector = self.config.selector or AllFrames()
        algorithm = self.config.algorithm or NoOpWatermark()

        try:
            # build_cache does the frame_index-independent work (DCT, sigma) once
            # per frame; detect_from_cache below only redoes the index-dependent
            # LFSR position selection and correlation per candidate offset.
            cached = [
                (algorithm.build_cache(frame), sync_index)
                for frame, sync_index in video.frames_with_index(reference_fps=reference_fps)
            ]
        finally:
            video.close()

        best_result = None
        for offset in range(-sync_search, sync_search + 1):
            scores = []
            for cache, sync_index in cached:
                shifted = sync_index + offset
                if shifted < 0:
                    continue
                if selector.should_watermark(shifted):
                    scores.append(algorithm.detect_from_cache(cache, shifted))
            result = DetectionResult(scores, threshold=threshold)
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        return best_result

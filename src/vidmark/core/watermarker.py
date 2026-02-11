from dataclasses import dataclass, field
from typing import Literal, Optional

from vidmark.core import (
    FrameSelector,
    WatermarkAlgorithm,
    EveryNthFrame,
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
        return {
            "low": 0.03,
            "medium": 0.06,
            "high": 0.1,
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
            selector=selector or EveryNthFrame(n=10),
            algorithm=algorithm,
        )

    def embed(self, input_file_path: str, output_file_path: str) -> None:
        video = VideoFile(input_file_path)
        meta = video.metadata

        writer = VideoWriter(
            output_file_path,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
        )

        selector = self.config.selector or EveryNthFrame(n=10)
        algorithm = self.config.algorithm or NoOpWatermark()

        try:
            for i, frame in enumerate(video.frames()):
                if selector.should_watermark(i):
                    frame = algorithm.apply(frame, i)
                writer.write(frame)
        finally:
            writer.close()
            video.close()

    def detect(self, input_file_path: str, threshold: float = 0.2) -> DetectionResult:
        video = VideoFile(input_file_path)
        selector = self.config.selector or EveryNthFrame(n=10)
        algorithm = self.config.algorithm or NoOpWatermark()

        scores = []

        try:
            for i, frame in enumerate(video.frames()):
                if selector.should_watermark(i):
                    scores.append(algorithm.detect(frame, i))
        finally:
            video.close()

        return DetectionResult(scores, threshold=threshold)

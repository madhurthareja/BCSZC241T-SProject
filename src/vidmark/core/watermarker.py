from dataclasses import dataclass, field
from typing import Literal, Optional

from vidmark.core import FrameSelector, WatermarkAlgorithm, EveryNthFrame, NoOpWatermark
from vidmark.io import VideoFile, VideoWriter

Strength = Literal["low", "medium", "high"]


@dataclass(slots=True)
class WatermarkConfig:
    key: str
    strength: Literal["low", "medium", "high"] = "medium"

    selector: Optional[FrameSelector] = None
    algorithm: Optional[WatermarkAlgorithm] = None

    alpha: float = field(init=False)

    def __post_init__(self):
        self.alpha = self._strength_to_alpha(self.strength)

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


class Watermarker:
    def __init__(
            self,
            key: str,
            strength: Strength,
            selector: Optional[FrameSelector] = None,
            algorithm: Optional[WatermarkAlgorithm] = None
    ):
        self.config = WatermarkConfig(
            key=key,
            strength=strength,
            selector=selector or EveryNthFrame(n=10),
            algorithm=algorithm or NoOpWatermark(),
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

        selector = EveryNthFrame(n=10)
        algorithm = NoOpWatermark()

        try:
            for i, frame in enumerate(video.frames()):
                if selector.should_watermark(i):
                    frame = algorithm.apply(frame)
                writer.write(frame)
        finally:
            writer.close()
            video.close()

    def detect(self, input: str):
        return None

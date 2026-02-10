import numpy as np


class WatermarkAlgorithm:
    def apply(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoOpWatermark(WatermarkAlgorithm):
    def apply(self, frame: np.ndarray) -> np.ndarray:
        return frame

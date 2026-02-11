import hashlib
from typing import List, Tuple

import cv2
import numpy as np


class WatermarkAlgorithm:
    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        raise NotImplementedError

    def detect(self, frame: np.ndarray, frame_index: int) -> float:
        raise NotImplementedError


def key_to_bipolar_sequence(key: str) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    return bits * 2 - 1


def expand_sequence(sequence: np.ndarray, repeat: int) -> np.ndarray:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if repeat == 1:
        return sequence
    return np.tile(sequence, repeat)


class NoOpWatermark(WatermarkAlgorithm):
    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return frame

    def detect(self, frame: np.ndarray, frame_index: int) -> float:
        return 0.0


class DctSpreadSpectrumWatermark(WatermarkAlgorithm):
    def __init__(
        self,
        key: str,
        alpha: float,
        repeat: int = 1,
        block_size: int = 8,
        max_positions: int = 6,
        sigma_floor: float = 5.0,
        sigma_scale: float = 128.0,
        alpha_cap: float = 0.08,
    ):
        self.alpha = float(alpha)
        self.block_size = block_size
        self.max_positions = max_positions
        self.sigma_floor = float(sigma_floor)
        self.sigma_scale = float(sigma_scale)
        self.alpha_cap = float(alpha_cap)
        self.sequence = expand_sequence(key_to_bipolar_sequence(key), repeat)
        self._seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        self._mid_freq_positions = [
            (0, 2), (0, 3), (1, 1), (1, 2), (2, 0), (2, 1),
            (2, 2), (3, 0), (3, 1), (1, 3), (0, 4), (4, 0),
            (2, 3), (3, 2), (4, 1), (1, 4), (3, 3), (4, 2),
        ]

    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        y_plane, cr, cb = self._split_ycrcb(frame)
        y_plane = y_plane.astype(np.float32)

        rng = self._frame_rng(frame_index)
        seq = self.sequence
        seq_len = len(seq)
        seq_index = 0

        for block, (row, col) in self._iter_blocks(y_plane):
            block_f = block - 128.0
            dct = cv2.dct(block_f)

            sigma = float(np.std(block_f))
            if sigma < self.sigma_floor:
                continue
            alpha_k = self.alpha * (sigma / self.sigma_scale)
            alpha_k = min(alpha_k, self.alpha_cap)

            positions = self._positions_for_block(rng)[:self.max_positions]
            for u, v in positions:
                dct[u, v] += alpha_k * seq[seq_index % seq_len]
                seq_index += 1

            idct = cv2.idct(dct) + 128.0
            y_plane[row:row + self.block_size, col:col + self.block_size] = idct

        y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
        return self._merge_ycrcb(y_plane, cr, cb)

    def detect(self, frame: np.ndarray, frame_index: int) -> float:
        coeffs = self.extract_coefficients(frame, frame_index)
        if coeffs.size == 0:
            return 0.0

        seq = self.sequence
        seq_len = len(seq)
        indices = np.arange(coeffs.size) % seq_len
        return float(np.dot(coeffs, seq[indices]) / coeffs.size)

    def extract_coefficients(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        y_plane, _, _ = self._split_ycrcb(frame)
        y_plane = y_plane.astype(np.float32)

        rng = self._frame_rng(frame_index)
        coeffs = []

        for block, _ in self._iter_blocks(y_plane):
            block_f = block - 128.0
            dct = cv2.dct(block_f)

            positions = self._positions_for_block(rng)[:self.max_positions]
            for u, v in positions:
                coeffs.append(float(dct[u, v]))

        return np.array(coeffs, dtype=np.float32)

    def _frame_rng(self, frame_index: int) -> np.random.Generator:
        mix = (frame_index + 1) * 0x9E3779B1
        seed = (self._seed ^ mix) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(seed)

    def _positions_for_block(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        positions = list(self._mid_freq_positions)
        rng.shuffle(positions)
        return positions

    def _iter_blocks(self, plane: np.ndarray):
        height, width = plane.shape
        max_row = height - (height % self.block_size)
        max_col = width - (width % self.block_size)

        for row in range(0, max_row, self.block_size):
            for col in range(0, max_col, self.block_size):
                block = plane[row:row + self.block_size, col:col + self.block_size]
                yield block, (row, col)

    @staticmethod
    def _split_ycrcb(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_plane, cr, cb = cv2.split(ycrcb)
        return y_plane, cr, cb

    @staticmethod
    def _merge_ycrcb(y_plane: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
        ycrcb = cv2.merge((y_plane, cr, cb))
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

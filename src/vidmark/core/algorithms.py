"""Watermark algorithms and the bitstream/sequence helpers they share.

Defines the abstract ``WatermarkAlgorithm`` interface, the key-to-bipolar-
sequence + LFSR helpers used to generate position/coefficient pseudorandom
streams, and the concrete ``DctSpreadSpectrumWatermark`` (spread-spectrum
embedding in 8x8 block DCT coefficients, sigma-gated and variance-adaptive).
Also exposes ``NoOpWatermark`` for non-watermarked baselines and the
``LOW_FREQ_POSITIONS`` coefficient pool used to maximise cross-codec
robustness."""

import hashlib
from typing import List, Tuple

import cv2
import numpy as np


class WatermarkAlgorithm:
    """Interface every concrete watermark algorithm implements.

    ``apply`` embeds the watermark into a frame; ``detect`` returns a single
    correlation score for the frame (typically normalised in [-1, 1]). The
    cache methods let detection amortise expensive frame_index-independent
    work (DCT, sigma, etc.) across many candidate sync offsets."""

    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        raise NotImplementedError

    def detect(self, frame: np.ndarray, frame_index: int) -> float:
        raise NotImplementedError

    def build_cache(self, frame: np.ndarray):
        """Precompute the frame_index-independent part of detection. Default:
        no precomputation available, cache is just the frame itself."""
        return frame

    def detect_from_cache(self, cache, frame_index: int) -> float:
        return self.detect(cache, frame_index)


def key_to_bipolar_sequence(key: str) -> np.ndarray:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    # bits is uint8; bits*2-1 in unsigned arithmetic underflows 0 -> 255
    # instead of -1, so the dtype must widen before the signed subtraction.
    return bits.astype(np.int8) * 2 - 1


def expand_sequence(sequence: np.ndarray, repeat: int) -> np.ndarray:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if repeat == 1:
        return sequence
    return np.tile(sequence, repeat)


class Lfsr:
    """Galois LFSR, maximal-length taps for the given width.

    Deterministic, key-seeded bitstream generator used for coefficient-position
    selection: unlike an opaque CSPRNG, every output bit is a fixed XOR of prior
    state bits, so a detector (or auditor) can replay the exact tap sequence from
    the seed alone.
    """

    # (width, taps) — taps are 1-indexed bit positions per Xilinx XAPP052.
    _TAPS = {
        32: (32, 22, 2, 1),
        64: (64, 63, 61, 60),
    }

    def __init__(self, seed: int, width: int = 64):
        if width not in self._TAPS:
            raise ValueError(f"unsupported LFSR width: {width}")
        self.width = width
        self.taps = self._TAPS[width]
        mask = (1 << width) - 1
        state = seed & mask
        self.state = state if state != 0 else 1  # all-zero state never toggles

    def _next_bit(self) -> int:
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.state >> (tap - 1)) & 1
        self.state = ((self.state << 1) | feedback) & ((1 << self.width) - 1)
        if self.state == 0:
            self.state = 1
        return feedback

    def _randbelow(self, n: int) -> int:
        if n <= 1:
            return 0
        bit_len = n.bit_length()
        while True:
            value = 0
            for _ in range(bit_len):
                value = (value << 1) | self._next_bit()
            if value < n:
                return value

    def shuffled(self, items: list) -> list:
        """Deterministic Fisher-Yates shuffle driven by the LFSR bitstream."""
        result = list(items)
        for i in range(len(result) - 1, 0, -1):
            j = self._randbelow(i + 1)
            result[i], result[j] = result[j], result[i]
        return result


class NoOpWatermark(WatermarkAlgorithm):
    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return frame

    def detect(self, frame: np.ndarray, frame_index: int) -> float:
        return 0.0


DEFAULT_MID_FREQ_POSITIONS = [
    (0, 2), (0, 3), (1, 1), (1, 2), (2, 0), (2, 1),
    (2, 2), (3, 0), (3, 1), (1, 3), (0, 4), (4, 0),
    (2, 3), (3, 2), (4, 1), (1, 4), (3, 3), (4, 2),
]

# Subset with u+v <= 4: drops the six highest-frequency candidates
# ((2,3),(3,2),(4,1),(1,4),(3,3),(4,2)). Lower-frequency coefficients carry
# more of a block's energy and are what every codec's own rate control
# protects most, regardless of its internal transform basis (H.264's near-DCT,
# HEVC's variable integer transforms, AV1's DCT/ADST/identity mix) — so this
# pool trades a bit of spatial spreading for cross-codec transform-basis
# robustness.
LOW_FREQ_POSITIONS = [
    (0, 2), (0, 3), (1, 1), (1, 2), (2, 0), (2, 1),
    (2, 2), (3, 0), (3, 1), (1, 3), (0, 4), (4, 0),
]


class DctSpreadSpectrumWatermark(WatermarkAlgorithm):
    def __init__(
        self,
        key: str,
        alpha: float,
        repeat: int = 1,
        block_size: int = 8,
        max_positions: int = 6,
        sigma_floor: float = 5.0,
        sigma_ref: float = 20.0,
        scale_min: float = 0.5,
        scale_max: float = 1.75,
        alpha_cap: float = 55.0,
        positions: List[Tuple[int, int]] = None,
    ):
        self.alpha = float(alpha)
        self.block_size = block_size
        self.max_positions = max_positions
        self.sigma_floor = float(sigma_floor)
        self.sigma_ref = float(sigma_ref)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.alpha_cap = float(alpha_cap)
        self.sequence = expand_sequence(key_to_bipolar_sequence(key), repeat)
        self._seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        self._mid_freq_positions = positions if positions is not None else list(LOW_FREQ_POSITIONS)

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

            # RNG advances only for blocks that pass the sigma gate; extract_coefficients
            # must use the same ordering so positions stay in sync.
            positions = self._positions_for_block(rng)[:self.max_positions]
            alpha_k = min(self.alpha * self._strength_scale(sigma), self.alpha_cap)

            for u, v in positions:
                dct[u, v] += alpha_k * seq[seq_index % seq_len]
                seq_index += 1

            idct = cv2.idct(dct) + 128.0
            y_plane[row:row + self.block_size, col:col + self.block_size] = idct

        y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
        return self._merge_ycrcb(y_plane, cr, cb)

    def detect(self, frame: np.ndarray, frame_index: int) -> float:
        return self.detect_from_cache(self.build_cache(frame), frame_index)

    def extract_coefficients(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self.extract_coefficients_from_cache(self.build_cache(frame), frame_index)

    def build_cache(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Per-block (DCT, sigma) pairs -- everything detection needs that does
        NOT depend on frame_index. Sigma-gating is index-independent (it's a
        content statistic), so DCT is skipped for blocks that will be skipped
        regardless of which position set a given frame_index selects. Building
        this once and reusing it across sync_search's candidate offsets avoids
        redoing the per-block DCT for every offset trial.
        """
        y_plane, _, _ = self._split_ycrcb(frame)
        y_plane = y_plane.astype(np.float32)

        cache = []
        for block, _ in self._iter_blocks(y_plane):
            block_f = block - 128.0
            sigma = float(np.std(block_f))
            dct = cv2.dct(block_f) if sigma >= self.sigma_floor else None
            cache.append((dct, sigma))
        return cache

    def extract_coefficients_from_cache(self, cache: List[Tuple[np.ndarray, float]], frame_index: int) -> np.ndarray:
        rng = self._frame_rng(frame_index)
        coeffs = []

        for dct, sigma in cache:
            # Mirror apply(): skip low-sigma blocks WITHOUT advancing the RNG so the
            # RNG state (and thus the shuffled positions) stays in sync with embedding.
            if sigma < self.sigma_floor:
                continue

            positions = self._positions_for_block(rng)[:self.max_positions]
            for u, v in positions:
                coeffs.append(float(dct[u, v]))

        return np.array(coeffs, dtype=np.float32)

    def detect_from_cache(self, cache: List[Tuple[np.ndarray, float]], frame_index: int) -> float:
        coeffs = self.extract_coefficients_from_cache(cache, frame_index)
        if coeffs.size == 0:
            return 0.0

        seq = self.sequence
        seq_len = len(seq)
        indices = np.arange(coeffs.size) % seq_len
        centered = coeffs - float(np.mean(coeffs))
        seq_slice = seq[indices]
        numerator = float(np.dot(centered, seq_slice))
        denom = float(np.linalg.norm(centered) * np.linalg.norm(seq_slice))
        if denom == 0.0:
            return 0.0
        return numerator / denom

    def _strength_scale(self, sigma: float) -> float:
        """Bounded variance-adaptive strength: textured blocks embed harder, but
        every block above sigma_floor still gets at least scale_min * alpha —
        an unbounded sigma/sigma_ref ratio produced near-zero alpha_k for blocks
        just above the floor, which codecs quantised away entirely.
        """
        ratio = sigma / self.sigma_ref
        return min(max(ratio, self.scale_min), self.scale_max)

    def _frame_rng(self, frame_index: int) -> Lfsr:
        mix = (frame_index + 1) * 0x9E3779B1
        seed = (self._seed ^ mix) & 0xFFFFFFFFFFFFFFFF
        return Lfsr(seed)

    def _positions_for_block(self, rng: Lfsr) -> List[Tuple[int, int]]:
        return rng.shuffled(self._mid_freq_positions)

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

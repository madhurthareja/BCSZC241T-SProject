import argparse
from typing import Tuple

import numpy as np

from vidmark.core import DctSpreadSpectrumWatermark, EveryNthFrame, WatermarkConfig
from vidmark.io import VideoFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze correlation bias on a video")
    parser.add_argument("input", help="Path to the input video")
    parser.add_argument("--key", required=True, help="Watermark key string")
    parser.add_argument("--strength", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--repeat", type=int, default=1, help="Sequence repeat factor")
    parser.add_argument("--every-n", type=int, default=10, help="Analyze every Nth frame")
    return parser.parse_args()


def summarize(values) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def main() -> None:
    args = parse_args()

    alpha = WatermarkConfig._strength_to_alpha(args.strength)
    algorithm = DctSpreadSpectrumWatermark(
        key=args.key,
        alpha=alpha,
        repeat=args.repeat,
    )
    selector = EveryNthFrame(args.every_n)

    coeff_means = []
    corr_raw = []
    corr_centered = []

    seq = algorithm.sequence
    seq_len = len(seq)

    video = VideoFile(args.input)
    try:
        for i, frame in enumerate(video.frames()):
            if not selector.should_watermark(i):
                continue

            coeffs = algorithm.extract_coefficients(frame, i)
            if coeffs.size == 0:
                continue

            indices = np.arange(coeffs.size) % seq_len
            seq_slice = seq[indices]

            mean_coeff = float(coeffs.mean())
            coeff_means.append(mean_coeff)

            corr_raw.append(float(np.dot(coeffs, seq_slice) / coeffs.size))

            centered = coeffs - mean_coeff
            corr_centered.append(float(np.dot(centered, seq_slice) / coeffs.size))
    finally:
        video.close()

    mean_mu, mean_sigma = summarize(coeff_means)
    raw_mu, raw_sigma = summarize(corr_raw)
    centered_mu, centered_sigma = summarize(corr_centered)

    print("Bias analysis")
    print(f"  Sequence mean: {float(seq.mean()):.6f}")
    print(f"  Coefficient mean: {mean_mu:.6f} / {mean_sigma:.6f}")
    print(f"  Correlation (raw): {raw_mu:.6f} / {raw_sigma:.6f}")
    print(f"  Correlation (centered): {centered_mu:.6f} / {centered_sigma:.6f}")


if __name__ == "__main__":
    main()

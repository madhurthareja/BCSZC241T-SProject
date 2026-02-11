import argparse
from typing import Tuple

import numpy as np

from vidmark import Watermarker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate detection threshold")
    parser.add_argument("--clean", required=True, help="Path to a clean (unwatermarked) video")
    parser.add_argument("--watermarked", required=True, help="Path to a watermarked video")
    parser.add_argument("--key", required=True, help="Watermark key string")
    parser.add_argument("--strength", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--repeat", type=int, default=1, help="Sequence repeat factor")
    parser.add_argument("--std-mult", type=float, default=3.0, help="Clean mean + N*std threshold")
    return parser.parse_args()


def summarize(scores) -> Tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    arr = np.array(scores, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main() -> None:
    args = parse_args()
    wm = Watermarker(key=args.key, strength=args.strength, repeat=args.repeat)

    clean = wm.detect(args.clean)
    watermarked = wm.detect(args.watermarked)

    clean_mean, clean_std = summarize(clean.scores)
    water_mean, water_std = summarize(watermarked.scores)

    threshold = clean_mean + args.std_mult * clean_std

    print("Calibration summary")
    print(f"  Clean mean/std: {clean_mean:.6f} / {clean_std:.6f}")
    print(f"  Watermarked mean/std: {water_mean:.6f} / {water_std:.6f}")
    print(f"  Suggested threshold (mean + {args.std_mult}*std): {threshold:.6f}")

    if water_mean <= threshold:
        print("  Warning: watermarked mean is below threshold; increase alpha or reduce std-mult.")


if __name__ == "__main__":
    main()

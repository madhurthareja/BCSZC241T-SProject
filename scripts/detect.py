import argparse

from vidmark import Watermarker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect watermark presence in a video")
    parser.add_argument("input", help="Path to the input video")
    parser.add_argument("--key", required=True, help="Watermark key string")
    parser.add_argument("--strength", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--repeat", type=int, default=1, help="Sequence repeat factor")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wm = Watermarker(key=args.key, strength=args.strength, repeat=args.repeat)
    result = wm.detect(args.input, threshold=args.threshold)
    print(result)


if __name__ == "__main__":
    main()

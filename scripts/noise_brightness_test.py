import tempfile
from pathlib import Path

import cv2
import numpy as np

from vidmark import AllFrames, Watermarker
from vidmark.core import DctSpreadSpectrumWatermark
from vidmark.io import VideoFile, VideoWriter

KEY = "vidmark-eval-key"
WRONG_KEY = "attacker-guess"
VIDEOS = ["dataset/clean/foreman_cif.mp4", "dataset/clean/akiyo_cif.mp4",
          "dataset/clean/bus_cif.mp4", "dataset/clean/coastguard_cif.mp4",
          "dataset/clean/football_cif.mp4", "dataset/clean/bowing_cif.mp4",
          "dataset/clean/deadline_cif.mp4", "dataset/clean/news_cif.mp4",
          "dataset/clean/paris_cif.mp4", "dataset/clean/silent_cif.mp4",
          "dataset/clean/ice_cif.mp4", "dataset/clean/soccer_cif.mp4",
          "dataset/clean/stefan_cif.mp4", "dataset/clean/hall_monitor_cif.mp4",
          "dataset/clean/crew_cif.mp4", "dataset/clean/highway_cif.mp4",
          "dataset/clean/city_cif.mp4", "dataset/clean/flower_cif.mp4",
          "dataset/clean/mobile_cif.mp4", "dataset/clean/sintel_cif.mp4"]
ALPHA = 20.0


def apply_gaussian_noise(frame: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, size=frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def transform_video(src_path: str, dst_path: str, transform) -> None:
    video = VideoFile(src_path)
    meta = video.metadata
    writer = VideoWriter(dst_path, fps=meta.fps, width=meta.width, height=meta.height, crf=18)
    try:
        for frame in video.frames():
            writer.write(transform(frame))
    finally:
        writer.close()
        video.close()


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for video in VIDEOS:
            name = Path(video).stem
            algo = DctSpreadSpectrumWatermark(key=KEY, alpha=ALPHA, alpha_cap=ALPHA * 1.75)
            wrong_algo = DctSpreadSpectrumWatermark(key=WRONG_KEY, alpha=ALPHA, alpha_cap=ALPHA * 1.75)

            wm_path = f"{tmp}/{name}_wm.mp4"
            Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).embed(
                video, wm_path, crf=18
            )
            baseline = Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).detect(
                wm_path, threshold=0.0
            ).confidence

            rng = np.random.default_rng(hash(name) % (2**31))
            attacks = {
                "gaussian_noise_sigma5": lambda f: apply_gaussian_noise(f, 5.0, rng),
                "brightness_plus10pct": lambda f: apply_brightness(f, 1.10),
                "brightness_minus10pct": lambda f: apply_brightness(f, 0.90),
            }

            for attack_name, transform in attacks.items():
                out_path = f"{tmp}/{name}_{attack_name}.mp4"
                transform_video(wm_path, out_path, transform)

                correct = Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).detect(
                    out_path, threshold=0.0
                )
                wrong = Watermarker(key=WRONG_KEY, strength="medium", selector=AllFrames(), algorithm=wrong_algo).detect(
                    out_path, threshold=0.0
                )
                row = {
                    "video": name, "attack": attack_name, "baseline": round(baseline, 5),
                    "correct_conf": round(correct.confidence, 5),
                    "wrong_conf": round(wrong.confidence, 5),
                    "retained_fraction": round(correct.confidence / baseline, 3) if baseline else 0.0,
                }
                results.append(row)
                print(row)

    print()
    print(f"=== Summary per attack (mean over {len(VIDEOS)} videos) ===")
    for attack_name in ("gaussian_noise_sigma5", "brightness_plus10pct", "brightness_minus10pct"):
        sub = [r for r in results if r["attack"] == attack_name]
        correct_mean = np.mean([r["correct_conf"] for r in sub])
        wrong_mean = np.mean([r["wrong_conf"] for r in sub])
        retained_mean = np.mean([r["retained_fraction"] for r in sub])
        n_win = sum(1 for r in sub if r["correct_conf"] > r["wrong_conf"])
        print(f"{attack_name}: correct_mean={correct_mean:.5f} wrong_mean={wrong_mean:.5f} "
              f"retained_fraction={retained_mean:.2f}x  correct>wrong: {n_win}/{len(sub)}")


if __name__ == "__main__":
    main()

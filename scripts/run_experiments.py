import argparse
import csv
import glob
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from vidmark import AllFrames, Watermarker
from vidmark.core import DctSpreadSpectrumWatermark
from vidmark.io import VideoFile


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - size // 2
    kernel_1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d)


_SSIM_WINDOW = _gaussian_window()


def ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu_a = cv2.filter2D(a, -1, _SSIM_WINDOW)
    mu_b = cv2.filter2D(b, -1, _SSIM_WINDOW)
    mu_a_sq, mu_b_sq, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b

    sigma_a_sq = cv2.filter2D(a ** 2, -1, _SSIM_WINDOW) - mu_a_sq
    sigma_b_sq = cv2.filter2D(b ** 2, -1, _SSIM_WINDOW) - mu_b_sq
    sigma_ab = cv2.filter2D(a * b, -1, _SSIM_WINDOW) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(numerator / denominator))


def quality_between(path_a: str, path_b: str) -> tuple[float, float]:
    va, vb = VideoFile(path_a), VideoFile(path_b)
    psnrs, ssims = [], []
    try:
        for fa, fb in zip(va.frames(), vb.frames()):
            psnrs.append(cv2.PSNR(fa, fb))
            ssims.append(ssim_frame(fa, fb))
    finally:
        va.close()
        vb.close()
    return float(np.mean(psnrs)) if psnrs else 0.0, float(np.mean(ssims)) if ssims else 0.0


def run(dataset_dir: str, alphas: list, crfs: list, key: str, wrong_key: str, out_csv: str) -> None:
    videos = sorted(glob.glob(os.path.join(dataset_dir, "*.mp4")))
    if not videos:
        raise SystemExit(f"no .mp4 files found in {dataset_dir}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for video in videos:
            name = Path(video).stem
            for alpha in alphas:
                algo = DctSpreadSpectrumWatermark(key=key, alpha=alpha, alpha_cap=alpha * 1.75)
                wrong_algo = DctSpreadSpectrumWatermark(key=wrong_key, alpha=alpha, alpha_cap=alpha * 1.75)

                embed_only_path = os.path.join(tmp, f"{name}_a{alpha}_embed.mp4")
                t0 = time.time()
                Watermarker(key=key, strength="medium", selector=AllFrames(), algorithm=algo).embed(
                    video, embed_only_path, crf=0
                )
                embed_secs = time.time() - t0
                psnr_embed, ssim_embed = quality_between(video, embed_only_path)

                for crf in crfs:
                    out_path = os.path.join(tmp, f"{name}_a{alpha}_crf{crf}.mp4")
                    Watermarker(key=key, strength="medium", selector=AllFrames(), algorithm=algo).embed(
                        video, out_path, crf=crf
                    )
                    correct = Watermarker(
                        key=key, strength="medium", selector=AllFrames(), algorithm=algo
                    ).detect(out_path, threshold=0.0)
                    wrong = Watermarker(
                        key=wrong_key, strength="medium", selector=AllFrames(), algorithm=wrong_algo
                    ).detect(out_path, threshold=0.0)

                    row = {
                        "video": name,
                        "alpha": alpha,
                        "crf": crf,
                        "psnr_embed_only_db": round(psnr_embed, 2),
                        "ssim_embed_only": round(ssim_embed, 4),
                        "correct_key_confidence": round(correct.confidence, 5),
                        "wrong_key_confidence": round(wrong.confidence, 5),
                        "embed_seconds": round(embed_secs, 2),
                    }
                    rows.append(row)
                    print(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PSNR/SSIM/detection sweep across alpha x CRF")
    parser.add_argument("--dataset-dir", default="dataset/clean")
    parser.add_argument("--alphas", nargs="+", type=float, default=[10.0, 20.0, 30.0])
    parser.add_argument("--crfs", nargs="+", type=int, default=[18, 23, 28, 35, 40])
    parser.add_argument("--key", default="vidmark-eval-key")
    parser.add_argument("--wrong-key", default="attacker-guess")
    parser.add_argument("--out", default="dataset/results.csv")
    args = parser.parse_args()
    run(args.dataset_dir, args.alphas, args.crfs, args.key, args.wrong_key, args.out)

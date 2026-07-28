"""Head-to-head: invisible-watermark (dwtDctSvd, classical DWT-DCT-SVD) vs our
own system, through the exact same evaluation harness (same 5 videos, same
CRF sweep, same cross-codec transcodes). invisible-watermark is a
fixed-message system (embed known bits, decode and check exact match /
bit accuracy) rather than correlation-against-threshold, so "detection rate"
here means message-match rate, not a correlation statistic -- reported on
its own terms, not forced into our metric.
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder

from vidmark.io import VideoFile, VideoWriter

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
CRFS = [18, 23, 28, 35, 40]
MESSAGE = "test"
METHOD = "dwtDctSvd"


def _gaussian_window(size=11, sigma=1.5):
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return np.outer(k, k)


_SSIM_WINDOW = _gaussian_window()


def ssim_frame(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.filter2D(a, -1, _SSIM_WINDOW)
    mu_b = cv2.filter2D(b, -1, _SSIM_WINDOW)
    mu_a_sq, mu_b_sq, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sigma_a_sq = cv2.filter2D(a ** 2, -1, _SSIM_WINDOW) - mu_a_sq
    sigma_b_sq = cv2.filter2D(b ** 2, -1, _SSIM_WINDOW) - mu_b_sq
    sigma_ab = cv2.filter2D(a * b, -1, _SSIM_WINDOW) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(np.mean(num / den))


def embed_video(src_path, dst_path, encoder):
    video = VideoFile(src_path)
    meta = video.metadata
    writer = VideoWriter(dst_path, fps=meta.fps, width=meta.width, height=meta.height, crf=18)
    try:
        for frame in video.frames():
            writer.write(encoder.encode(frame, METHOD))
    finally:
        writer.close()
        video.close()


def decode_match_rate(path, decoder):
    video = VideoFile(path)
    n_total, n_match = 0, 0
    try:
        for frame in video.frames():
            n_total += 1
            try:
                decoded = decoder.decode(frame, METHOD)
                if decoded == MESSAGE.encode("utf-8"):
                    n_match += 1
            except Exception:
                pass
    finally:
        video.close()
    return n_match / n_total if n_total else 0.0


def main():
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", MESSAGE.encode("utf-8"))
    decoder = WatermarkDecoder("bytes", 32)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for video in VIDEOS:
            name = Path(video).stem
            wm_path = f"{tmp}/{name}_iw.mp4"
            embed_video(video, wm_path, encoder)

            # embedding-only PSNR/SSIM
            src, wm = VideoFile(video), VideoFile(wm_path)
            psnrs, ssims = [], []
            try:
                for fa, fb in zip(src.frames(), wm.frames()):
                    psnrs.append(cv2.PSNR(fa, fb))
                    ssims.append(ssim_frame(fa, fb))
            finally:
                src.close()
                wm.close()
            psnr_embed, ssim_embed = float(np.mean(psnrs)), float(np.mean(ssims))
            print(f"{name}: embed-only PSNR={psnr_embed:.2f} SSIM={ssim_embed:.4f}", flush=True)

            for crf in CRFS:
                out_path = f"{tmp}/{name}_crf{crf}.mp4"
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", wm_path,
                     "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                     "-pix_fmt", "yuv420p", out_path],
                    check=True,
                )
                match_rate = decode_match_rate(out_path, decoder)
                row = {"video": name, "crf": crf, "psnr_embed": psnr_embed,
                       "ssim_embed": ssim_embed, "match_rate": match_rate}
                rows.append(row)
                print(f"  CRF={crf}: message match rate={match_rate:.2%}", flush=True)

    print()
    print(f"=== Summary: mean message-match rate across {len(VIDEOS)} videos ===")
    for crf in CRFS:
        vals = [r["match_rate"] for r in rows if r["crf"] == crf]
        print(f"CRF={crf}: mean match rate={np.mean(vals):.2%}")

    import csv
    with open("dataset/invisible_watermark_results.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print("saved dataset/invisible_watermark_results.csv")


if __name__ == "__main__":
    main()

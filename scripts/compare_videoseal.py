"""Head-to-head: Video Seal (Meta, video-native neural watermarking, 256-bit
fixed message) vs our own system, through the same evaluation harness (same
5 videos, same CRF sweep). Bit accuracy (fraction of 256 bits correctly
recovered) is Video Seal's native metric, reported on its own terms.

Two packaging bugs in the PyPI release (videoseal==1.0.1) are worked around
here rather than patched upstream: (1) setup_model_from_model_card() resolves
"videoseal/cards" relative to CWD, not the installed package location -- we
chdir into site-packages for the load() call only; (2) configs/attenuation.yaml
is referenced but not bundled in the PyPI sdist -- fetched from the GitHub
repo and placed at the same CWD-relative path the code expects.
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

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


def load_model():
    import videoseal
    site_packages = os.path.dirname(os.path.dirname(videoseal.__file__))
    cwd = os.getcwd()
    os.chdir(site_packages)
    try:
        model = videoseal.load("videoseal")
    finally:
        os.chdir(cwd)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device), device


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


def embed_video(model, device, src_path, dst_path):
    video = VideoFile(src_path)
    meta = video.metadata
    frames_bgr = list(video.frames())
    video.close()

    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        outputs = model.embed(tensor, is_video=True)
    watermarked = outputs["imgs_w"].clamp(0, 1).cpu().numpy()
    msg = outputs["msgs"][0].cpu().numpy()  # the 256-bit message actually embedded

    writer = VideoWriter(dst_path, fps=meta.fps, width=meta.width, height=meta.height, crf=18)
    try:
        for i in range(watermarked.shape[0]):
            frame_rgb = (watermarked[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.close()
    return msg


def bit_accuracy(model, device, path, true_msg):
    video = VideoFile(path)
    frames_bgr = list(video.frames())
    video.close()
    if not frames_bgr:
        return 0.0

    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        detected = model.detect(tensor)
    preds = (detected["preds"][:, 1:] > 0).float().cpu().numpy()  # (N, 256)
    true = (true_msg > 0.5).astype(np.float32)
    acc_per_frame = (preds == true).mean(axis=1)
    return float(acc_per_frame.mean())


def main():
    model, device = load_model()
    print(f"model loaded on device={device}", flush=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for video in VIDEOS:
            name = Path(video).stem
            wm_path = f"{tmp}/{name}_vs.mp4"

            t0 = time.perf_counter()
            msg = embed_video(model, device, video, wm_path)
            embed_time = time.perf_counter() - t0

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
            print(f"{name}: embed-only PSNR={psnr_embed:.2f} SSIM={ssim_embed:.4f} "
                  f"embed_time={embed_time:.1f}s", flush=True)

            for crf in CRFS:
                out_path = f"{tmp}/{name}_crf{crf}.mp4"
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", wm_path,
                     "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                     "-pix_fmt", "yuv420p", out_path],
                    check=True,
                )
                acc = bit_accuracy(model, device, out_path, msg)
                row = {"video": name, "crf": crf, "psnr_embed": psnr_embed,
                       "ssim_embed": ssim_embed, "bit_accuracy": acc}
                rows.append(row)
                print(f"  CRF={crf}: bit accuracy={acc:.2%}", flush=True)

    print()
    print(f"=== Summary: mean bit accuracy across {len(VIDEOS)} videos ===")
    for crf in CRFS:
        vals = [r["bit_accuracy"] for r in rows if r["crf"] == crf]
        print(f"CRF={crf}: mean bit accuracy={np.mean(vals):.2%}")

    import csv
    with open("dataset/videoseal_results.csv", "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print("saved dataset/videoseal_results.csv")


if __name__ == "__main__":
    main()

"""Peak process memory (RSS) for Video Seal: model load + one detect pass,
using the already-persisted watermarked video so no re-embedding is needed.
Reported alongside our own system's memory footprint in the paper's
multi-dimensional comparison table.
"""
import os
import resource

import cv2
import numpy as np
import torch

from vidmark.io import VideoFile


def rss_mb():
    # ru_maxrss is bytes on macOS, KB on Linux
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val / (1024 * 1024) if os.uname().sysname == "Darwin" else val / 1024


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


def main():
    print(f"RSS before model load: {rss_mb():.1f} MB", flush=True)
    model, device = load_model()
    print(f"device={device}", flush=True)
    print(f"RSS after model load: {rss_mb():.1f} MB", flush=True)

    wm_path = "dataset/videoseal_embedded/foreman_cif_wm.mp4"
    video = VideoFile(wm_path)
    frames_bgr = list(video.frames())
    video.close()
    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        _ = model.detect(tensor)
    print(f"RSS after detect pass: {rss_mb():.1f} MB", flush=True)
    print(f"PEAK RSS (ru_maxrss): {rss_mb():.1f} MB", flush=True)


if __name__ == "__main__":
    main()

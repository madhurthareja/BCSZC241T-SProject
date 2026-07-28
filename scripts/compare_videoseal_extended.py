"""Extend the Video Seal comparison to cross-codec transcoding and frame
dropping, matching exactly the tests already run for our own system
(scripts/cross_codec_ab_test.py, scripts/frame_drop_test.py). Embeddings are
persisted to dataset/videoseal_embedded/ so re-running only the new tests
doesn't repeat Video Seal's (highly variable, sometimes 250s+) embed cost.

Both result CSVs are written incrementally with a resume-skip so an
interrupted run doesn't lose already-computed rows -- only the embed step
was previously persisted; the cross-codec/frame-drop detection loops were
not, and a partial run was lost once already.
"""
import csv
import os
import subprocess
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
PERSIST_DIR = "dataset/videoseal_embedded"
CROSS_CODEC = {
    "hevc": ["-c:v", "libx265", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"],
    "vp9":  ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p"],
    "av1":  ["-c:v", "libsvtav1", "-crf", "32", "-pix_fmt", "yuv420p"],
}
DROP_FRACS = [0.05, 0.10, 0.20, 0.30]

CC_CSV = "dataset/videoseal_crosscodec.csv"
CC_FIELDS = ["video", "codec", "bit_accuracy"]
FD_CSV = "dataset/videoseal_framedrop.csv"
FD_FIELDS = ["video", "drop_frac", "baseline", "bit_accuracy", "retained_x"]


def _load_done(csv_path, key_fields):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {tuple(r[k] for k in key_fields) for r in csv.DictReader(f)}


def _append_row(csv_path, fieldnames, row):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


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
    msg = outputs["msgs"][0].cpu().numpy()

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
    preds = (detected["preds"][:, 1:] > 0).float().cpu().numpy()
    true = (true_msg > 0.5).astype(np.float32)
    return float((preds == true).mean(axis=1).mean())


def ensure_embeds(model, device):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    entries = []
    for video in VIDEOS:
        name = Path(video).stem
        wm_path = f"{PERSIST_DIR}/{name}_wm.mp4"
        msg_path = f"{PERSIST_DIR}/{name}_msg.npy"
        if os.path.exists(wm_path) and os.path.exists(msg_path):
            print(f"{name}: reusing persisted embed", flush=True)
            msg = np.load(msg_path)
        else:
            t0 = time.perf_counter()
            msg = embed_video(model, device, video, wm_path)
            print(f"{name}: embedded in {time.perf_counter()-t0:.1f}s", flush=True)
            np.save(msg_path, msg)
        entries.append((name, wm_path, msg))
    return entries


def run_cross_codec(model, device, entries):
    done = _load_done(CC_CSV, ["video", "codec"])
    if done:
        print(f"resuming cross-codec: {len(done)} (video, codec) pairs already done", flush=True)
    for name, wm_path, msg in entries:
        for codec, args in CROSS_CODEC.items():
            if (name, codec) in done:
                continue
            out_path = f"/tmp/{name}_{codec}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wm_path, *args, out_path], check=True)
            acc = bit_accuracy(model, device, out_path, msg)
            _append_row(CC_CSV, CC_FIELDS, {"video": name, "codec": codec, "bit_accuracy": acc})
            print(f"{name} {codec}: bit accuracy={acc:.2%}", flush=True)

    with open(CC_CSV, newline="") as f:
        cc_rows = list(csv.DictReader(f))
    print()
    print("=== Cross-codec summary: mean bit accuracy ===")
    for codec in CROSS_CODEC:
        vals = [float(r["bit_accuracy"]) for r in cc_rows if r["codec"] == codec]
        print(f"{codec}: mean={np.mean(vals):.2%}")


def run_frame_drop(model, device, entries):
    done = _load_done(FD_CSV, ["video", "drop_frac"])
    if done:
        print(f"resuming frame-drop: {len(done)} (video, drop_frac) pairs already done", flush=True)
    for name, wm_path, msg in entries:
        remaining = [d for d in DROP_FRACS if (name, str(d)) not in done]
        if not remaining:
            continue

        video = VideoFile(wm_path)
        n_frames = video.metadata.frame_count
        video.close()
        baseline = bit_accuracy(model, device, wm_path, msg)

        for drop_frac in remaining:
            rng = np.random.default_rng(abs(hash((name, drop_frac))) % (2**31))
            keep_mask = rng.random(n_frames) >= drop_frac
            kept = np.where(keep_mask)[0]
            expr = "+".join(f"eq(n\\,{i})" for i in kept)
            dropped_path = f"/tmp/{name}_drop{drop_frac}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", wm_path,
                 "-vf", f"select='{expr}'", "-vsync", "vfr",
                 "-c:v", "libx264", "-crf", "18", dropped_path],
                check=True,
            )
            acc = bit_accuracy(model, device, dropped_path, msg)
            _append_row(FD_CSV, FD_FIELDS, {
                "video": name, "drop_frac": drop_frac, "baseline": baseline,
                "bit_accuracy": acc, "retained_x": acc / baseline if baseline else 0.0,
            })
            print(f"{name} drop={drop_frac:.0%}: bit accuracy={acc:.2%} "
                  f"({acc/baseline:.2f}x baseline)", flush=True)

    with open(FD_CSV, newline="") as f:
        fd_rows = list(csv.DictReader(f))
    print()
    print("=== Frame-drop summary: mean retained fraction ===")
    for drop_frac in DROP_FRACS:
        vals = [float(r["retained_x"]) for r in fd_rows if r["drop_frac"] == str(drop_frac)]
        print(f"drop={drop_frac:.0%}: mean retained={np.mean(vals):.2f}x")


def main():
    model, device = load_model()
    print(f"model loaded on device={device}", flush=True)
    entries = ensure_embeds(model, device)

    run_cross_codec(model, device, entries)
    run_frame_drop(model, device, entries)


if __name__ == "__main__":
    main()

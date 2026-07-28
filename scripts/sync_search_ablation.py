"""Ablation over sync_search window width: does detection confidence plateau
once W covers the true offset, and how does runtime scale with W?
"""
import subprocess
import time

import numpy as np

from vidmark import AllFrames, Watermarker
from vidmark.core import DctSpreadSpectrumWatermark
from vidmark.io import VideoFile

KEY = "vidmark-eval-key"
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
WINDOWS = [0, 1, 3, 5, 7, 10]
DROP_FRAC = 0.20


def main():
    print(f"{'video':>15} {'W':>3} {'confidence':>10} {'x_baseline':>10} {'time_s':>8}")
    agg = {w: {"conf_x": [], "time": []} for w in WINDOWS}

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for video in VIDEOS:
            name = video.split("/")[-1].replace(".mp4", "")
            algo = DctSpreadSpectrumWatermark(key=KEY, alpha=20.0, alpha_cap=35.0)
            wm_path = f"{tmp}/{name}_wm.mp4"
            Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).embed(
                video, wm_path, crf=18
            )
            src_fps = VideoFile(video).metadata.fps
            n_frames = VideoFile(wm_path).metadata.frame_count
            baseline = Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).detect(
                wm_path, threshold=0.0
            ).confidence

            # force frame 0 to be dropped so a real offset is needed
            rng = np.random.default_rng(abs(hash(name)) % (2**31))
            keep_mask = rng.random(n_frames) >= DROP_FRAC
            keep_mask[0] = False
            kept = np.where(keep_mask)[0]
            expr = "+".join(f"eq(n\\,{i})" for i in kept)
            dropped_path = f"{tmp}/{name}_dropped.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", wm_path,
                 "-vf", f"select='{expr}'", "-vsync", "vfr",
                 "-c:v", "libx264", "-crf", "18", dropped_path],
                check=True,
            )

            for w in WINDOWS:
                t0 = time.perf_counter()
                result = Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).detect(
                    dropped_path, threshold=0.0, reference_fps=src_fps, sync_search=w
                )
                elapsed = time.perf_counter() - t0
                conf_x = result.confidence / baseline if baseline else 0.0
                agg[w]["conf_x"].append(conf_x)
                agg[w]["time"].append(elapsed)
                print(f"{name:>15} {w:>3} {result.confidence:>10.5f} {conf_x:>10.2f} {elapsed:>8.2f}", flush=True)

    print()
    print(f"=== Summary: mean over {len(VIDEOS)} videos (frame 0 forced dropped, 20% total drop) ===")
    for w in WINDOWS:
        print(f"W={w:>2}: mean_confidence_x={np.mean(agg[w]['conf_x']):.2f}  "
              f"mean_time_s={np.mean(agg[w]['time']):.2f}  "
              f"n_offsets_tried={2*w+1}")


if __name__ == "__main__":
    main()

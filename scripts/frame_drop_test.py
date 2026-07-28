import csv
import os
import subprocess
import tempfile
from pathlib import Path

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
ALPHA = 20.0
DROP_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
# +-2 was the largest offset observed in validation (a single early frame drop
# shifting the vfr PTS epoch); +-5 keeps comfortable margin at ~3x lower cost
# than the +-15 window used for initial validation.
SYNC_SEARCH_WINDOW = 5
OUT_CSV = "dataset/frame_drop_results.csv"
FIELDNAMES = ["video", "drop_frac", "frame0_kept", "baseline",
              "no_ref", "no_search", "with_search",
              "no_ref_x", "no_search_x", "with_search_x"]


def drop_frames(src: str, dst: str, drop_frac: float, seed: int, n_frames: int) -> bool:
    # ffmpeg's random(N) expression argument is a state-slot index (0-9), not
    # a seed -- passing arbitrary "seed" values silently produces the exact
    # same output every time (verified: identical file hashes across calls).
    # Generate the keep mask in Python instead, with a real RNG, and pass it
    # to ffmpeg as an explicit frame-number selection.
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(n_frames) >= drop_frac
    kept_indices = np.where(keep_mask)[0]
    if len(kept_indices) == 0:
        return False
    expr = "+".join(f"eq(n\\,{i})" for i in kept_indices)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"select='{expr}'", "-vsync", "vfr",
         "-c:v", "libx264", "-crf", "18", dst],
        check=True,
    )
    return bool(kept_indices[0] == 0)


def _load_done() -> set:
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV, newline="") as f:
        return {(r["video"], r["drop_frac"]) for r in csv.DictReader(f)}


def _append_row(row: dict) -> None:
    is_new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main():
    done = _load_done()
    if done:
        print(f"resuming: {len(done)} (video, drop_frac) pairs already in {OUT_CSV}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        for video in VIDEOS:
            name = Path(video).stem
            remaining = [d for d in DROP_FRACTIONS if (name, str(d)) not in done]
            if not remaining:
                print(f"{name}: all drop fractions already done, skipping embed", flush=True)
                continue

            algo = DctSpreadSpectrumWatermark(key=KEY, alpha=ALPHA, alpha_cap=ALPHA * 1.75)
            wm_path = f"{tmp}/{name}_wm.mp4"
            Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).embed(
                video, wm_path, crf=18
            )
            src_fps = VideoFile(video).metadata.fps
            n_frames = VideoFile(wm_path).metadata.frame_count
            baseline = Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).detect(
                wm_path, threshold=0.0
            ).confidence

            for drop_frac in remaining:
                dropped_path = f"{tmp}/{name}_drop{drop_frac}.mp4"
                seed = abs(hash((name, drop_frac))) % (2**31)
                frame0_kept = drop_frames(wm_path, dropped_path, drop_frac, seed, n_frames)

                no_ref = Watermarker(
                    key=KEY, strength="medium", selector=AllFrames(), algorithm=algo
                ).detect(dropped_path, threshold=0.0).confidence
                no_search = Watermarker(
                    key=KEY, strength="medium", selector=AllFrames(), algorithm=algo
                ).detect(dropped_path, threshold=0.0, reference_fps=src_fps, sync_search=0).confidence
                with_search = Watermarker(
                    key=KEY, strength="medium", selector=AllFrames(), algorithm=algo
                ).detect(dropped_path, threshold=0.0, reference_fps=src_fps, sync_search=SYNC_SEARCH_WINDOW).confidence

                row = {
                    "video": name, "drop_frac": drop_frac, "frame0_kept": frame0_kept,
                    "baseline": baseline, "no_ref": no_ref, "no_search": no_search, "with_search": with_search,
                    "no_ref_x": no_ref / baseline if baseline else 0.0,
                    "no_search_x": no_search / baseline if baseline else 0.0,
                    "with_search_x": with_search / baseline if baseline else 0.0,
                }
                _append_row(row)
                print(f"{name} drop={drop_frac:.0%} frame0_kept={frame0_kept}: baseline={baseline:.5f} "
                      f"no_ref={row['no_ref_x']:.2f}x no_search={row['no_search_x']:.2f}x "
                      f"with_search={row['with_search_x']:.2f}x", flush=True)

    print(flush=True)
    print("=== Summary: mean retained fraction of baseline confidence ===", flush=True)
    with open(OUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for drop_frac in DROP_FRACTIONS:
        sub = [r for r in rows if r["drop_frac"] == str(drop_frac)]
        for key in ("no_ref_x", "no_search_x", "with_search_x"):
            vals = [float(r[key]) for r in sub]
            print(f"drop={drop_frac:.0%} {key}: mean={np.mean(vals):.2f}x median={np.median(vals):.2f}x", flush=True)


if __name__ == "__main__":
    main()

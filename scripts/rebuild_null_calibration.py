"""Rebuild the pooled wrong-key null from scratch for the n=20 video corpus,
rather than layering more trials onto dataset/pooled_null_scores*.npy, which
were built from the original 5-video grid and are no longer a valid base.

Pools two sources, exactly as the original n=1005 calibration did:
  1. The 300 wrong_key_confidence values already in dataset/results.csv
     (20 videos x 3 alpha x 5 CRF, from run_experiments.py).
  2. 62 additional distinct wrong keys per (video, alpha=20, CRF) file,
     against the same 3 conditions used originally (CRF 18/28/40), for all
     20 videos -- reusing the 15 embeds already persisted in
     dataset/null_calibration/ from the 5-video run and creating the
     remaining 45 (15 new videos x 3 conditions).
"""
import csv
import os
from pathlib import Path

import numpy as np

from vidmark import AllFrames, Watermarker
from vidmark.core import DctSpreadSpectrumWatermark

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
CONDITIONS = [(20.0, 18), (20.0, 28), (20.0, 40)]
PERSIST_DIR = "dataset/null_calibration"
N_WRONG_KEYS_PER_FILE = 62


def ensure_embeds():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    paths = []
    for video in VIDEOS:
        name = Path(video).stem
        for alpha, crf in CONDITIONS:
            out_path = f"{PERSIST_DIR}/{name}_a{alpha}_crf{crf}.mp4"
            if not os.path.exists(out_path):
                algo = DctSpreadSpectrumWatermark(key=KEY, alpha=alpha, alpha_cap=alpha * 1.75)
                Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).embed(
                    video, out_path, crf=crf
                )
                print(f"embedded {out_path}", flush=True)
            else:
                print(f"reusing existing {out_path}", flush=True)
            paths.append(out_path)
    return paths


def main():
    grid_rows = list(csv.DictReader(open("dataset/results.csv")))
    grid_null = np.array([float(r["wrong_key_confidence"]) for r in grid_rows])
    print(f"base grid pool: n={len(grid_null)} (from dataset/results.csv)")

    files = ensure_embeds()
    expanded_scores = []
    for out_path in files:
        for trial_idx in range(N_WRONG_KEYS_PER_FILE):
            wrong_key = f"attacker-guess-{trial_idx}"
            wrong_algo = DctSpreadSpectrumWatermark(key=wrong_key, alpha=20.0, alpha_cap=20.0 * 1.75)
            result = Watermarker(
                key=wrong_key, strength="medium", selector=AllFrames(), algorithm=wrong_algo
            ).detect(out_path, threshold=0.0)
            expanded_scores.append(result.confidence)
        print(f"{out_path}: {N_WRONG_KEYS_PER_FILE} wrong-key trials done "
              f"(running total: {len(expanded_scores)})", flush=True)

    expanded_scores = np.array(expanded_scores)
    combined = np.concatenate([grid_null, expanded_scores])

    print()
    print(f"=== Expanded-only pool: n={len(expanded_scores)} ===")
    print(f"mean={expanded_scores.mean():.6f}  std={expanded_scores.std(ddof=1):.6f}")
    print()
    print(f"=== Combined pooled null (n=20 videos): n={len(combined)} ===")
    print(f"mean={combined.mean():.6f}  std={combined.std(ddof=1):.6f}")
    print(f"tau (99th percentile): {np.quantile(combined, 0.99):.6f}")
    print(f"min={combined.min():.6f}  max={combined.max():.6f}")

    np.save("dataset/pooled_null_scores_n20.npy", combined)
    print("saved dataset/pooled_null_scores_n20.npy")


if __name__ == "__main__":
    main()

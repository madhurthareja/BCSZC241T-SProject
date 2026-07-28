"""Detection throughput and peak process memory for our own system, matching
the methodology of measure_videoseal_memory.py: single-threaded, no sync
search (W=0), on a representative already-embedded clip. Embedding throughput
is reported directly from dataset/results.csv's embed_seconds column (no
separate measurement needed, that figure comes from the same 20-video grid
run this script is meant to accompany).
"""
import os
import resource
import time

from vidmark import AllFrames, Watermarker
from vidmark.core import DctSpreadSpectrumWatermark


def rss_mb():
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val / (1024 * 1024) if os.uname().sysname == "Darwin" else val / 1024


def main():
    key = "vidmark-eval-key"
    algo = DctSpreadSpectrumWatermark(key=key, alpha=20.0, alpha_cap=35.0)
    watermarker = Watermarker(key=key, strength="medium", selector=AllFrames(), algorithm=algo)

    src = "dataset/clean/foreman_cif.mp4"
    embed_path = "/tmp/foreman_runtime_probe.mp4"
    watermarker.embed(src, embed_path, crf=18)

    print(f"RSS before detect: {rss_mb():.1f} MB", flush=True)
    n_trials = 10
    times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        result = watermarker.detect(embed_path, threshold=0.0)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"trial {i}: {dt:.3f}s  confidence={result.confidence:.5f}", flush=True)
    print(f"RSS after detect: {rss_mb():.1f} MB", flush=True)
    print(f"PEAK RSS: {rss_mb():.1f} MB", flush=True)

    mean_t = sum(times) / len(times)
    print(f"\nmean detect time: {mean_t:.3f}s over {n_trials} trials (90 frames)")
    print(f"detect fps: {90/mean_t:.2f}")


if __name__ == "__main__":
    main()

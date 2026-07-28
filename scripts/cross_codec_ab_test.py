"""A/B test: default (18-position) vs low-frequency-biased (12-position)
coefficient pool, both re-encoded H.264 -> HEVC/VP9/AV1, to see whether
biasing toward lower frequencies improves cross-codec watermark survival.
"""
import csv
import subprocess
import tempfile
from pathlib import Path

from vidmark import AllFrames, Watermarker
from vidmark.core import DctSpreadSpectrumWatermark
from vidmark.core.algorithms import DEFAULT_MID_FREQ_POSITIONS, LOW_FREQ_POSITIONS

CODECS = {
    "h265_hevc": ["-c:v", "libx265", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"],
    "vp9":       ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p"],
    "av1":       ["-c:v", "libsvtav1", "-crf", "32", "-pix_fmt", "yuv420p"],
}

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
KEY = "vidmark-eval-key"
WRONG_KEY = "attacker-guess"


def transcode(src, dst, args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, *args, dst], check=True)


def run():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for video in VIDEOS:
            name = Path(video).stem
            for pool_name, positions in [("default_18pos", DEFAULT_MID_FREQ_POSITIONS), ("lowfreq_12pos", LOW_FREQ_POSITIONS)]:
                algo = DctSpreadSpectrumWatermark(key=KEY, alpha=ALPHA, alpha_cap=ALPHA * 1.75, positions=positions)
                wrong_algo = DctSpreadSpectrumWatermark(key=WRONG_KEY, alpha=ALPHA, alpha_cap=ALPHA * 1.75, positions=positions)

                wm_h264 = f"{tmp}/{name}_{pool_name}_wm.mp4"
                Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).embed(video, wm_h264, crf=18)

                for codec_name, args in CODECS.items():
                    out_path = f"{tmp}/{name}_{pool_name}_{codec_name}.mp4"
                    transcode(wm_h264, out_path, args)

                    correct = Watermarker(key=KEY, strength="medium", selector=AllFrames(), algorithm=algo).detect(out_path, threshold=0.0)
                    wrong = Watermarker(key=WRONG_KEY, strength="medium", selector=AllFrames(), algorithm=wrong_algo).detect(out_path, threshold=0.0)
                    margin = correct.confidence - wrong.confidence

                    row = {
                        "video": name, "pool": pool_name, "codec": codec_name,
                        "correct_conf": round(correct.confidence, 5),
                        "wrong_conf": round(wrong.confidence, 5),
                        "margin": round(margin, 5),
                    }
                    rows.append(row)
                    print(row)

    with open("dataset/cross_codec_ab_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import numpy as np
    for pool in ("default_18pos", "lowfreq_12pos"):
        margins = [r["margin"] for r in rows if r["pool"] == pool]
        print(f"\n{pool}: mean margin={np.mean(margins):.5f}  median={np.median(margins):.5f}  "
              f"fraction positive={np.mean([m > 0 for m in margins]):.2f}")


if __name__ == "__main__":
    run()

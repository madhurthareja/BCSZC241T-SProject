"""Trim newly downloaded Xiph/derf raw y4m sequences to 90 frames (3s @ ~30fps)
and re-encode losslessly (H.264 CRF 0) as dataset/clean/*.mp4, matching exactly
the conversion already applied to the original 5 test videos (verified via
ffprobe: 352x288, 30000/1001 fps, 90 frames, H.264 profile High 4:4:4
Predictive / CRF 0, yuv420p).

Sintel is animation content sourced at 480p and is explicitly resized to CIF
(352x288) to match the rest of the corpus; this resize is disclosed in the
paper rather than silently applied.
"""
import glob
import os
import subprocess

RAW_DIR = "dataset/raw"
CLEAN_DIR = "dataset/clean"

STANDARD = ["bowing", "deadline", "news", "paris", "silent", "ice", "soccer",
            "stefan", "hall_monitor", "crew", "highway", "city", "flower", "mobile"]


def convert_standard(name):
    src = f"{RAW_DIR}/{name}_cif.y4m"
    dst = f"{CLEAN_DIR}/{name}_cif.mp4"
    if os.path.exists(dst):
        print(f"{name}: already converted, skipping")
        return
    if not os.path.exists(src):
        print(f"{name}: raw file missing, skipping")
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-frames:v", "90", "-c:v", "libx264", "-crf", "0",
         "-pix_fmt", "yuv420p", dst],
        check=True,
    )
    print(f"{name}: converted -> {dst}")


def convert_sintel():
    src = f"{RAW_DIR}/sintel_trailer_2k_480p24.y4m"
    dst = f"{CLEAN_DIR}/sintel_cif.mp4"
    if os.path.exists(dst):
        print("sintel: already converted, skipping")
        return
    if not os.path.exists(src):
        print("sintel: raw file missing, skipping")
        return
    # The trailer opens with a long fade-in from black (confirmed by probing:
    # frames up to ~t=18s are at or near mean=0); starting the clip at t=0
    # silently produced 72 pure-black frames, not genuine animated content.
    # t=24s onward is a stable, well-lit shot with no cuts (probed at 0.5s
    # resolution through t=27.5s), used here instead. Scale to 352x288 (CIF),
    # re-encode losslessly, matching the rest of the corpus.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "24", "-i", src,
         "-frames:v", "72", "-vf", "scale=352:288",
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", dst],
        check=True,
    )
    print(f"sintel: converted (t=24-27.5s, downscaled from 480p, 72 frames @24fps) -> {dst}")


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    for name in STANDARD:
        convert_standard(name)
    convert_sintel()


if __name__ == "__main__":
    main()

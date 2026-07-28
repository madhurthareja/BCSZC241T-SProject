"""Compute every derived statistic the paper's Experiments section needs,
from the already-collected n=20 raw data, in one place. Read-only; no new
experiments run here.
"""
import csv
from collections import defaultdict

import numpy as np

results = list(csv.DictReader(open("dataset/results.csv")))
for r in results:
    for k in ("alpha", "crf", "psnr_embed_only_db", "ssim_embed_only",
              "correct_key_confidence", "wrong_key_confidence", "embed_seconds"):
        r[k] = float(r[k])

print("=== Perceptual quality by alpha (Table quality) ===")
for alpha in sorted(set(r["alpha"] for r in results)):
    sub = [r for r in results if r["alpha"] == alpha]
    # one psnr/ssim value per video (constant across CRF), so dedupe by video
    per_video = {}
    for r in sub:
        per_video[r["video"]] = (r["psnr_embed_only_db"], r["ssim_embed_only"])
    psnrs = [v[0] for v in per_video.values()]
    ssims = [v[1] for v in per_video.values()]
    print(f"alpha={alpha}: PSNR mean={np.mean(psnrs):.2f}dB range=[{min(psnrs):.2f},{max(psnrs):.2f}] "
          f"SSIM mean={np.mean(ssims):.4f}  n_videos={len(per_video)}")

print()
print("=== Embedding time (Setup section) ===")
embed_secs = [r["embed_seconds"] for r in results]
print(f"mean={np.mean(embed_secs):.2f}s  min={min(embed_secs):.2f}s  max={max(embed_secs):.2f}s  n={len(embed_secs)}")
fps_vals = [90 / e for e in embed_secs]
print(f"embed fps: mean={np.mean(fps_vals):.2f}  range=[{min(fps_vals):.2f},{max(fps_vals):.2f}]")

print()
print("=== Wrong-key null (grid only, n=300) sanity vs pooled n=4020 ===")
neg_all = np.array([r["wrong_key_confidence"] for r in results])
print(f"grid-only: mean={neg_all.mean():.6f} std={neg_all.std(ddof=1):.6f}")
pooled = np.load("dataset/pooled_null_scores_n20.npy")
print(f"pooled n=20: n={len(pooled)} mean={pooled.mean():.6f} std={pooled.std(ddof=1):.6f} "
      f"tau99={np.quantile(pooled, 0.99):.6f}")

print()
print("=== Cross-codec A/B (Table crosscodec) ===")
cc_rows = list(csv.DictReader(open("dataset/cross_codec_ab_results.csv")))
for r in cc_rows:
    r["margin"] = float(r["margin"])
for pool in ("default_18pos", "lowfreq_12pos"):
    for codec in ("h265_hevc", "vp9", "av1"):
        sub = [r for r in cc_rows if r["pool"] == pool and r["codec"] == codec]
        margins = [r["margin"] for r in sub]
        n_correct = sum(1 for m in margins if m > 0)
        print(f"{pool} {codec}: mean_margin={np.mean(margins):.4f}  {n_correct}/{len(margins)} correct")

print()
print("=== invisible-watermark quality (Table iwbaseline) ===")
iw_rows = list(csv.DictReader(open("dataset/invisible_watermark_results.csv")))
for r in iw_rows:
    r["psnr_embed"] = float(r["psnr_embed"])
    r["ssim_embed"] = float(r["ssim_embed"])
per_video = {}
for r in iw_rows:
    per_video[r["video"]] = (r["psnr_embed"], r["ssim_embed"])
psnrs = [v[0] for v in per_video.values()]
ssims = [v[1] for v in per_video.values()]
print(f"mean PSNR={np.mean(psnrs):.2f}dB  mean SSIM={np.mean(ssims):.4f}  n_videos={len(per_video)}")

print()
print("=== Video Seal quality (Table videoseal) ===")
vs_rows = list(csv.DictReader(open("dataset/videoseal_results.csv")))
for r in vs_rows:
    r["psnr_embed"] = float(r["psnr_embed"])
    r["ssim_embed"] = float(r["ssim_embed"])
per_video = {}
for r in vs_rows:
    per_video[r["video"]] = (r["psnr_embed"], r["ssim_embed"])
psnrs = [v[0] for v in per_video.values()]
ssims = [v[1] for v in per_video.values()]
print(f"mean PSNR={np.mean(psnrs):.2f}dB  mean SSIM={np.mean(ssims):.4f}  n_videos={len(per_video)}")

import csv

import matplotlib.pyplot as plt
import numpy as np

TAU = 0.005022  # n=4020 pooled threshold (20-video corpus)

rows = list(csv.DictReader(open("dataset/results.csv")))
for r in rows:
    for k in ("alpha", "crf", "correct_key_confidence"):
        r[k] = float(r[k])

alphas = [10, 20, 30]
crfs = [18, 23, 28, 35, 40]
rates = np.zeros((len(alphas), len(crfs)))
for i, alpha in enumerate(alphas):
    for j, crf in enumerate(crfs):
        sub = [r for r in rows if r["alpha"] == alpha and r["crf"] == crf]
        rates[i, j] = 100 * np.mean([r["correct_key_confidence"] > TAU for r in sub])

fig, ax = plt.subplots(figsize=(4.6, 2.6))
im = ax.imshow(rates, cmap="Greys", vmin=0, vmax=100, aspect="auto")

ax.set_xticks(range(len(crfs)))
ax.set_xticklabels(crfs)
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels(alphas)
ax.set_xlabel("H.264 CRF")
ax.set_ylabel(r"$\alpha$")
ax.set_title(f"Detection rate (%), $\\tau$={TAU}, $n$=4020 null", fontsize=9)

for i in range(len(alphas)):
    for j in range(len(crfs)):
        val = rates[i, j]
        color = "white" if val > 50 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=8)

fig.colorbar(im, ax=ax, label="Detection rate (%)", fraction=0.046, pad=0.04)
fig.tight_layout()

import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/detection_rate_heatmap.pdf")
print("wrote figures/detection_rate_heatmap.pdf")

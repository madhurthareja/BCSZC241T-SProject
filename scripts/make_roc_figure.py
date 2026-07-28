import csv

import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open("dataset/results.csv")))
for r in rows:
    for k in ("alpha", "crf", "correct_key_confidence", "wrong_key_confidence"):
        r[k] = float(r[k])

neg_all = np.array([r["wrong_key_confidence"] for r in rows])


def roc_points(pos, neg):
    thresholds = np.sort(np.unique(np.concatenate([pos, neg])))[::-1]
    tpr, fpr = [0.0], [0.0]
    for t in thresholds:
        tpr.append(np.mean(pos >= t))
        fpr.append(np.mean(neg >= t))
    tpr.append(1.0)
    fpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def auc(pos, neg):
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(all_scores)
    ranks = np.empty(len(all_scores))
    ranks[order] = np.arange(1, len(all_scores) + 1)
    sorted_scores, sorted_ranks = all_scores[order], ranks[order].astype(float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            sorted_ranks[i : j + 1] = sorted_ranks[i : j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


fig, ax = plt.subplots(figsize=(4.2, 3.6))
for crf, style in [(18, "-"), (28, "--")]:
    pos = np.array([r["correct_key_confidence"] for r in rows if r["crf"] == crf])
    fpr, tpr = roc_points(pos, neg_all)
    a = auc(pos, neg_all)
    ax.plot(fpr, tpr, style, label=f"CRF {crf} (AUC={a:.3f})", color="black")

ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8, label="chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", fontsize=8)
ax.set_title(f"ROC: H.264 CRF 18 vs. CRF 28 (H0 pooled, n={len(neg_all)})", fontsize=9)
fig.tight_layout()

import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/roc_curve.pdf")
print("wrote figures/roc_curve.pdf")

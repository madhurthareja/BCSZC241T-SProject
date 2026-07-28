"""Bootstrap confidence intervals for AUC (per CRF) and Wilson-score intervals
for detection-rate percentages (Table II), computed from data already
collected -- no new experiments needed.
"""
import csv
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open("dataset/results.csv")))
for r in rows:
    for k in ("alpha", "crf", "correct_key_confidence", "wrong_key_confidence"):
        r[k] = float(r[k])

neg_all = np.array([r["wrong_key_confidence"] for r in rows])
tau = 0.005022  # n=4020 pooled threshold (20-video corpus)


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


def bootstrap_auc_ci(pos, neg, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.array(pos), np.array(neg)
    boot_aucs = []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        boot_aucs.append(auc(bp, bn))
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return lo, hi


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


print(f"=== AUC with 95% bootstrap CI, per CRF (pooled over alpha, n=60 pos vs n={len(neg_all)} neg) ===")
for crf in sorted(set(r["crf"] for r in rows)):
    pos = [r["correct_key_confidence"] for r in rows if r["crf"] == crf]
    point = auc(pos, neg_all)
    lo, hi = bootstrap_auc_ci(pos, neg_all)
    print(f"CRF={int(crf):>2}: AUC={point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print()
print(f"=== Detection rate with 95% Wilson CI, per (alpha, CRF), tau={tau} ===")
by_ac = defaultdict(list)
for r in rows:
    by_ac[(r["alpha"], r["crf"])].append(r)
for alpha in sorted(set(r["alpha"] for r in rows)):
    for crf in sorted(set(r["crf"] for r in rows)):
        sub = by_ac[(alpha, crf)]
        successes = sum(1 for r in sub if r["correct_key_confidence"] > tau)
        n = len(sub)
        lo, hi = wilson_ci(successes, n)
        print(f"alpha={alpha:>5.1f} CRF={int(crf):>2}: {successes}/{n} = {100*successes/n:.0f}%  "
              f"95% CI [{100*lo:.0f}%, {100*hi:.0f}%]")

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open("dataset/frame_drop_results.csv")))
for r in rows:
    r["drop_frac"] = float(r["drop_frac"])
    for k in ("no_ref_x", "no_search_x", "with_search_x"):
        r[k] = float(r[k])

drop_fracs = sorted(set(r["drop_frac"] for r in rows))
by_frac = defaultdict(lambda: defaultdict(list))
for r in rows:
    for k in ("no_ref_x", "no_search_x", "with_search_x"):
        by_frac[r["drop_frac"]][k].append(r[k])

labels = [f"{int(f*100)}%" for f in drop_fracs]
means = {k: [np.mean(by_frac[f][k]) for f in drop_fracs] for k in ("no_ref_x", "no_search_x", "with_search_x")}

x = np.arange(len(drop_fracs))
width = 0.26

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.bar(x - width, means["no_ref_x"], width, label="No fix", color="0.75")
ax.bar(x, means["no_search_x"], width, label="Timestamp fix only", color="0.45")
ax.bar(x + width, means["with_search_x"], width, label="+ Sync search", color="0.1")

ax.axhline(1.0, linestyle=":", color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Frame drop rate")
ax.set_ylabel("Detection confidence\n(fraction of no-drop baseline)")
ax.set_title(f"Frame-drop recovery: mean over {len(set(r['video'] for r in rows))} videos", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()

import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/frame_drop_recovery.pdf")
print("wrote figures/frame_drop_recovery.pdf")

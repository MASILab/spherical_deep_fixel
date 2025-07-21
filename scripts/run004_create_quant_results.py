import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib
import numpy as np
import matplotlib.ticker as mticker
from statannotations.Annotator import Annotator

# Scale everything by 0.5
matplotlib.rcParams['font.size'] = 8

total_results = pd.read_csv("total_results.csv")
sensitivity_results = pd.read_csv("sensitivity_analysis_large.csv", dtype={"true_n_fibers": str})

# Convert true_angular_separation from radians to degrees
sensitivity_results["true_angular_separation"] = np.round(sensitivity_results["true_angular_separation"] * 180 / np.pi, 2)

# fig, ax = plt.subplots(1, 3, figsize=(6.5, 4))
fig, ax = plt.subplots(figsize=(6.5, 4))

# Plot stripplot colored by experiment and split by n_fibers
sns.boxplot(x="experiment", y="acc", data=total_results, showfliers=False, ax=ax, color="white", linewidth=1.5, order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
sns.stripplot(x="experiment", y="acc", hue="true_n_fibers", data=total_results, jitter=0.2, alpha=0.3, ax=ax, palette="colorblind", order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
ax.set_ylabel("ACC")
ax.set_xlabel("Method")
ax.legend(title="Number of fibers", loc='upper left')
legend = ax.get_legend()
for lh in legend.legend_handles:
    lh.set_alpha(1)

# Add Wilcoxon test between pairs
pairs = [("DeepFixel MLP", "fod2fixel"), ("DeepFixel Spherical CNN", "fod2fixel"), ("DeepFixel MLP", "DeepFixel Spherical CNN")]
annotator = Annotator(ax, pairs, data=total_results, x="experiment", y="acc")
annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
annotator.apply_and_annotate()

# Rename x-axis labels
ax.set_xticklabels(["FISSILE", "fod2fixel", "DeepFixel\nMLP", "DeepFixel\nSpherical\nCNN"])

# Also get median and IQR
print("Total results median and IQR")
print(total_results.groupby("experiment")["acc"].agg(["median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)]))

plt.tight_layout()

fig.savefig("/home/local/VANDERBILT/saundam1/Pictures/deepfixel/spie_2025/fig_quant_results.png", dpi=600)

plt.show()
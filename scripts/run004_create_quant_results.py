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
from deep_fixel.utils import angular_separation

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
annotator.configure(test="Wilcoxon", text_format="star", loc="inside", verbose=2)
annotator.apply_and_annotate()

# Rename x-axis labels
ax.set_xticklabels(["FISSILE (ours)", "fod2fixel", "DeepFixel\nMLP (ours)", "DeepFixel\nSpherical\nCNN (ours)"])

# Also get median and IQR
print("Total results median and IQR")
print(total_results.groupby("experiment")["acc"].agg(["median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)]))

# Get effect size
fod2fixel = total_results[total_results["experiment"] == "fod2fixel"]["acc"]
deepfixel_mlp = total_results[total_results["experiment"] == "DeepFixel MLP"]["acc"]
deepfixel_scnn = total_results[total_results["experiment"] == "DeepFixel Spherical CNN"]["acc"]

print("Effect size between DeepFixel MLP and fod2fixel:", (np.mean(deepfixel_mlp) - np.mean(fod2fixel)) / np.std(np.concatenate([deepfixel_mlp, fod2fixel])))
print("Effect size between DeepFixel Spherical CNN and fod2fixel:", (np.mean(deepfixel_scnn) - np.mean(fod2fixel)) / np.std(np.concatenate([deepfixel_scnn, fod2fixel])))
print("Effect size between DeepFixel MLP and DeepFixel Spherical CNN:", (np.mean(deepfixel_mlp) - np.mean(deepfixel_scnn)) / np.std(np.concatenate([deepfixel_mlp, deepfixel_scnn])))

plt.tight_layout()

fig.savefig("/home/local/VANDERBILT/saundam1/Pictures/deepfixel/spie_2025/fig_quant_results.png", dpi=600)

plt.show()

# Remove NaN rows
total_results = total_results.dropna(subset=["angular_error"])
total_results["angular_error"] = total_results["angular_error"] * 180 / np.pi  # Convert to degrees

print("Median and IQR of angular error")
print(total_results.groupby("experiment")["angular_error"].agg(["median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)]))
print("Median and IQR of volume fraction error")
print(total_results.groupby("experiment")["volume_fraction_error"].agg(["median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)]))

# Plot angulaar error as stripplot colored by experiment and split by n_fibers
fig, ax = plt.subplots(figsize=(6.5, 4))
sns.boxplot(x="experiment", y="angular_error", data=total_results, showfliers=False, ax=ax, color="white", linewidth=1.5, order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
sns.stripplot(x="experiment", y="angular_error", hue="true_n_fibers", data=total_results, jitter=0.2, alpha=0.3, ax=ax, palette="colorblind", order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
ax.set_ylabel("Angular Error (degrees)")
ax.set_xlabel("Method")
ax.legend(title="Number of fibers", loc='upper left')
legend = ax.get_legend()
for lh in legend.legend_handles:
    lh.set_alpha(1)

# Add Wilcoxon test between pairs
pairs = [("DeepFixel MLP", "fod2fixel"), ("DeepFixel Spherical CNN", "fod2fixel"), ("DeepFixel MLP", "DeepFixel Spherical CNN"), ("FISSILE", "fod2fixel"), ("FISSILE", "DeepFixel MLP"), ("FISSILE", "DeepFixel Spherical CNN")]
annotator = Annotator(ax, pairs, data=total_results, x="experiment", y="angular_error")
annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
annotator.apply_and_annotate()

fig, ax = plt.subplots(figsize=(6.5, 4))
sns.boxplot(x="experiment", y="volume_fraction_error", data=total_results, showfliers=False, ax=ax, color="white", linewidth=1.5, order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
sns.stripplot(x="experiment", y="volume_fraction_error", hue="true_n_fibers", data=total_results, jitter=0.2, alpha=0.3, ax=ax, palette="colorblind", order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
ax.set_ylabel("Volume Fraction Error")
ax.set_xlabel("Method")
ax.legend(title="Number of fibers", loc='upper left')
legend = ax.get_legend()
for lh in legend.legend_handles:
    lh.set_alpha(1)

pairs = [("DeepFixel MLP", "fod2fixel"), ("DeepFixel Spherical CNN", "fod2fixel"), ("DeepFixel MLP", "DeepFixel Spherical CNN"), ("FISSILE", "fod2fixel"), ("FISSILE", "DeepFixel MLP"), ("FISSILE", "DeepFixel Spherical CNN")]
annotator = Annotator(ax, pairs, data=total_results, x="experiment", y="volume_fraction_error")
annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
annotator.apply_and_annotate()

plt.show()
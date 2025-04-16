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
fig = plt.figure(figsize=(6.5, 4))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])
ax = [ax1, ax2, ax3]

# Plot stripplot colored by experiment and split by n_fibers
sns.boxplot(x="experiment", y="acc", data=total_results, showfliers=False, ax=ax[0], color="white", linewidth=1.5, order=["fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
sns.stripplot(x="experiment", y="acc", hue="true_n_fibers", data=total_results, jitter=0.2, alpha=0.3, ax=ax[0], palette="colorblind", order=["fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"])
ax[0].set_ylabel("ACC")
ax[0].set_xlabel("")
ax[0].legend(title="Number of fibers", loc='lower center')
legend = ax[0].get_legend()
for lh in legend.legend_handles:
    lh.set_alpha(1)

# Add Wilcoxon test between pairs
pairs = [("DeepFixel MLP", "fod2fixel"), ("DeepFixel Spherical CNN", "fod2fixel"), ("DeepFixel MLP", "DeepFixel Spherical CNN")]
annotator = Annotator(ax[0], pairs, data=total_results, x="experiment", y="acc")
annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=2)
annotator.apply_and_annotate()

# Rename x-axis labels
ax[0].set_xticklabels(["fod2fixel", "DeepFixel MLP", "DeepFixel\nSpherical CNN"])

# Plot sensitivity to volume fraction and angular separation in separate plots
sns.lineplot(x="vol_frac", y="acc", hue="method", hue_order=["fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"], data=sensitivity_results[sensitivity_results["experiment"] == "vol"], estimator="median", errorbar=("pi", 50), err_style="band", markeredgecolor=None, ax=ax[1], marker='.', markersize=4)
ax[1].set_yscale('log')
ax[1].set_xlabel("Volume fraction")
ax[1].set_ylabel("ACC (log scale)")
ax[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax[1].yaxis.get_major_formatter().set_scientific(False)

# Put legend in lower right
legend = ax[1].get_legend()
legend.remove()
ax[1].legend(title="Method", loc='lower right')

sns.lineplot(x="true_angular_separation", y="acc", hue="method", hue_order=["fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"], data=sensitivity_results[sensitivity_results["experiment"] == "angle"], estimator="median", errorbar=("pi", 50), err_style="band", ax=ax[2], markeredgecolor=None, marker='.', markersize=4)
ax[2].set_yscale('log')
ax[2].set_xlabel("Angular separation (degrees)")
ax[2].set_ylabel("ACC (log scale)")
ax[2].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax[2].yaxis.get_major_formatter().set_scientific(False)

# Put legend in lower right
legend = ax[2].get_legend()
legend.remove()
ax[2].legend(title="Method", loc='lower right')

# # Save tidy results
# total_results.to_csv("total_results.csv", index=False)
# sensitivity_results.to_csv("sensitivity_results.csv", index=False)

plt.tight_layout()

plt.show()
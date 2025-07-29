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

fig, ax = plt.subplots(2, 1, figsize=(6.5, 4))

# Plot sensitivity to volume fraction and angular separation in separate plots
sns.lineplot(x="vol_frac", y="acc", hue="method", hue_order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"], data=sensitivity_results[sensitivity_results["experiment"] == "vol"], estimator="median", errorbar=("pi", 50), err_style="band", markeredgecolor=None, ax=ax[0], marker='.', markersize=4)
ax[0].set_yscale('log')
ax[0].set_xlabel("Volume fraction")
ax[0].set_ylabel("ACC (log scale)")
ax[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax[0].yaxis.get_major_formatter().set_scientific(False)

# Put legend in lower right
legend = ax[0].get_legend()
handles, labels = ax[0].get_legend_handles_labels()
legend.remove()
ax[0].legend(handles, ["FISSILE (ours)", "fod2fixel", "DeepFixel MLP (ours)", "DeepFixel Spherical CNN (ours)"], title="Method", loc='lower right')

sns.lineplot(x="true_angular_separation", y="acc", hue="method", hue_order=["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"], data=sensitivity_results[sensitivity_results["experiment"] == "angle"], estimator="median", errorbar=("pi", 50), err_style="band", ax=ax[1], markeredgecolor=None, marker='.', markersize=4)
ax[1].set_yscale('log')
ax[1].set_xlabel("Angular separation (degrees)")
ax[1].set_ylabel("ACC (log scale)")
ax[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
ax[1].yaxis.get_major_formatter().set_scientific(False)

# Put legend in lower right
legend = ax[1].get_legend()
handles, labels = ax[1].get_legend_handles_labels()
legend.remove()
ax[1].legend(handles, ["FISSILE (ours)", "fod2fixel", "DeepFixel MLP (ours)", "DeepFixel Spherical CNN (ours)"], title="Method", loc='lower right')

# # Save tidy results
# total_results.to_csv("total_results.csv", index=False)
# sensitivity_results.to_csv("sensitivity_results.csv", index=False)

plt.tight_layout()

fig.savefig("/home/local/VANDERBILT/saundam1/Pictures/deepfixel/spie_2025/fig_sensitivity.png", dpi=600)

plt.show()
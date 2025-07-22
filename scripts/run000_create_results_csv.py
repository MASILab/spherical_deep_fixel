import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

results_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/results")
results_dir_local = Path("../outputs")

experiments = ["fissile", "fod2fixel_2", "deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03", "deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03"]
experiment_names = ["FISSILE", "fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"]

total_results = []
for experiment, experiment_name in zip(experiments, experiment_names):
    if experiment.startswith("deepfixel"):
        results_df = pd.read_csv(results_dir_local / experiment / f"test_results.csv", dtype={"true_n_fibers": str})
    else:
        results_df = pd.read_csv(results_dir / experiment / f"test_results.csv", dtype={"true_n_fibers": str})
    results_df["experiment"] = experiment_name

    # Loop through voxel index, find average acc of each and add to each row
    for voxel_index in results_df["voxel_index"].unique():
        results_df.loc[results_df["voxel_index"] == voxel_index, "acc_mean"] = results_df[results_df["voxel_index"] == voxel_index]["acc"].mean()
        # Weight mean by true volume fraction
        # results_df.loc[results_df["voxel_index"] == voxel_index, "acc_mean"] = np.average(results_df[results_df["voxel_index"] == voxel_index]["acc"], weights=results_df[results_df["voxel_index"] == voxel_index]["true_v"])

    total_results.append(results_df)

total_results = pd.concat(total_results)

# Add angular error
from deep_fixel.utils import angular_separation
total_results["angular_error"] = total_results.apply(lambda row: angular_separation([row["true_theta"], row["true_phi"]], [row["est_theta_matched"], row["est_phi_matched"]]), axis=1)
total_results["volume_fraction_error"] = total_results.apply(lambda row: np.sqrt((row["true_v"] - row["est_v_matched"]) ** 2), axis=1)

total_results.to_csv("total_results.csv")
print(total_results)

# Get wilcoxon between fod2fixel and crossing fiber net
fod2fixel_results = total_results[total_results["experiment"] == "fod2fixel"]
deepfixel_mlp_results = total_results[total_results["experiment"] == "DeepFixel MLP"]
deepfixel_scnn_results = total_results[total_results["experiment"] == "DeepFixel Spherical CNN"]
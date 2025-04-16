import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib
from torch.utils.data import DataLoader
from deep_fixel.dataset import GeneratedMeshDataset
from deep_fixel.models import CrossingFiberMeshMLP, CrossingFiberMeshSCNN
from deep_fixel.utils import plot_odf, plot_mesh, pdf2odfs, match_odfs, plot_multiple_odf
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.reconst.shm import sh_to_sf, convert_sh_descoteaux_tournier, gen_dirac, sph_harm_ind_list
import torch
import numpy as np
import nibabel as nib

total_results = pd.read_csv("total_results.csv")

# Scale everything by 0.5
matplotlib.rcParams['font.size'] = 8

# sns.set_theme('paper', font_scale=0.8)
results_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/results")

# Get voxel_index/fiber_index for 10th, 50th and 90th percentile of ACC for FISSILE
fissile_results = total_results[total_results["experiment"] == "DeepFixel Spherical CNN"]
percentile_low = fissile_results["acc_mean"].quantile(0.1)
percentile_mid = fissile_results["acc_mean"].quantile(0.5)
percentile_high = fissile_results["acc_mean"].quantile(0.9)
print(percentile_low, percentile_mid, percentile_high)

low_voxel_indices = fissile_results[np.isclose(fissile_results["acc_mean"], percentile_low, atol=1e-3)]["voxel_index"]
mid_voxel_indices = fissile_results[np.isclose(fissile_results["acc_mean"], percentile_mid, atol=1e-3)]["voxel_index"]
high_voxel_indices = fissile_results[(np.isclose(fissile_results["acc_mean"], percentile_high, atol=1e-3))]["voxel_index"]
print(low_voxel_indices, mid_voxel_indices, high_voxel_indices)

# Choose one from each
low_voxel_index = low_voxel_indices.iloc[0]
mid_voxel_index = mid_voxel_indices.iloc[0]
high_voxel_index = high_voxel_indices.iloc[5]

print(f'{low_voxel_index=}, {mid_voxel_index=}, {high_voxel_index=}')

# Set up datasets
seed = 42
lr = 1e-3
mesh_subdivide = 1
kappa = 100
batch_size = 512
mlp_model_path = Path("../models/deepfixel_mesh_mlp_healpix_2025-04-14_11-18-12/best_model.pth")
scnn_model_path = Path("../models/deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03/best_model.pth")

test_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/fissile_data")
fod2fixel_dir = Path("/home-local/saundam1/fissile/mrtrix_comparison_test_2")


test_dataset = GeneratedMeshDataset(n_fibers='both', directory=test_dir, return_fixels=True, return_fissile=True, subdivide=mesh_subdivide, kappa=kappa, healpix=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(test_dataset.n_mesh)
mlp_model = CrossingFiberMeshMLP(n_mesh=test_dataset.n_mesh)
mlp_model.load_state_dict(torch.load(mlp_model_path, weights_only=True))

scnn_model = CrossingFiberMeshSCNN(n_side=8, depth=5, patch_size=1, sh_degree=6, pooling_mode='average', pooling_name='spherical', use_hemisphere=True,
                  in_channels=1, out_channels=1, filter_start=2, block_depth=1, in_depth=1, kernel_sizeSph=3, kernel_sizeSpa=3, isoSpa=True, keepSphericalDim = True)
scnn_model.load_state_dict(torch.load(scnn_model_path, weights_only=True), strict=False)

# Set up dataloaders
n_mesh = test_dataset.n_mesh
sphere = test_dataset.icosphere

# Try plotting a few outputs
pdf_mesh, total_odf_mesh, fixels, fissile_outputs = next(iter(test_loader))
print(pdf_mesh.shape, total_odf_mesh.shape)

with torch.no_grad():
    mlp_output = mlp_model(total_odf_mesh)
    scnn_output = scnn_model(total_odf_mesh)

pdf_mesh = pdf_mesh.numpy()
total_odf_mesh = total_odf_mesh.numpy()
mlp_output = mlp_output.numpy()
scnn_output = scnn_output.numpy()
fixels = fixels.numpy()
fissile_outputs = fissile_outputs.numpy()
print(pdf_mesh.shape, total_odf_mesh.shape, mlp_output.shape, fixels.shape, fissile_outputs.shape)

m_list, l_list = sph_harm_ind_list(6)

# Plot
fig, ax = plt.subplots(3, 5, figsize=(10,10), subplot_kw={"projection": "3d"})
for i, voxel_idx in enumerate([high_voxel_index, mid_voxel_index, low_voxel_index]):
    fixel_to_plot = fixels[voxel_idx]
    fissile_odf= fissile_outputs[voxel_idx]

    theta, phi, vol = fixel_to_plot.T
    theta = theta[vol > 0]
    phi = phi[vol > 0]
    vol = vol[vol > 0]
    
    print(f"{voxel_idx}: {theta=}, {phi=}, {vol=}")
    true_odf = np.array([convert_sh_descoteaux_tournier(gen_dirac(m_list, l_list, theta=t, phi=p)) * v for t, p, v in zip(theta, phi, vol)])
    total_odf = np.sum(true_odf, axis=0)
    est_odf_mlp, net_est_dirs_mlp, net_est_vols_mlp = pdf2odfs(mlp_output[voxel_idx], sphere, amp_threshold=0.1)
    est_odf_scnn, net_est_dirs_scnn, net_est_vols_scnn = pdf2odfs(scnn_output[voxel_idx], sphere, amp_threshold=0.1)

    # Load fod2fixel
    afd = nib.load(fod2fixel_dir / f"fod_{voxel_idx}" / f"afd.nii.gz").get_fdata().squeeze()
    index = nib.load(fod2fixel_dir / f"fod_{voxel_idx}" / f"index_nii.nii.gz").get_fdata().squeeze().astype(int) 
    n_fixels = min(index[0], 3)
    first_idx = index[1]

    est_fod_mrtrix = np.zeros((n_fixels, 28))
    est_dirs = np.zeros((n_fixels, 3))
    est_afd = np.zeros(n_fixels)
    directions = nib.load(fod2fixel_dir / f"fod_{voxel_idx}" / f"directions.nii.gz").get_fdata().squeeze()
    print(f'{voxel_idx}: {index.shape=} and {index}, {directions.shape=}, {afd.shape=}')
    for j in range(n_fixels):
        # est_fod_mrtrix[j] = nib.load(data_dir / f"fod_{i}" / f"fixel_{j}_dixel_sh.nii.gz").get_fdata().squeeze()
        est_dirs[j] = directions[first_idx+j,:]
        est_afd[j] = afd[first_idx+j]
        v = est_afd[j] / np.sum(afd[first_idx:first_idx+n_fixels])
        r, t, p = cart2sphere(*est_dirs[j]) 
        est_fod_mrtrix[j] = convert_sh_descoteaux_tournier(gen_dirac(m_list, l_list, theta=t, phi=p))*v#est_afd[j]
    
    # Sort by estimated AFD 
    # print(f"{est_afd=}")
    # sort_idx = np.argsort(est_afd)[::-1]
    # est_fod_mrtrix = est_fod_mrtrix[sort_idx]

    # Match them (fod2fixel)
    fod2fixel_matched, _ = match_odfs(true_odf, est_fod_mrtrix)
    # fod2fixel_matched = est_fod_mrtrix

    # Match them after sorting by volume
    sort_idx = np.argsort(net_est_vols_mlp)[::-1]
    est_odf_mlp = est_odf_mlp[sort_idx]
    est_odf_mlp_matched, _ = match_odfs(true_odf, est_odf_mlp)

    sort_idx = np.argsort(net_est_vols_scnn)[::-1]
    est_odf_scnn = est_odf_scnn[sort_idx]
    est_odf_scnn_matched, _ = match_odfs(true_odf, est_odf_scnn)

    plot_odf(total_odf, ax=ax[i, 0], color="r")

    # Remove dims of Fissile that are all 0
    fissile_odf = fissile_odf[fissile_odf.sum(axis=1) != 0]

    # Match FISSILE outputs
    fissile_odf_matched, _ = match_odfs(true_odf, fissile_odf)

    # Plot true odf
    # for j, fod in enumerate(true_odf):
    #     plot_odf(fod, ax=ax[i, j+1], color="b")
    plot_multiple_odf(true_odf, ax=ax[i,1], color="r")

    # Plot estimated ODFs (fod2fixel)
    # for j, fod in enumerate(fissile_odf_matched):
    #     plot_odf(fod, ax=ax[i, j+4], color="g")
    plot_multiple_odf(fod2fixel_matched, ax=ax[i,2], color="g")

    # Plot estimated ODFs (DeepFixel MLP)
    # for j, fod in enumerate(fod2fixel_matched):
    #     plot_odf(fod, ax=ax[i, j+7], color="b")
    plot_multiple_odf(est_odf_mlp_matched, ax=ax[i,3], color="m")

    # # Plot estimated ODFs (crossing fiber net)
    # for j, fod in enumerate(est_odf_matched):
    #     plot_odf(fod, ax=ax[i, j+10], color="b")
    plot_multiple_odf(est_odf_scnn_matched, ax=ax[i,4], color="b")

ax[0,0].set_title("Multi-fiber ODF")
ax[0,1].set_title("Single-fiber ODFs")
ax[0,2].set_title("Estimated ODFs\n(fod2fixel)")
ax[0,3].set_title("Estimated ODFs\n(DeepFixel MLP)")
ax[0,4].set_title("Estimated ODFs\n(DeepFixel Spherical CNN)")

for a in ax.flatten():
    a.tick_params(axis='both', which='major', pad=-2)
    a.set_xlabel("x", labelpad=-15)
    a.set_ylabel("y", labelpad=-15)
    a.set_zlabel("z", labelpad=-15)
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_zticklabels([])
    a.view_init(azim=60, elev=25)


# plt.tight_layout()
# save_dir = Path("/home/local/VANDERBILT/saundam1/Pictures/fissile")
# plt.savefig(save_dir / "fig_qual_results.png", dpi=300)
plt.show()
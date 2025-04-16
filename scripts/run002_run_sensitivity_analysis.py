import torch
from deep_fixel.dataset import GeneratedMeshDataset, GeneratedMeshNIFTIDataset
from deep_fixel.models import CrossingFiberMeshMLP, CrossingFiberMeshSCNN
from deep_fixel.utils import plot_odf, plot_mesh, pdf2odfs, match_odfs, angular_corr_coeff, angular_separation
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dipy.reconst.shm import sf_to_sh, convert_sh_descoteaux_tournier, gen_dirac, sph_harm_ind_list
import numpy as np
import pandas as pd
import time
from line_profiler import profile
from dipy.core.geometry import cart2sphere, sphere2cart
from joblib import Parallel, delayed
import pdb

# v_save_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/sensitivity_large/sensitivity_dataset_v")
# theta_save_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/sensitivity_large/sensitivity_dataset_theta")
# rand_save_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/sensitivity_large/sensitivity_dataset_rand")

# seed = 42
# lr = 1e-3
# mesh_subdivide = 1
# kappa = 100
# batch_size = 512
# amp_threshold = 0.1
# mlp_model_path = Path("../models/deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03/best_model.pth")
# scnn_model_path = Path("../models/deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03/best_model.pth")

# # Load data
# sensitivity_angle_dataset = GeneratedMeshDataset(n_fibers=2, directory=theta_save_dir, return_fixels=True, subdivide=mesh_subdivide, kappa=kappa, healpix=True)
# sensitivity_angle_loader = DataLoader(sensitivity_angle_dataset, batch_size=512, shuffle=False)

# sensitivity_vol_dataset = GeneratedMeshDataset(n_fibers=2, directory=v_save_dir, return_fixels=True, subdivide=mesh_subdivide, kappa=kappa, healpix=True)
# sensitivity_vol_loader = DataLoader(sensitivity_vol_dataset, batch_size=512, shuffle=False)

# n_mesh = sensitivity_angle_dataset.n_mesh
# sphere = sensitivity_angle_dataset.icosphere
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Load the models
# mlp_model = CrossingFiberMeshMLP(n_mesh=sensitivity_angle_dataset.n_mesh)
# mlp_model.load_state_dict(torch.load(mlp_model_path, weights_only=True))

# scnn_model = CrossingFiberMeshSCNN(n_side=8, depth=5, patch_size=1, sh_degree=6, pooling_mode='average', pooling_name='spherical', use_hemisphere=True,
#                   in_channels=1, out_channels=1, filter_start=2, block_depth=1, in_depth=1, kernel_sizeSph=3, kernel_sizeSpa=3, isoSpa=True, keepSphericalDim = True)
# scnn_model.load_state_dict(torch.load(scnn_model_path, weights_only=True), strict=False)

# m_list, l_list = sph_harm_ind_list(6)

# for model_name in ['SCNN']:
#     if model_name == 'MLP':
#         model = mlp_model 
#         output_dir = '../outputs/deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03'
#     elif model_name == 'SCNN':
#         model = scnn_model
#         output_dir = '../outputs/deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03'

#     model = model.to(device)

#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True, parents=True)

#     test_results = {
#         "voxel_index": [],
#         "fiber_index": [],
#         "true_n_fibers": [],
#         "est_n_fibers": [],
#         "true_theta": [],
#         "true_phi": [],
#         "true_v": [],
#         "est_theta_matched": [],
#         "est_phi_matched": [],
#         "est_v_matched": [],
#         "acc": [],
#     }
#     with torch.no_grad():
        
#         for idx, test_data in enumerate(tqdm(sensitivity_angle_loader, desc='angle set')):
#             pdf_mesh, total_odf_mesh, fixels = test_data
#             pdf_mesh = pdf_mesh.to(device)
#             total_odf_mesh = total_odf_mesh.to(device)

#             output = model(total_odf_mesh)

#             # Move back to CPU
#             pdf_mesh = pdf_mesh.cpu().numpy()
#             total_odf_mesh = total_odf_mesh.cpu().numpy()
#             output = output.cpu().numpy()
    #         fixels = fixels.numpy()

    #         for i in range(len(pdf_mesh)):
    #             single_pdf_mesh = pdf_mesh[i]
    #             single_output = output[i]
    #             theta, phi, vol = fixels[i].T

    #             # Sort by vol and remove any with vol = 0
    #             sort_idx = np.argsort(vol)[::-1]
    #             sort_idx = sort_idx[vol[sort_idx] > 0]
    #             theta = theta[sort_idx]
    #             phi = phi[sort_idx]
    #             vol = vol[sort_idx]

    #             true_odf = np.array([convert_sh_descoteaux_tournier(gen_dirac(m_list, l_list, theta=t, phi=p))*v for t, p, v in zip(theta, phi, vol)])
    #             est_odf, est_dirs, est_vol = pdf2odfs(single_output, sphere, amp_threshold=amp_threshold)

    #             # If all empty, just get a random dir and vol of 0
    #             if est_odf.shape[0] == 0 and est_dirs.shape[0] == 0 and est_vol.shape[0] == 0:
    #                 est_odf = np.zeros((1, 28))
    #                 est_dirs = np.zeros((1, 2))
    #                 est_vol = np.zeros(1)
    #                 est_vol[0] = 1

    #             # Sort by est_vol
    #             sort_idx = np.argsort(est_vol)[::-1]
    #             est_odf = est_odf[sort_idx]
    #             est_dirs = est_dirs[sort_idx]
    #             est_vol = est_vol[sort_idx]

    #             est_theta, est_phi = est_dirs.T

    #             # Match them
    #             est_odf_matched, index_array = match_odfs(true_odf, est_odf)           
    #             est_theta_matched = est_theta[index_array] 
    #             est_phi_matched = est_phi[index_array] 
    #             est_vol_matched = est_vol[index_array] 

    #             est_n_fibers = len(est_odf_matched)
    #             true_n_fibers = len(true_odf)

    #             # if idx == 0 and i < 3:
    #             #     fig, ax = plt.subplots(1, 5, figsize=(10, 5), subplot_kw={"projection": "3d"})
    #             #     plot_mesh(pdf_mesh[i], sphere, ax=ax[0], cmap="cmc.batlow", alpha=0.5)
    #             #     plot_mesh(total_odf_mesh[i], sphere, ax=ax[1], cmap="cmc.batlow", alpha=0.5)
    #             #     plot_mesh(output[i], sphere, ax=ax[2], cmap="cmc.batlow", alpha=0.5)
    #             #     [plot_odf(odf, ax=ax[3], color="b", alpha=0.2) for odf in true_odf]
    #             #     [plot_odf(odf, ax=ax[4], color="r", alpha=0.2) for odf in est_odf_matched]

    #             #     x, y, z = sphere2cart(np.ones_like(theta), theta, phi)
    #             #     for a in ax:
    #             #         for i in range(len(x)):
    #             #             a.plot([-x[i], x[i]], [-y[i], y[i]], [-z[i], z[i]], color="k", alpha=0.5)
    #             #     print(f'{theta=}, {phi=}, {vol=}')
    #             #     print(f'{est_theta=}, {est_phi=}, {est_vol=}')
    #             #     print(f'{est_theta_matched=}, {est_phi_matched=}, {est_vol_matched=}')

    #             #     plt.show()

    #             # Calculate ACC for each fiber
    #             for j, (odf1, odf2) in enumerate(zip(true_odf, est_odf_matched)):
    #                 acc = angular_corr_coeff(odf1, odf2)
    #                 test_results["voxel_index"].append(i+idx*512)
    #                 test_results["fiber_index"].append(j)
    #                 test_results["true_n_fibers"].append(true_n_fibers)
    #                 test_results["est_n_fibers"].append(est_n_fibers)
    #                 test_results["true_theta"].append(theta[j])
    #                 test_results["true_phi"].append(phi[j])
    #                 test_results["true_v"].append(vol[j])
    #                 test_results["est_theta_matched"].append(est_theta_matched[j])
    #                 test_results["est_phi_matched"].append(est_phi_matched[j])
    #                 test_results["est_v_matched"].append(est_vol_matched[j])
    #                 test_results["acc"].append(acc)
    #                 # print(f"{i},{j},{acc=}")

    #             # if est_n_fibers < true_n_fibers, add dummy rows
    #             missing_fibers = true_n_fibers - est_n_fibers
    #             if missing_fibers > 0:
    #                 for j in range(est_n_fibers, true_n_fibers):
    #                     test_results["voxel_index"].append(i+idx*512)
    #                     test_results["fiber_index"].append(j)
    #                     test_results["true_n_fibers"].append(true_n_fibers)
    #                     test_results["est_n_fibers"].append(est_n_fibers)
    #                     test_results["true_theta"].append(theta[j])
    #                     test_results["true_phi"].append(phi[j])
    #                     test_results["true_v"].append(vol[j])
    #                     test_results["est_theta_matched"].append(np.nan)
    #                     test_results["est_phi_matched"].append(np.nan)
    #                     test_results["est_v_matched"].append(np.nan)
    #                     test_results["acc"].append(0)

    # # Save to CSV
    # test_results = pd.DataFrame(test_results)
    # test_results.to_csv(output_dir / "test_results_angle.csv", index=False)

    # test_results = {
    #     "voxel_index": [],
    #     "fiber_index": [],
    #     "true_n_fibers": [],
    #     "est_n_fibers": [],
    #     "true_theta": [],
    #     "true_phi": [],
    #     "true_v": [],
    #     "est_theta_matched": [],
    #     "est_phi_matched": [],
    #     "est_v_matched": [],
    #     "acc": [],
    # }
    # with torch.no_grad():
        
    #     for idx, test_data in enumerate(tqdm(sensitivity_vol_loader, desc='vol set')):
    #         pdf_mesh, total_odf_mesh, fixels = test_data
    #         pdf_mesh = pdf_mesh.to(device)
    #         total_odf_mesh = total_odf_mesh.to(device)

    #         output = model(total_odf_mesh)

    #         # Move back to CPU
    #         pdf_mesh = pdf_mesh.cpu().numpy()
    #         output = output.cpu().numpy()
    #         fixels = fixels.numpy()

    #         for i in range(len(pdf_mesh)):
    #             single_pdf_mesh = pdf_mesh[i]
    #             single_output = output[i]
    #             theta, phi, vol = fixels[i].T

    #             # Sort by vol and remove any with vol = 0
    #             sort_idx = np.argsort(vol)[::-1]
    #             sort_idx = sort_idx[vol[sort_idx] > 0]
    #             theta = theta[sort_idx]
    #             phi = phi[sort_idx]
    #             vol = vol[sort_idx]

    #             true_odf = np.array([convert_sh_descoteaux_tournier(gen_dirac(m_list, l_list, theta=t, phi=p))*v for t, p, v in zip(theta, phi, vol)])
    #             est_odf, est_dirs, est_vol = pdf2odfs(single_output, sphere, amp_threshold=amp_threshold)

    #             # If all empty, just get a random dir and vol of 0
    #             if est_odf.shape[0] == 0 and est_dirs.shape[0] == 0 and est_vol.shape[0] == 0:
    #                 est_odf = np.zeros((1, 28))
    #                 est_dirs = np.zeros((1, 2))
    #                 est_vol = np.zeros(1)
    #                 est_vol[0] = 1

    #             est_theta, est_phi = est_dirs.T


    #             # Match them
    #             est_odf_matched, index_array = match_odfs(true_odf, est_odf)
    #             est_theta_matched = est_theta[index_array]
    #             est_phi_matched = est_phi[index_array]
    #             est_vol_matched = est_vol[index_array]

    #             est_n_fibers = len(est_odf_matched)
    #             true_n_fibers = len(true_odf)

    #             # Calculate ACC for each fiber
    #             for j, (odf1, odf2) in enumerate(zip(true_odf, est_odf_matched)):
    #                 acc = angular_corr_coeff(odf1, odf2)
    #                 test_results["voxel_index"].append(i+idx*512)
    #                 test_results["fiber_index"].append(j)
    #                 test_results["true_n_fibers"].append(true_n_fibers)
    #                 test_results["est_n_fibers"].append(est_n_fibers)
    #                 test_results["true_theta"].append(theta[j])
    #                 test_results["true_phi"].append(phi[j])
    #                 test_results["true_v"].append(vol[j])
    #                 test_results["est_theta_matched"].append(est_theta_matched[j])
    #                 test_results["est_phi_matched"].append(est_phi_matched[j])
    #                 test_results["est_v_matched"].append(est_vol_matched[j])
    #                 test_results["acc"].append(acc)

    #             # if est_n_fibers < true_n_fibers, add dummy rows
    #             missing_fibers = true_n_fibers - est_n_fibers
    #             if missing_fibers > 0:
    #                 for j in range(est_n_fibers, true_n_fibers):
    #                     test_results["voxel_index"].append(i+idx*512)
    #                     test_results["fiber_index"].append(j)
    #                     test_results["true_n_fibers"].append(true_n_fibers)
    #                     test_results["est_n_fibers"].append(est_n_fibers)
    #                     test_results["true_theta"].append(theta[j])
    #                     test_results["true_phi"].append(phi[j])
    #                     test_results["true_v"].append(vol[j])
    #                     test_results["est_theta_matched"].append(np.nan)
    #                     test_results["est_phi_matched"].append(np.nan)
    #                     test_results["est_v_matched"].append(np.nan)
    #                     test_results["acc"].append(0)

    # # Save to CSV
    # test_results = pd.DataFrame(test_results)
    # test_results.to_csv(output_dir / "test_results_vol.csv", index=False)

# Combine them all into one
df_list = []
for method in ["fod2fixel", "DeepFixel MLP", "DeepFixel Spherical CNN"]:
    if method == "fod2fixel":
        output_dir = Path("../outputs/fod2fixel")
    elif method == "DeepFixel MLP":
        output_dir = Path("../outputs/deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03")
    elif method == "DeepFixel Spherical CNN":
        output_dir = Path("../outputs/deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03")

    # Read tes_results_angle.csv and test_results_vol.csv
    angle_results = pd.read_csv(output_dir / "test_results_angle.csv")
    vol_results = pd.read_csv(output_dir / "test_results_vol.csv")

    # Add true_angular_separation
    for voxel_index in angle_results["voxel_index"].unique():
        voxel_results = angle_results[angle_results["voxel_index"] == voxel_index]
        
        # Skip 3 voxels
        if len(voxel_results) == 3:
            angle_results.loc[angle_results["voxel_index"] == voxel_index, "true_angular_separation"] = np.nan
            angle_results.loc[angle_results["voxel_index"] == voxel_index, "est_angular_separation"] = np.nan
        else:
            fiber1_row = voxel_results[voxel_results["fiber_index"] == 0]
            fiber2_row = voxel_results[voxel_results["fiber_index"] == 1]

            true_ang_sep = angular_separation([fiber1_row["true_theta"].values[0], fiber1_row["true_phi"].values[0]], [fiber2_row["true_theta"].values[0], fiber2_row["true_phi"].values[0]])
            angle_results.loc[angle_results["voxel_index"] == voxel_index, "true_angular_separation"] = true_ang_sep

            est_ang_sep = angular_separation([fiber1_row["est_theta_matched"].values[0], fiber1_row["est_phi_matched"].values[0]], [fiber2_row["est_theta_matched"].values[0], fiber2_row["est_phi_matched"].values[0]])
            angle_results.loc[angle_results["voxel_index"] == voxel_index, "est_angular_separation"] = est_ang_sep

    for voxel_index in vol_results["voxel_index"].unique():
        voxel_results = vol_results[vol_results["voxel_index"] == voxel_index]
        
        # Skip 3 voxels
        if len(voxel_results) == 3:
            vol_results.loc[vol_results["voxel_index"] == voxel_index, "true_angular_separation"] = np.nan
            vol_results.loc[vol_results["voxel_index"] == voxel_index, "est_angular_separation"] = np.nan
        else:
            fiber1_row = voxel_results[voxel_results["fiber_index"] == 0]
            fiber2_row = voxel_results[voxel_results["fiber_index"] == 1]

            true_ang_sep = angular_separation([fiber1_row["true_theta"].values[0], fiber1_row["true_phi"].values[0]], [fiber2_row["true_theta"].values[0], fiber2_row["true_phi"].values[0]])
            vol_results.loc[vol_results["voxel_index"] == voxel_index, "true_angular_separation"] = true_ang_sep

            est_ang_sep = angular_separation([fiber1_row["est_theta_matched"].values[0], fiber1_row["est_phi_matched"].values[0]], [fiber2_row["est_theta_matched"].values[0], fiber2_row["est_phi_matched"].values[0]])
            vol_results.loc[vol_results["voxel_index"] == voxel_index, "est_angular_separation"] = est_ang_sep

    # Add vol_frac (lowest true vol in each voxel)
    angle_results["vol_frac"] = angle_results.groupby("voxel_index")["true_v"].transform(lambda x: x.min())
    vol_results["vol_frac"] = vol_results.groupby("voxel_index")["true_v"].transform(lambda x: x.min())

    # Add a column to each with the method name and experiment name
    angle_results["method"] = method
    angle_results["experiment"] = "angle"
    vol_results["method"] = method
    vol_results["experiment"] = "vol"

    df_list.append(angle_results)
    df_list.append(vol_results)

# Combine them
combined_results = pd.concat(df_list, ignore_index=True)
combined_results.to_csv("sensitivity_analysis_large.csv", index=False)
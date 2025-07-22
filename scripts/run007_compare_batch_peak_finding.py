import torch
from deep_fixel.dataset import GeneratedMeshNIFTIDataset
from deep_fixel.models import CrossingFiberMeshMLP
from deep_fixel.utils import pdf2odfs
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from joblib import Parallel, delayed
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.direction import peak_directions
from scipy.spatial.distance import cdist

# === Compare functions ===
def compare_functions(meshes, sphere, amp_threshold=0.1):
    N = len(meshes)
    print(f"\n--- Comparing on {N} voxels ---")

    # Fast peak finder timing
    t2 = time.time()
    # dirs_dipy, vals_dipy, ind_dipy = peak_directions(meshes, sphere, relative_peak_threshold=0.1, min_separation_angle=10, is_symmetric=True)
    results_dipy = Parallel(n_jobs=1)(
        delayed(peak_directions)(mesh, sphere, relative_peak_threshold=amp_threshold, min_separation_angle=10, is_symmetric=True) for mesh in tqdm(meshes, desc="Dipy peak finding")
    )
    t3 = time.time()
    print(f"Fast find_peaks time: {t3 - t2:.2f}s")

    # Split results
    dirs_dipy, vals_dipy, ind_dipy = zip(*results_dipy)

    # Original function timing
    t0 = time.time()
    results_original = Parallel(n_jobs=1)(
        delayed(pdf2odfs)(mesh, sphere, amp_threshold) for mesh in tqdm(meshes, desc="Original pdf2odfs")
    )
    t1 = time.time()
    print(f"Original pdf2odfs time: {t1 - t0:.2f}s")

    # Split results
    odfs_original, dirs_original, vals_original = zip(*results_original)

    # Loop over original results and compare
    theta_mse, phi_mse, vol_mse = [], [], []
    for i in range(N):
        theta_orig, phi_orig = dirs_original[i][:, 0], dirs_original[i][:, 1]
        x_dipy, y_dipy, z_dipy = dirs_dipy[i][:, 0], dirs_dipy[i][:, 1], dirs_dipy[i][:, 2]
        r_dipy, theta_dipy, phi_dipy = cart2sphere(x_dipy, y_dipy, z_dipy)

        vol_orig = vals_original[i]
        vol_dipy = vals_dipy[i]

        # Sort by dipy volume and match
        sorted_indices = np.argsort(vol_dipy)[::-1]
        theta_dipy = theta_dipy[sorted_indices]
        phi_dipy = phi_dipy[sorted_indices]
        vol_dipy = vol_dipy[sorted_indices]

        sorted_indices_orig = np.argsort(vol_orig)[::-1]
        theta_orig = theta_orig[sorted_indices_orig]
        phi_orig = phi_orig[sorted_indices_orig]
        vol_orig = vol_orig[sorted_indices_orig]

        # Keep only top 3 peaks (if we even have that many)
        if len(theta_dipy) > 3:
            theta_dipy = theta_dipy[:3]
            phi_dipy = phi_dipy[:3]
            vol_dipy = vol_dipy[:3]
        if len(theta_orig) > 3:
            theta_orig = theta_orig[:3]
            phi_orig = phi_orig[:3]
            vol_orig = vol_orig[:3]

        vol_dipy /= np.sum(vol_dipy)  # Normalize Dipy volumes
        # theta_orig = theta_orig % np.pi
        # theta_dipy = theta_dipy % np.pi
        # phi_orig = phi_orig % (2 * np.pi)
        # phi_dipy = phi_dipy % (2 * np.pi)

        print(f'Voxel {i+1}/{N}:')
        print(f'  Original: {len(theta_orig)} peaks, Dipy: {len(theta_dipy)} peaks')
        print(f'  Original peaks: {theta_orig}, {phi_orig}')
        print(f'  Dipy peaks: {theta_dipy}, {phi_dipy}')
        print(f'  Original volumes: {vol_orig}')
        print(f'  Dipy volumes: {vol_dipy}')

# === Main ===
if __name__ == "__main__":
    rand_save_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/sensitivity_large/sensitivity_dataset_rand2")
    deepfixel_mlp_model = "/home/local/VANDERBILT/saundam1/Documents/spherical_deep_fixel/models/deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03"

    test_dataset = GeneratedMeshNIFTIDataset(
        n_fibers=3,
        nifti_path=rand_save_dir / "sensitivity_dataset_rand2.nii.gz",
        subdivide=1,
        kappa=100,
        healpix=True
    )
    sphere = test_dataset.icosphere
    n_mesh = test_dataset.n_mesh
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Load model
    model = CrossingFiberMeshMLP(n_mesh=n_mesh)
    model.load_state_dict(torch.load(f"{deepfixel_mlp_model}/best_model.pth",
                                     weights_only=True, map_location='cuda'))
    model.to('cuda')
    model.eval()

    # Get model outputs
    outputs = []
    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Running model"):
            total_odf_mesh = test_data.to('cuda')
            output = model(total_odf_mesh).cpu().numpy().astype(np.float64)
            outputs.append(output)
    outputs = np.concatenate(outputs)  # (N, V)

    # Compare
    compare_functions(outputs, sphere, amp_threshold=0.1)

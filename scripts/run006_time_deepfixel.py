import torch
from deep_fixel.dataset import GeneratedMeshDataset, GeneratedMeshNIFTIDataset
from deep_fixel.models import CrossingFiberMeshMLP, CrossingFiberMeshSCNN
from deep_fixel.utils import pdf2odfs
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

rand_save_dir = Path("/fs5/p_masi/saundam1/outputs/crossing_fibers/sensitivity_large/sensitivity_dataset_rand2")

deepfixel_mlp_model = "/home/local/VANDERBILT/saundam1/Documents/spherical_deep_fixel/models/deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03"
deepfixel_scnn_model = "/home/local/VANDERBILT/saundam1/Documents/spherical_deep_fixel/models/deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03"

subdivide_mesh = 1
kappa = 100
amp_threshold = 0.1
n_fibers = 3

# Load data
test_dataset = GeneratedMeshNIFTIDataset(n_fibers=3, nifti_path=rand_save_dir/"sensitivity_dataset_rand2.nii.gz", subdivide=subdivide_mesh, kappa=kappa, healpix=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

n_mesh = test_dataset.n_mesh
sphere = test_dataset.icosphere
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model = CrossingFiberMeshMLP(n_mesh=n_mesh)
model_path = f"{deepfixel_mlp_model}/best_model.pth"
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.to(device)
model.eval()

m_list, l_list = sph_harm_ind_list(6)

# First, time test_dataset
@profile
def test(test_loader, model, sphere, amp_threshold=0.1):
    start_time = time.time()
    
    outputs = []
    with torch.no_grad():
        for idx, test_data in enumerate(tqdm(test_loader, desc='time test')):
            total_odf_meshes = test_data
            total_odf_mesh = total_odf_meshes.to(device)

            output = model(total_odf_mesh)

            total_odf_mesh = total_odf_mesh.cpu().numpy()
            output = output.cpu().numpy()

            outputs.append(output)
        
        # Parallel search for pdf2odfs
        outputs = np.concatenate(outputs).astype(np.float64)
        Parallel(n_jobs=-1)(delayed(pdf2odfs)(single_output, sphere, amp_threshold=0.1, use_dipy=True, min_separation_angle=10, is_symmetric=True) for single_output in tqdm(outputs))
        
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

print("Timing for MLP...")
test(test_loader, model, sphere, amp_threshold)

n_mesh = test_dataset.n_mesh
sphere = test_dataset.icosphere
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model = CrossingFiberMeshSCNN(device=device, n_side=8, depth=5, patch_size=1, sh_degree=6, pooling_mode='average', pooling_name='spherical', use_hemisphere=True,
            in_channels=1, out_channels=1, filter_start=2, block_depth=1, in_depth=1, kernel_sizeSph=3, kernel_sizeSpa=3, isoSpa=True, keepSphericalDim = True)
model_path = f"{deepfixel_scnn_model}/best_model.pth"
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=False)
model.to(device)
model.eval()

print("Timing for SCNN...")
test(test_loader, model, sphere, amp_threshold)

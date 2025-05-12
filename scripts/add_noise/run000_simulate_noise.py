from deep_fixel.dataset import RandomMeshDataset
import numpy as np
import torch 
from pathlib import Path
from deep_fixel.utils import plot_multiple_odf, plot_odf, rotate_odf, plot_mesh
import matplotlib.pyplot as plt
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.core.sphere import Sphere, hemi_icosahedron
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.data import get_fnames
from dipy.sims.voxel import add_noise, single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, AxSymShResponse, recursive_response, auto_response_ssst
from dipy.reconst.shm import (
    convert_sh_descoteaux_tournier,
    gen_dirac,
    sh_to_sf,
    sf_to_sh,
    sph_harm_ind_list,
)
from numpy.random import default_rng
from scipy.stats import vonmises_fisher
from scipy.spatial.transform import Rotation as R

# Create gradient table from Stanford HARDI and extract 2000 shell
dmri_path, bval_path, bvec_path = get_fnames(name="stanford_hardi")
mask_path = Path(dmri_path).parent / "mask.nii.gz"

data, affine = load_nifti(dmri_path)
mask_data, _ = load_nifti(mask_path)
mask = mask_data.astype(bool)
bval, bvec = read_bvals_bvecs(bval_path, bvec_path)
gtab = gradient_table(
    bvals=bval,
    bvecs=bvec,
)
gtab_sphere = Sphere(x=gtab.bvecs[:, 0], y=gtab.bvecs[:, 1], z=gtab.bvecs[:, 2])

# First, simulate a response function
b = 2000
lambda_mean = 0.9e-3
lambda_perp = 0.6*lambda_mean
S0 = 1

# Plot response on sphere
sphere = hemi_icosahedron.subdivide(n=4)
theta = sphere.theta
response_amp = np.exp(-b * lambda_perp)*np.exp(-3*b*(lambda_mean - lambda_perp)* (np.cos(theta)**2))
response_sh = sf_to_sh(
    response_amp,
    sphere,
    sh_order_max=6,
    basis_type="descoteaux07",
)
m_list, l_list = sph_harm_ind_list(6)
response = AxSymShResponse(S0, response_sh[(m_list == 0) & (l_list % 2 == 0)])
# response = recursive_response(
#     gtab=gtab,
#     data=data,
#     mask=mask,
#     sh_order_max=6,
# )
# response, _ = auto_response_ssst(
#     gtab=gtab,
#     data=data,
#     roi_radii=10,
#     fa_thr=0.7,
# )
# evals = response[0]
# evecs = np.array([[0,1,0], [0,0,1], [1,0,0]]).T
# response = single_tensor_odf(sphere.vertices, evals, evecs)
# print(response.shape)
# ax = plot_mesh(response, sphere)
# ax.set_title("Response function")

# response = recursive_response(
#     gtab=gtab,
#     data=data,
#     mask=mask,
#     sh_order_max=6,
#     peak_thr=0.01,
#     init_fa=0.08,
#     convergence=0.001,
#     parallel=False,
#     num_processes=1
# )

response_plot = response.on_sphere(sphere)
ax = plot_odf(response_sh, basis="descoteaux07")
ax.set_title("Response function")

# Generate data from random mesh dataset
dataset = RandomMeshDataset(
    n_fibers=3,
    l_max=6,
    subdivide=1,
    kappa=100,
    seed=42,
    deterministic=True,
    size=1000,
    healpix=True
)

# Plot a few samples
dataset.rng = default_rng(dataset.seed)
n_fibers = dataset.n_fibers

total_odfs = []
total_dodfs_noisy = []
for i in range(5):

    # Generate random volume fraction using Dirichlet distribution
    vol = dataset.rng.dirichlet(np.ones(n_fibers))

    # Generate random x, y, z by normalizing random Gaussian direction and converting to spherical
    xyz = dataset.rng.normal(size=(3, n_fibers))
    xyz = xyz / np.linalg.norm(xyz, axis=0)

    # Keep the z > 0
    xyz[2] = np.abs(xyz[2])
    x, y, z = xyz
    r, theta, phi = cart2sphere(x, y, z)

    # Simulate ODFs at these angles and volume fractions
    odfs = [
        v
        * convert_sh_descoteaux_tournier(gen_dirac(dataset.m_list, dataset.l_list, t, p))
        for v, t, p in zip(vol, theta, phi)
    ]
    odfs = np.array(odfs)
    total_odf = np.sum(odfs, axis=0)
    total_odfs.append(total_odf)

    # Simulate *diffusion* ODFs at these angles and volume fractions by rotating the response function
    total_dodf = np.zeros((28,))
    for j in range(n_fibers):
        # Rotate the response function
        rot = R.from_euler("ZYZ", [phi[j], -theta[j], 0])
        rotated_response = rotate_odf(response_sh, rot)

        # Simulate the ODF
        odf = vol[j] * rotated_response
        total_dodf += odf
    
    # Now project onto gtab_sphere and add noise
    total_dodf_gtab = sh_to_sf(
        total_dodf,
        gtab_sphere,
        sh_order_max=dataset.l_max,
        basis_type="tournier07",
    )

    # Add noise
    total_dodf_noisy = add_noise(
        total_dodf_gtab,
        snr=30,
        S0=S0,
        noise_type="rician",
        rng=dataset.rng,
    )

    print(response_sh.shape)
    csd_model = ConstrainedSphericalDeconvModel(
        gtab,
        response=response,
        sh_order_max=dataset.l_max
    )
    print(total_dodf_noisy.shape)
    total_dodf_noisy_sh = csd_model.fit(total_dodf_noisy).shm_coeff
    print(total_dodf_noisy_sh.shape)
    total_dodf_noisy_sh = convert_sh_descoteaux_tournier(total_dodf_noisy_sh)

    # Scale to unit integral
    # total_dodf_noisy_sh[0] = 1 / np.sqrt(4 * np.pi)

    # # Convert back to SH by fitting with CSD
    # total_odf_noisy_sh = sf_to_sh(
    #     total_odf_noisy,
    #     gtab_sphere,
    #     sh_order_max=dataset.l_max,
    #     basis_type="tournier07",
    # )
    total_dodf_noisy_sh = np.array(total_dodf_noisy_sh)
    total_dodfs_noisy.append(total_dodf_noisy_sh)

# Plot
total_odfs = np.array(total_odfs)
ax = plot_multiple_odf(total_odfs, color='cyan', alpha=0.15)
ax.set_title("Simulated ODFs without noise")

total_odfs_noisy = np.array(total_dodfs_noisy)
ax = plot_multiple_odf(total_odfs_noisy, color='red', alpha=0.15)
ax.set_title("Simulated ODFs with noise")

# Plot individual ones
ax = plot_odf(total_odfs[0], color='cyan', alpha=0.15)
ax = plot_odf(total_odfs_noisy[0], color='red', alpha=0.5, ax=ax)
ax.set_title("Simulated ODFs with noise (red) and without noise (cyan)")

print(total_odfs[0])
print(total_odfs_noisy[0])

plt.show()
from deep_fixel.dataset import RandomMeshDataset
import numpy as np
import torch 
from deep_fixel.utils import plot_multiple_odf, plot_odf
import matplotlib.pyplot as plt
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.core.sphere import Sphere, hemi_icosahedron
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import add_noise
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, AxSymShResponse
from dipy.reconst.shm import (
    convert_sh_descoteaux_tournier,
    gen_dirac,
    sh_to_sf,
    sf_to_sh,
    sph_harm_ind_list,
)
from numpy.random import default_rng
from scipy.stats import vonmises_fisher

# Create gradient table from Stanford HARDI and extract 2000 shell
bval_selected = 2000
bval = np.loadtxt("/home/local/VANDERBILT/saundam1/.dipy/stanford_hardi/HARDI150.bval")
bvec = np.loadtxt("/home/local/VANDERBILT/saundam1/.dipy/stanford_hardi/HARDI150.bvec")

bvec = bvec[:,bval == bval_selected]
bval = bval[bval == bval_selected]

gtab = gradient_table(
    bvals=bval,
    bvecs=bvec,
)
gtab_sphere = Sphere(x=gtab.bvecs[:, 0], y=gtab.bvecs[:, 1], z=gtab.bvecs[:, 2])

# First, simulate a response function
b = bval_selected
lambda_mean = 0.9e-3
lambda_perp = 0.6*lambda_mean

# Plot response on sphere
sphere = hemi_icosahedron.subdivide(n=4)
theta = sphere.theta
response_amp = np.exp(-b * lambda_perp)*np.exp(-3*b*(lambda_mean - lambda_perp) * (np.cos(theta)**2))
response_sh = sf_to_sh(
    response_amp,
    sphere,
    sh_order_max=6,
    basis_type="tournier07",
)
response = AxSymShResponse(1, response_sh)
ax = plot_odf(response_sh)
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
total_odfs_noisy = []
for i in range(1):

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

    # Now project onto gtab_sphere and add noise
    total_odf_gtab = sh_to_sf(
        total_odf,
        gtab_sphere,
        sh_order_max=dataset.l_max,
        basis_type="tournier07",
    )

    # Add noise
    total_odf_noisy = add_noise(
        total_odf_gtab,
        snr=30,
        S0=1,
        noise_type="rician",
        rng=dataset.rng,
    )
    
    print(response_sh.shape)
    csd_model = ConstrainedSphericalDeconvModel(
        gtab,
        response=response,
        sh_order_max=dataset.l_max
    )
    print(total_odf_noisy.shape)
    total_odf_noisy_sh = csd_model.fit(total_odf_noisy).shm_coeff
    print(total_odf_noisy_sh.shape)
    total_odf_noisy_sh = convert_sh_descoteaux_tournier(total_odf_noisy_sh)

    # # Convert back to SH by fitting with CSD
    # total_odf_noisy_sh = sf_to_sh(
    #     total_odf_noisy,
    #     gtab_sphere,
    #     sh_order_max=dataset.l_max,
    #     basis_type="tournier07",
    # )
    total_odf_noisy_sh = np.array(total_odf_noisy_sh)
    total_odfs_noisy.append(total_odf_noisy_sh)

# Plot
total_odfs = np.array(total_odfs)
ax = plot_multiple_odf(total_odfs)
ax.set_title("Simulated ODFs (no noise)", color='blue') 

total_odfs_noisy = np.array(total_odfs_noisy)
ax = plot_multiple_odf(total_odfs_noisy, color='red')
ax.set_title("Simulated ODFs (with noise)")

plt.show()
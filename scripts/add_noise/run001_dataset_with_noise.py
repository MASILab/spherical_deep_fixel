from deep_fixel.dataset import RandomMeshDataset
from deep_fixel.utils import plot_mesh, plot_odf, plot_multiple_odf
from dipy.core.geometry import sphere2cart
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Create a random mesh dataset
dataset_with_noise = RandomMeshDataset(
    n_fibers=3,
    l_max=6,
    subdivide=1,
    kappa=100,
    seed=42,
    size=5000,
    deterministic=True,
    return_fixels=True,
    healpix=True,
    csd=True,
    snr=30,
)

dataset_without_noise = RandomMeshDataset(
    n_fibers=3,
    l_max=6,
    subdivide=1,
    kappa=100,
    seed=42,
    size=5000,
    deterministic=True,
    return_fixels=True,
    healpix=True,
    csd=False,
    snr=None,
)

fig, axes = plt.subplots(1, 5, figsize=(20, 4), subplot_kw={"projection": "3d"})
for i, (pdf_mesh, total_odf_mesh, fixels) in enumerate(iter(dataset_with_noise)):
    ax = axes[i]
    pdf_mesh = pdf_mesh.cpu().numpy()
    total_odf_mesh = total_odf_mesh.cpu().numpy()
    fixels = fixels.cpu().numpy()

    # Plot the mesh
    ax = plot_mesh(pdf_mesh, sphere=dataset_with_noise.sphere, ax=ax, alpha=0.4)

    # Plot the total ODF
    ax = plot_odf(total_odf_mesh, sphere=dataset_with_noise.sphere, ax=ax, color="red")

    # Also plot fixels (lines)
    # fixels = np.stack([theta, phi, vol], axis=1)
    for fixel in fixels:
        theta, phi, vol = fixel
        x, y, z = sphere2cart(vol, theta, phi)
        ax.plot([-x, x], [-y, y], [-z, z], color="black", linewidth=1)

    if i >= 4:
        break

fig.suptitle("With noise")

fig, axes = plt.subplots(1, 5, figsize=(20, 4), subplot_kw={"projection": "3d"})
for i, (pdf_mesh, total_odf_mesh, fixels) in enumerate(iter(dataset_without_noise)):
    ax = axes[i]
    pdf_mesh = pdf_mesh.cpu().numpy()
    total_odf_mesh = total_odf_mesh.cpu().numpy()
    fixels = fixels.cpu().numpy()

    # Plot the mesh
    ax = plot_mesh(pdf_mesh, sphere=dataset_without_noise.sphere, ax=ax, alpha=0.4)

    # Plot the total ODF
    ax = plot_odf(
        total_odf_mesh, sphere=dataset_without_noise.sphere, ax=ax, color="red"
    )

    # Also plot fixels (lines)
    for fixel in fixels:
        theta, phi, vol = fixel
        x, y, z = sphere2cart(vol, theta, phi)
        ax.plot([-x, x], [-y, y], [-z, z], color="black", linewidth=1)

    if i >= 4:
        break


fig.suptitle("Without noise")
plt.show()

# Create a dataloader
dataloader = DataLoader(
    dataset_with_noise,
    batch_size=512,
)
for i, (pdf_mesh, total_odf_mesh, fixels) in enumerate(dataloader):
    print(f"Batch {i}:")
    print(f"PDF mesh shape: {pdf_mesh.shape}")
    print(f"Total ODF mesh shape: {total_odf_mesh.shape}")
    print(f"Fixels shape: {fixels.shape}")

    if i >= 4:
        break

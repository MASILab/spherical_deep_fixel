import torch
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
from tqdm import tqdm
from ..dataset import GeneratedMeshNIFTIDataset
from ..models import CrossingFiberMeshMLP, CrossingFiberMeshSCNN
from ..utils import pdf2odfs
from pathlib import Path
from dipy.reconst.shm import (
    sph_harm_ind_list,
)
from dipy.core.geometry import sphere2cart
import argparse
import pdb


def run_deep_fixel(
    nifti_path,
    output_dir,
    model_path,
    mask=None,
    max_num=None,
    lmax=6,
    subdivide=3,
    healpix=True,
    amp_threshold=0.1,
    model="mesh_scnn",
    batch_size=512,
    gpu_id=None,
    **kwargs,  # For pdf2odfs function
):
    # Load data
    test_dataset = GeneratedMeshNIFTIDataset(
        nifti_path=nifti_path,
        lmax=lmax,
        subdivide=subdivide,
        healpix=healpix,
        mask=mask,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_mesh = test_dataset.n_mesh
    sphere = test_dataset.icosphere
    if gpu_id is None:
        device = "cpu"
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load the model
    strict = False if model == "mesh_scnn" else True
    if model == "mesh_mlp":
        model = CrossingFiberMeshMLP(n_mesh=n_mesh, device=device, sphere=sphere)
    elif model == "mesh_scnn":
        model = CrossingFiberMeshSCNN(
            device=device,
            n_side=8,
            depth=5,
            patch_size=1,
            sh_degree=lmax,
            pooling_mode="average",
            pooling_name="spherical",
            use_hemisphere=True,
            in_channels=1,
            out_channels=1,
            filter_start=2,
            block_depth=1,
            in_depth=1,
            kernel_sizeSph=3,
            kernel_sizeSpa=3,
            isoSpa=True,
            keepSphericalDim=True,
        )
    else:
        raise ValueError(f"Model {model} not recognized")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=strict)
    model.to(device)

    m_list, l_list = sph_harm_ind_list(lmax)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    est_odf_list = []
    est_dirs_list = []
    est_vol_list = []
    n_fixels_list = []
    with torch.no_grad():
        for idx, test_data in enumerate(tqdm(test_loader, desc="DeepFixel")):
            total_odf_mesh = test_data
            total_odf_mesh = total_odf_mesh.to(device)

            output = model(total_odf_mesh)

            # Move back to CPU
            output = output.cpu().numpy().astype(np.float64)

            for i in range(len(output)):
                single_output = output[i]
                est_odf, est_dirs, est_vol = pdf2odfs(
                    single_output,
                    sphere,
                    amp_threshold=amp_threshold,
                    lmax=lmax,
                    max_num=max_num,
                    use_dipy=True,
                    **kwargs,
                )
                est_dirs_xyz = sphere2cart(1.0, est_dirs[:, 0], est_dirs[:, 1])
                est_dirs_xyz = np.stack(est_dirs_xyz, axis=1)

                # Sort by volume (largest to smallest)
                index = np.argsort(-est_vol)
                est_odf = est_odf[index]
                est_dirs_xyz = est_dirs_xyz[index]
                est_vol = est_vol[index]

                est_odf_list.append(est_odf)
                est_dirs_list.append(est_dirs_xyz)
                est_vol_list.append(est_vol)
                n_fixels_list.append(len(est_dirs))

    # Now form fixel directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if mask is not None:
        mask_nifti = nib.load(mask)
        mask_data = mask_nifti.get_fdata().squeeze().astype(bool)
        mask_data = mask_data.flatten()
    else:
        mask_data = np.ones(test_dataset.shape, dtype=bool)
        mask_data = mask_data.flatten()

    # index.nii.gz is i by j by k by 2
    # 1st dim: # of fixels in voxel
    # 2nd dim: index of first fixel in voxel
    index_data = np.zeros(test_dataset.shape[:-1] + (2,), dtype=np.int32)
    index_data.reshape(-1, 2)[mask_data, 0] = n_fixels_list
    index_data.reshape(-1, 2)[mask_data, 1] = np.cumsum([0] + n_fixels_list[:-1])
    index_nifti = nib.nifti2.Nifti2Image(index_data, affine=test_dataset.affine)

    # fraction.nii.gz: n x 1 x 1
    # nth fixel's volume fraction
    fraction_data = np.concatenate(est_vol_list).astype(np.float32)
    fraction_data = fraction_data[:, np.newaxis, np.newaxis]
    fraction_nifti = nib.nifti2.Nifti2Image(fraction_data, affine=test_dataset.affine)

    # direction.nii.gz: n x 3 x 1
    # nth fixel's direction
    direction_data = np.concatenate(est_dirs_list).astype(np.float32)
    direction_data = direction_data[:, :, np.newaxis]
    direction_nifti = nib.nifti2.Nifti2Image(direction_data, affine=np.eye(4))

    # fod.nii.gz: i x j x k x num_sh for each
    fod_data = np.concatenate(est_odf_list).astype(np.float32)
    fod_data = fod_data[:, :, np.newaxis]
    indices = index_data[..., 1].flatten()
    num_fixels = index_data[..., 0].flatten()
    if max_num is None:
        max_num = np.max(num_fixels)
    for m in range(max_num):
        idx = indices + m
        fod_m = np.zeros(test_dataset.shape[:-1] + (len(m_list),), dtype=np.float32)
        fod_m = fod_m.reshape(-1, len(m_list))
        fod_m[(idx < num_fixels + indices)] = fod_data[
            idx[idx < num_fixels + indices], :, 0
        ]

        fod_m = fod_m.reshape(test_dataset.shape[:-1] + (len(m_list),))
        fod_nifti = nib.nifti2.Nifti2Image(fod_m, affine=test_dataset.affine)
        fod_nifti.to_filename(output_dir / f"fod_{m}.nii.gz")

    # Save all
    index_nifti.to_filename(output_dir / "index.nii.gz")
    fraction_nifti.to_filename(output_dir / "fraction.nii.gz")
    direction_nifti.to_filename(output_dir / "directions.nii.gz")
    print(f"Saved files to fixel directory {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepFixel on a NIFTI file containing multi-fiber FODs."
    )
    parser.add_argument(
        "fod",
        type=str,
        help="Path to input FOD file represented as spherical harmonics in tournier07/mrtrix convention",
    )
    parser.add_argument(
        "fixel_directory", type=str, help="Path to output fixel directory"
    )
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument(
        "--mask", type=str, default=None, help="Optional path to mask NIFTI file"
    )
    parser.add_argument(
        "--maxnum",
        type=int,
        default=None,
        help="Optional maximum number of fixels per voxel",
    )
    parser.add_argument(
        "--lmax", type=int, default=6, help="Maximum spherical harmonic order"
    )
    parser.add_argument(
        "--subdivide", type=int, default=3, help="Subdivision level for icosphere"
    )
    parser.add_argument(
        "--amp_threshold",
        type=float,
        default=0.1,
        help="Amplitude threshold for peak detection",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mesh_scnn",
        choices=["mesh_mlp", "mesh_scnn"],
        help="Model type",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for inference"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="Optional GPU ID to use for inference"
    )

    args = parser.parse_args()

    run_deep_fixel(
        nifti_path=args.fod,
        output_dir=args.fixel_directory,
        model_path=args.model_path,
        mask=args.mask,
        max_num=args.maxnum,
        lmax=args.lmax,
        subdivide=args.subdivide,
        healpix=True,
        amp_threshold=args.amp_threshold,
        model=args.model,
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()

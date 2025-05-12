import numpy as np
import torch
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.core.sphere import Sphere, hemi_icosahedron
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.data import get_fnames
from dipy.data.fetcher import fetch_stanford_hardi
from dipy.reconst.shm import (
    convert_sh_descoteaux_tournier,
    gen_dirac,
    sh_to_sf,
    sf_to_sh,
    sph_harm_ind_list,
)
from dipy.sims.voxel import add_noise
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, AxSymShResponse
from numpy.random import default_rng
from scipy.stats import vonmises_fisher
from torch.utils.data import Dataset, IterableDataset
from trimesh import Trimesh
import nibabel as nib
from pathlib import Path
import re
from .utils import load_fissile_mat, rotate_odf
from hsd.utils.sampling import HealpixSampling
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R
import os


class RandomODFDataset(IterableDataset):
    def __init__(self, n_fibers, l_max=6, seed=None, size=None, deterministic=False):
        """Generate ODFs at random angles and volume fractions (using Tournier07/mrtrix convention)

        Parameters
        ----------
        n_fibers : int
            Number of fibers to simulate
        seed : int, optional
            Random seed, by default None
        l_max : int, optional
            Maximum spherical harmonic order, by default 6
        size : int, optional
            Number of samples to generate, by default None for infinite
        deterministic : bool, optional
            If True, generate same for each index, by default False. Only used if size is not None.
        """
        self.n_fibers = n_fibers
        self.seed = seed
        self.size = size
        self.rng = default_rng(seed)

        self.m_list, self.l_list = sph_harm_ind_list(l_max)
        self.deteministic = deterministic

    def generate_odf(self, seed=None):
        """Return odfs, a n_fibers x n_coeff array of random single-fiber odfs and total_odf, the 1 x n_coeff sum of all odfs."""
        if seed is not None:
            self.rng = default_rng(seed)

        # Generate random volume fraction using Dirichlet distribution
        vol = self.rng.dirichlet(np.ones(self.n_fibers))

        # Generate random x, y, z by normalizing random Gaussian direction and converting to spherical
        xyz = self.rng.normal(size=(3, self.n_fibers))
        xyz /= np.linalg.norm(xyz, axis=0)
        r, theta, phi = cart2sphere(*xyz)

        # Simulate ODFs at these angles and volume fractions
        odfs = [
            v
            * convert_sh_descoteaux_tournier(gen_dirac(self.m_list, self.l_list, t, p))
            for v, t, p in zip(vol, theta, phi)
        ]
        odfs = np.array(odfs)
        total_odf = np.sum(odfs, axis=0)

        odfs = torch.tensor(odfs, dtype=torch.float32)
        total_odf = torch.tensor(total_odf, dtype=torch.float32)

        return odfs, total_odf

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.size is not None:
            for i in range(self.size):
                if self.deteministic:
                    yield self.generate_odf(self.seed + i)
                else:
                    yield self.generate_odf()
        else:
            while True:
                yield self.generate_odf()


class RandomFixelDataset(IterableDataset):
    def __init__(self, n_fibers, l_max=6, seed=None, size=None, deterministic=False):
        """Generate ODFs at random angles and volume fractions (using Tournier07/mrtrix convention)

        Parameters
        ----------
        n_fibers : int
            Number of fibers to simulate. If 'both', will generate both 2 and 3 fibers (50%).
        l_max : int, optional
            Maximum spherical harmonic order, by default 6
        seed : int, optional
            Random seed, by default None
        size : int, optional
            Number of samples to generate, by default None for infinite
        deterministic : bool, optional
            If True, generate same for each index, by default False. Only used if size is not None.
        """
        self.n_fibers = n_fibers
        self.seed = seed
        self.size = size
        self.rng = default_rng(seed)

        self.m_list, self.l_list = sph_harm_ind_list(6)
        self.deteministic = deterministic

    def generate_odf(self, seed=None):
        """Return fixels, a n_fibers x 3 (theta,phi,vol_frac) array of random single-fiber odfs and total_odf, the 1 x n_coeff sum of all odfs."""
        if seed is not None:
            self.rng = default_rng(seed)

        if self.n_fibers == "both":
            n_fibers = self.rng.choice([2, 3])
        else:
            n_fibers = self.n_fibers

        # Generate random volume fraction using Dirichlet distribution
        vol = self.rng.dirichlet(np.ones(n_fibers))

        # Generate random x, y, z by normalizing random Gaussian direction and converting to spherical
        xyz = self.rng.normal(size=(3, n_fibers))
        xyz /= np.linalg.norm(xyz, axis=0)
        r, theta, phi = cart2sphere(*xyz)

        # Simulate ODFs at these angles and volume fractions
        odfs = [
            v
            * convert_sh_descoteaux_tournier(gen_dirac(self.m_list, self.l_list, t, p))
            for v, t, p in zip(vol, theta, phi)
        ]
        odfs = np.array(odfs)
        total_odf = np.sum(odfs, axis=0)

        # If 2 fibers, add a fixel with vol frac 0 and random angle
        if n_fibers == 2:
            rand_xyz = self.rng.normal(size=(3,))
            rand_xyz /= np.linalg.norm(rand_xyz)
            r, extra_theta, extra_phi = cart2sphere(*rand_xyz)
            theta = np.append(theta, extra_theta)
            phi = np.append(phi, extra_phi)
            vol = np.append(vol, 0)

        # Sort by volume fraction
        index = np.argsort(vol)[::-1]
        theta = theta[index]
        phi = phi[index]
        vol = vol[index]

        theta = theta % (np.pi)
        phi = phi % (2 * np.pi)

        fixels = np.stack([theta, phi, vol], axis=1)
        fixels = torch.tensor(fixels, dtype=torch.float32)
        total_odf = torch.tensor(total_odf, dtype=torch.float32)

        return fixels, total_odf

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.size is not None:
            for i in range(self.size):
                if self.deteministic:
                    yield self.generate_odf(self.seed + i)
                else:
                    yield self.generate_odf()
        else:
            while True:
                yield self.generate_odf()


class RandomMeshDataset(IterableDataset):
    def __init__(
        self,
        n_fibers,
        l_max=6,
        subdivide=3,
        kappa=100,
        seed=None,
        size=None,
        deterministic=False,
        return_fixels=False,
        healpix=False,
        csd=False,
        snr=None,
    ):
        """Generate ODFs at random angles and volume fractions (using Tournier07/mrtrix convention)

        Parameters
        ----------
        n_fibers : int or str
            Number of fibers to simulate. If 'both', will simulate both 2 and 3 fibers (50%).
        seed : int, optional
            Random seed, by default None
        l_max : int, optional
            Maximum spherical harmonic order, by default 6
        subdivide : int, optional
            Number of times to subdivide the ico-hemisphere if healpix=False,
            otherwise corresponds to depth of Healpix sampling (where smaller is more vertices), by default 3
        kappa : float, optional
            Concentration parameter for von Mises-Fisher distribution, by default 100
        size : int, optional
            Number of samples to generate, by default None for infinite
        deterministic : bool, optional
            If True, generate same for each index, by default False. Only used if size is not None.
        return_fixels : bool, optional
            If True, return fixels along with PDF meshes, by default False
        healpix : bool, optional
            If True, sample on healpix instead of icosphere, by default False
        csd : bool, optional
            If True, use CSD instead of simulating Dirac delta functions, by default False
        snr : float, optional
            Signal to noise ratio for CSD, by default None
        """
        self.n_fibers = n_fibers
        self.seed = seed
        self.size = size
        self.rng = default_rng(seed)

        self.m_list, self.l_list = sph_harm_ind_list(l_max)
        self.l_max = l_max
        self.deteministic = deterministic

        self.csd = csd
        self.snr = snr
        if self.csd:
            # Create gradient table from Stanford HARDI and extract 2000 shell
            dmri_path, bval_path, bvec_path = get_fnames(name="stanford_hardi")
            bval, bvec = read_bvals_bvecs(bval_path, bvec_path)
            self.gtab = gradient_table(
                bvals=bval,
                bvecs=bvec,
            )
            self.gtab_sphere = Sphere(
                x=self.gtab.bvecs[:, 0],
                y=self.gtab.bvecs[:, 1],
                z=self.gtab.bvecs[:, 2],
            )

            # First, simulate a response function
            b = 2000
            lambda_mean = 0.9e-3
            lambda_perp = 0.6 * lambda_mean
            S0 = 1
            sphere = hemi_icosahedron.subdivide(n=4)
            theta = sphere.theta
            response_amp = np.exp(-b * lambda_perp) * np.exp(
                -3 * b * (lambda_mean - lambda_perp) * (np.cos(theta) ** 2)
            )
            self.response_sh = sf_to_sh(
                response_amp,
                sphere,
                sh_order_max=6,
                basis_type="descoteaux07",
            )
            m_list, l_list = sph_harm_ind_list(6)
            self.response = AxSymShResponse(
                S0, self.response_sh[(m_list == 0) & (l_list % 2 == 0)]
            )
            self.csd_model = ConstrainedSphericalDeconvModel(
                self.gtab, response=self.response, sh_order_max=self.l_max
            )

        if healpix:
            n_side = 8
            depth = subdivide
            patch_size = 1
            sh_degree = 6
            pooling_mode = "average"
            pooling_name = "mixed"
            use_hemisphere = True
            sampling = HealpixSampling(
                n_side,
                depth,
                patch_size,
                sh_degree,
                pooling_mode,
                pooling_name,
                use_hemisphere,
            )
            vecs = sampling.vec[0]
            self.icosphere = Sphere(xyz=vecs)
            self.n_mesh = len(vecs)
            self.sphere = self.icosphere
        else:
            self.icosphere = hemi_icosahedron.subdivide(n=subdivide)
            self.n_mesh = len(self.icosphere.vertices)
            self.sphere = self.icosphere
        self.kappa = kappa

        self.return_fixels = return_fixels

    def generate_odf(self, seed=None):
        """Return odfs, a n_fibers x n_mesh array of random single-fiber odfs and total_odf, the 1 x n_coeff sum of all odfs."""
        if seed is not None:
            self.rng = default_rng(seed)

        # Set n_fibers
        if self.n_fibers == "both":
            n_fibers = self.rng.choice([2, 3])
        else:
            n_fibers = self.n_fibers

        # Generate random volume fraction using Dirichlet distribution
        vol = self.rng.dirichlet(np.ones(n_fibers))

        # Generate random x, y, z by normalizing random Gaussian direction and converting to spherical
        xyz = self.rng.normal(size=(3, n_fibers))
        xyz = xyz / np.linalg.norm(xyz, axis=0)

        # Keep the z > 0
        xyz[2] = np.abs(xyz[2])
        x, y, z = xyz
        r, theta, phi = cart2sphere(x, y, z)

        if self.csd:
            # Simulate diffusion ODFs at given angles/volume fractions, sample and fit CSD
            total_dodf = np.zeros((28,))
            for j in range(n_fibers):
                # Rotate the response function
                rot = R.from_euler("ZYZ", [phi[j], -theta[j], 0])
                rotated_response = rotate_odf(self.response_sh, rot)

                # Simulate the ODF
                odf = vol[j] * rotated_response
                total_dodf += odf

            # Now project onto gtab_sphere and add noise
            total_dodf_gtab = sh_to_sf(
                total_dodf,
                self.gtab_sphere,
                sh_order_max=self.l_max,
                basis_type="tournier07",
            )
            if self.snr is not None:
                total_dodf_gtab = add_noise(
                    total_dodf_gtab,
                    snr=self.snr,
                    S0=1,
                    noise_type="rician",
                    rng=self.rng,
                )

            # Fit CSD
            total_odf = self.csd_model.fit(total_dodf_gtab).shm_coeff
            total_odf = convert_sh_descoteaux_tournier(total_odf)

        else:
            # Simulate ODFs at these angles and volume fractions
            odfs = [
                v
                * convert_sh_descoteaux_tournier(
                    gen_dirac(self.m_list, self.l_list, t, p)
                )
                for v, t, p in zip(vol, theta, phi)
            ]
            odfs = np.array(odfs)
            total_odf = np.sum(odfs, axis=0)

        # Sample total_odf along mesh
        total_odf_mesh = sh_to_sf(
            total_odf,
            self.icosphere,
            sh_order_max=self.l_max,
            basis_type="tournier07",
        )

        pdf = [
            v * vonmises_fisher(mu, self.kappa).pdf(self.icosphere.vertices)
            for v, mu in zip(vol, xyz.T)
        ]
        pdf_mesh = np.sum(pdf, axis=0)

        # Now return
        pdf_mesh = torch.tensor(pdf_mesh, dtype=torch.float32)
        total_odf_mesh = torch.tensor(total_odf_mesh, dtype=torch.float32)

        if self.return_fixels:
            if len(vol) == 2 and self.n_fibers == "both":
                rand_xyz = np.random.normal(size=(3,))
                rand_xyz /= np.linalg.norm(rand_xyz)
                r, extra_theta, extra_phi = cart2sphere(*rand_xyz)
                theta = np.append(theta, extra_theta)
                phi = np.append(phi, extra_phi)
                vol = np.append(vol, 0)

            fixels = np.stack([theta, phi, vol], axis=1)
            fixels = torch.tensor(fixels, dtype=torch.float32)

            return pdf_mesh, total_odf_mesh, fixels

        return pdf_mesh, total_odf_mesh

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.size is not None:
            for i in range(self.size):
                if self.deteministic:
                    yield self.generate_odf(self.seed + i)
                else:
                    yield self.generate_odf()
        else:
            while True:
                yield self.generate_odf()


class GeneratedMeshDataset(Dataset):
    def __init__(
        self,
        n_fibers,
        directory,
        subdivide=3,
        kappa=100,
        glob_name=None,
        return_fixels=False,
        return_fissile=False,
        healpix=False,
    ):
        """Load ODFs from a directory of .mat files from FISSILE outputs.

        Parameters
        ----------
        n_fibers : int or str
            Number of fibers in each ODF. If 'both', will use 2 and 3.
        directory : str
            Directory containing .mat files
        subdivide : int, optional
            Number of times to subdivide the ico-hemisphere if healpix=False,
            otherwise corresponds to depth of Healpix sampling (where smaller is more vertices), by default 3
        kappa : float, optional
            Concentration parameter for von Mises-Fisher distribution, by default 100
        glob_name : str, optional
            Glob name to search for, by default "*2fibers*.mat" or "*3fibers*.mat" if None
        return_fixels : bool, optional
            If True, return fixels along with PDF meshes, by default False
        return_fissile : bool, optional
            If True, return the FISSILE outputs, by default False
        healpix : bool, optional
            If True, sample on healpix instead of icosphere, by default False
        """
        self.n_fibers = n_fibers
        self.directory = directory
        self.l_max = 6

        if healpix:
            n_side = 8
            depth = subdivide
            patch_size = 1
            sh_degree = self.l_max
            pooling_mode = "average"
            pooling_name = "mixed"
            use_hemisphere = True
            sampling = HealpixSampling(
                n_side,
                depth,
                patch_size,
                sh_degree,
                pooling_mode,
                pooling_name,
                use_hemisphere,
            )
            vecs = sampling.vec[0]
            self.icosphere = Sphere(xyz=vecs)
            self.n_mesh = len(vecs)
            self.sphere = self.icosphere
        else:
            self.icosphere = hemi_icosahedron.subdivide(n=subdivide)
            self.n_mesh = len(self.icosphere.vertices)
            self.sphere = self.icosphere

        self.kappa = kappa

        self.return_fixels = return_fixels
        self.return_fissile = return_fissile

        # Search directory for *<n_fibers>fibers*.mat files
        if glob_name is not None:
            mat_files = list(Path(directory).glob(glob_name))

        else:
            if self.n_fibers == "both":
                mat_files = list(Path(directory).glob("*2fibers*.mat")) + list(
                    Path(directory).glob("*3fibers*.mat")
                )
            else:
                mat_files = sorted(
                    list(Path(directory).glob(f"*{n_fibers}fibers*.mat"))
                )

        # Sort by basename
        mat_files = sorted(mat_files, key=lambda f: f.stem)
        # mat_files = sorted(mat_files, key=lambda f: int(re.search(r'_(\d+)(?=\.)', str(f)).group(1)) if re.search(r'_(\d+)(?=\.)', str(f)) else float('inf'))
        print(mat_files)

        # Append each ODF in file to list
        self.pdf_meshes = []
        self.total_odf_meshes = []
        if return_fixels:
            self.fixels = []
        if return_fissile:
            self.fissile_outputs = []
        for mat_file in tqdm(mat_files, desc="Loading .mat files"):
            mat_dict_list = load_fissile_mat(mat_file)
            for mat_dict in mat_dict_list:
                theta = mat_dict["true_theta"]
                phi = mat_dict["true_phi"]
                vol = mat_dict["true_v"]

                if self.return_fissile:
                    # Extend to 10 x 28
                    fissile_outputs = np.zeros((10, 28))
                    fissile_outputs_dict = mat_dict["rotate_separate_fod_matched"]
                    est_v = mat_dict["est_v_matched"]
                    # Sort by est_v
                    sort_idx = np.argsort(est_v)[::-1]
                    fissile_outputs_dict = fissile_outputs_dict[sort_idx]
                    fissile_outputs[: len(fissile_outputs_dict)] = mat_dict[
                        "rotate_separate_fod_matched"
                    ]
                    self.fissile_outputs.append(
                        torch.tensor(fissile_outputs, dtype=torch.float32)
                    )

                if self.return_fixels:
                    # If only two, need to add a dummy fixel
                    if len(vol) == 2:
                        rand_xyz = np.random.normal(size=(3,))
                        rand_xyz /= np.linalg.norm(rand_xyz)
                        r, extra_theta, extra_phi = cart2sphere(*rand_xyz)
                        theta_f = np.append(theta, extra_theta)
                        phi_f = np.append(phi, extra_phi)
                        vol_f = np.append(vol, 0)
                    else:
                        theta_f = theta
                        phi_f = phi
                        vol_f = vol
                    fixel = np.stack([theta_f, phi_f, vol_f], axis=1)
                    self.fixels.append(torch.tensor(fixel, dtype=torch.float32))

                total_odf = mat_dict["total_fod"].squeeze()

                x, y, z = sphere2cart(np.ones_like(theta), theta, phi)
                xyz = np.stack([x, y, z], axis=0)

                # Keep the z > 0
                xyz[2] = np.abs(xyz[2])
                x, y, z = xyz
                r, theta, phi = cart2sphere(x, y, z)

                # Sample total_odf along mesh
                total_odf_mesh = sh_to_sf(
                    total_odf,
                    self.icosphere,
                    sh_order_max=self.l_max,
                    basis_type="tournier07",
                )

                pdf = [
                    v * vonmises_fisher(mu, self.kappa).pdf(self.icosphere.vertices)
                    for v, mu in zip(vol, xyz.T)
                ]
                pdf_mesh = np.sum(pdf, axis=0)

                # Now return
                pdf_mesh = torch.tensor(pdf_mesh, dtype=torch.float32)
                total_odf_mesh = torch.tensor(total_odf_mesh, dtype=torch.float32)

                self.pdf_meshes.append(pdf_mesh)
                self.total_odf_meshes.append(total_odf_mesh)

    def __len__(self):
        return len(self.pdf_meshes)

    def __getitem__(self, idx):
        if self.return_fixels and self.return_fissile:
            return (
                self.pdf_meshes[idx],
                self.total_odf_meshes[idx],
                self.fixels[idx],
                self.fissile_outputs[idx],
            )
        elif self.return_fixels:
            return self.pdf_meshes[idx], self.total_odf_meshes[idx], self.fixels[idx]
        elif self.return_fissile:
            return (
                self.pdf_meshes[idx],
                self.total_odf_meshes[idx],
                self.fissile_outputs[idx],
            )
        else:
            return self.pdf_meshes[idx], self.total_odf_meshes[idx]


class GeneratedMeshNIFTIDataset(Dataset):
    def __init__(
        self,
        n_fibers,
        nifti_path,
        subdivide=3,
        kappa=100,
        healpix=False,
    ):
        """Load ODFs from a directory of .mat files from FISSILE outputs.

        Parameters
        ----------
        n_fibers : int or str
            Number of fibers in each ODF. If 'both', will use 2 and 3.
        nifti_path : str
            Path to nifti file
        subdivide : int, optional
            Number of times to subdivide the ico-hemisphere if healpix=False,
            otherwise corresponds to depth of Healpix sampling (where smaller is more vertices), by default 3
        kappa : float, optional
            Concentration parameter for von Mises-Fisher distribution, by default 100
        healpix : bool, optional
            If True, sample on healpix instead of icosphere, by default False
        """
        self.n_fibers = n_fibers
        self.nifti_path = nifti_path
        self.l_max = 6

        if healpix:
            n_side = 8
            depth = subdivide
            patch_size = 1
            sh_degree = 6
            pooling_mode = "average"
            pooling_name = "mixed"
            use_hemisphere = True
            sampling = HealpixSampling(
                n_side,
                depth,
                patch_size,
                sh_degree,
                pooling_mode,
                pooling_name,
                use_hemisphere,
            )
            vecs = sampling.vec[0]
            self.icosphere = Sphere(xyz=vecs)
            self.n_mesh = len(vecs)
            self.sphere = self.icosphere
        else:
            self.icosphere = hemi_icosahedron.subdivide(n=subdivide)
            self.n_mesh = len(self.icosphere.vertices)
            self.sphere = self.icosphere

        self.kappa = kappa

        # Load NIFTI
        nifti = nib.load(nifti_path)

        # Flatten all except last axis
        nifti_data = nifti.get_fdata().squeeze()
        nifti_data = nifti_data.reshape(-1, nifti_data.shape[-1])

        # Append each ODF in file to list
        self.total_odf_meshes = []
        self.total_odf_meshes = sh_to_sf(
            nifti_data,
            self.icosphere,
            sh_order_max=self.l_max,
            basis_type="tournier07",
        )
        self.total_odf_meshes = torch.tensor(self.total_odf_meshes, dtype=torch.float32)

    def __len__(self):
        return len(self.total_odf_meshes)

    def __getitem__(self, idx):
        return self.total_odf_meshes[idx]

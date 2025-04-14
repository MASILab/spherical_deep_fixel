import numpy as np
import torch
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.core.sphere import Sphere, hemi_icosahedron
from dipy.reconst.shm import (
    convert_sh_descoteaux_tournier,
    gen_dirac,
    sh_to_sf,
    sph_harm_ind_list,
)
from numpy.random import default_rng
from scipy.stats import vonmises_fisher
from torch.utils.data import Dataset, IterableDataset
from trimesh import Trimesh
import nibabel as nib
from pathlib import Path
import re
from .utils import load_fissile_mat

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
        face=False,
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
            Number of times to subdivide the ico-hemisphere, by default 3
        kappa : float, optional
            Concentration parameter for von Mises-Fisher distribution, by default 100
        size : int, optional
            Number of samples to generate, by default None for infinite
        deterministic : bool, optional
            If True, generate same for each index, by default False. Only used if size is not None.
        return_fixels : bool, optional
            If True, return fixels along with PDF meshes, by default False
        face : bool, optional
            Sample on face of icosphere instead of at vertices, by default False
        """
        self.n_fibers = n_fibers
        self.seed = seed
        self.size = size
        self.rng = default_rng(seed)

        self.m_list, self.l_list = sph_harm_ind_list(l_max)
        self.l_max = l_max
        self.deteministic = deterministic

        self.icosphere = hemi_icosahedron.subdivide(n=subdivide)
        self.kappa = kappa

        self.return_fixels = return_fixels
        self.face = face

        if self.face:
            mesh = Trimesh(vertices=self.icosphere.vertices, faces=self.icosphere.faces)
            self.face_centers = mesh.triangles_center
            self.face_centers = (
                self.face_centers / np.linalg.norm(self.face_centers, axis=1)[:, None]
            )
            self.face_sphere = Sphere(xyz=self.face_centers)

        self.n_mesh = (
            len(self.icosphere.vertices) if not self.face else len(self.face_centers)
        )
        self.sphere = self.icosphere if not self.face else self.face_sphere

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

        # Simulate ODFs at these angles and volume fractions
        odfs = [
            v
            * convert_sh_descoteaux_tournier(gen_dirac(self.m_list, self.l_list, t, p))
            for v, t, p in zip(vol, theta, phi)
        ]
        odfs = np.array(odfs)
        total_odf = np.sum(odfs, axis=0)

        # Sample total_odf along mesh
        if self.face:
            total_odf_mesh = sh_to_sf(
                total_odf,
                self.face_sphere,
                sh_order_max=self.l_max,
                basis_type="tournier07",
            )
        else:
            total_odf_mesh = sh_to_sf(
                total_odf,
                self.icosphere,
                sh_order_max=self.l_max,
                basis_type="tournier07",
            )

        if self.face:
            pdf = [
                v * vonmises_fisher(mu, self.kappa).pdf(self.face_sphere.vertices)
                for v, mu in zip(vol, xyz.T)
            ]
        else:
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
        face=False,
    ):
        """Load ODFs from a directory of .mat files from FISSILE outputs.

        Parameters
        ----------
        n_fibers : int or str
            Number of fibers in each ODF. If 'both', will use 2 and 3.
        directory : str
            Directory containing .mat files
        subdivide : int, optional
            Number of times to subdivide the ico-hemisphere, by default 3
        kappa : float, optional
            Concentration parameter for von Mises-Fisher distribution, by default 100
        glob_name : str, optional
            Glob name to search for, by default "*2fibers*.mat" or "*3fibers*.mat" if None
        return_fixels : bool, optional
            If True, return fixels along with PDF meshes, by default False
        return_fissile : bool, optional
            If True, return the FISSILE outputs, by default False
        """
        self.n_fibers = n_fibers
        self.directory = directory

        self.icosphere = hemi_icosahedron.subdivide(n=subdivide)
        self.kappa = kappa
        self.l_max = 6

        self.return_fixels = return_fixels
        self.return_fissile = return_fissile

        self.face = face
        if self.face:
            mesh = Trimesh(vertices=self.icosphere.vertices, faces=self.icosphere.faces)
            self.face_centers = mesh.triangles_center  # (n,3)
            self.face_centers = (
                self.face_centers / np.linalg.norm(self.face_centers, axis=1)[:, None]
            )
            self.face_sphere = Sphere(xyz=self.face_centers)

        self.n_mesh = (
            len(self.icosphere.vertices) if not self.face else len(self.face_centers)
        )
        self.sphere = self.icosphere if not self.face else self.face_sphere

        # Search directory for *<n_fibers>fibers*.mat files
        if glob_name is not None:
            mat_files = list(Path(directory).glob(glob_name))

        else:
            if self.n_fibers == "both":
                mat_files = list(Path(directory).glob("*2fibers*.mat")) + list(
                    Path(directory).glob("*3fibers*.mat")
                )
            else:
                mat_files = sorted(list(Path(directory).glob(f"*{n_fibers}fibers*.mat")))
        mat_files = sorted(mat_files, key=lambda f: int(re.search(r'_(\d+)(?=\.)', str(f)).group(1)) if re.search(r'_(\d+)(?=\.)', str(f)) else float('inf'))
        print(mat_files)

        # Append each ODF in file to list
        self.pdf_meshes = []
        self.total_odf_meshes = []
        if return_fixels:
            self.fixels = []
        if return_fissile:
            self.fissile_outputs = []
        for mat_file in mat_files:
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
                if self.face:
                    total_odf_mesh = sh_to_sf(
                        total_odf,
                        self.face_sphere,
                        sh_order_max=self.l_max,
                        basis_type="tournier07",
                    )
                else:
                    total_odf_mesh = sh_to_sf(
                        total_odf,
                        self.icosphere,
                        sh_order_max=self.l_max,
                        basis_type="tournier07",
                    )

                if self.face:
                    pdf = [
                        v
                        * vonmises_fisher(mu, self.kappa).pdf(self.face_sphere.vertices)
                        for v, mu in zip(vol, xyz.T)
                    ]
                else:
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
    ):
        """Load ODFs from a directory of .mat files from FISSILE outputs.

        Parameters
        ----------
        n_fibers : int or str
            Number of fibers in each ODF. If 'both', will use 2 and 3.
        nifti_path : str
            Path to nifti file
        subdivide : int, optional
            Number of times to subdivide the ico-hemisphere, by default 3
        kappa : float, optional
            Concentration parameter for von Mises-Fisher distribution, by default 100
        """
        self.n_fibers = n_fibers
        self.nifti_path = nifti_path

        self.icosphere = hemi_icosahedron.subdivide(n=subdivide)
        self.kappa = kappa
        self.l_max = 6

        self.n_mesh = len(self.icosphere.vertices)
        self.sphere = self.icosphere

        # Load NIFTI
        nifti = nib.load(nifti_path)
        
        # Flatten all except last axis
        nifti_data = nifti.get_fdata().squeeze()
        nifti_data = nifti_data.reshape(-1, nifti_data.shape[-1])

        # Append each ODF in file to list
        self.total_odf_meshes = []
        for i in range(nifti_data.shape[0]):
            odf = nifti_data[i]
            total_odf_mesh = sh_to_sf(
                odf,
                self.icosphere,
                sh_order_max=self.l_max,
                basis_type="tournier07",
            )
        
            total_odf_mesh = torch.tensor(total_odf_mesh, dtype=torch.float32)
            self.total_odf_meshes.append(total_odf_mesh)

    def __len__(self):
        return len(self.total_odf_meshes)

    def __getitem__(self, idx):
        return self.total_odf_meshes[idx]

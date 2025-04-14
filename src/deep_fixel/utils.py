from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from dipy.core.geometry import cart2sphere
from dipy.core.sphere import unit_icosahedron
from dipy.reconst.shm import (
    convert_sh_descoteaux_tournier,
    convert_sh_to_full_basis,
    gen_dirac,
    order_from_ncoef,
    real_sh_descoteaux,
    real_sh_tournier,
    sph_harm_ind_list,
)
from matplotlib.tri import Triangulation
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import minimize
from scipy.signal import argrelmax
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
import cmcrameri 

def plot_odf(odf, ax=None, color="blue", basis="tournier", alpha=1, linewidth=0.1):
    """Plot a spherical orientation distribution function represented by spherical harmonic coefficients.

    Parameters
    ----------
    odf : NumPy array
        Array of spherical harmonic coefficients representing the ODF.
    ax : Matplotlib axis, optional
        Axis to plot the ODF on, by default None (creates a new figure).
    color : str, optional
        Color for ODF, by default 'blue'
    basis : str, optional
        Either "tournier" or "descoteaux", by default "tournier". (See https://docs.dipy.org/stable/theory/sh_basis.html).
    alpha : float, optional
        Opacity, by default 1
    linewidth : float, optional
        Linewidth for the ODF, by default 0.1

    Returns
    -------
    ax : Matplotlib axis
        Axis with the ODF plotted.
    """
    if odf.ndim == 1:
        odf = odf[:, None]
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    sphere = unit_icosahedron.subdivide(n=4)
    x, y, z = sphere.vertices.T
    _, theta, phi = cart2sphere(x, y, z)
    l_max = order_from_ncoef(odf.shape[0])
    if basis == "tournier":
        B = real_sh_tournier(sh_order_max=l_max, theta=theta, phi=phi)[0]
    else:
        B = real_sh_descoteaux(sh_order_max=l_max, theta=theta, phi=phi)[0]
    odf_amp = B @ odf
    odf_verts = sphere.vertices * odf_amp
    ax.plot_trisurf(
        odf_verts[:, 0],
        odf_verts[:, 1],
        odf_verts[:, 2],
        triangles=sphere.faces,
        color=color,
        shade=False,
        edgecolor="k",
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")

    return ax


def plot_multiple_odf(
    odfs,
    ax=None,
    color="blue",
    basis="tournier",
    alpha=1,
    linewidth=0.1,
    separation=1.5,
):
    """Plot a spherical orientation distribution function represented by spherical harmonic coefficients.

    Parameters
    ----------
    odfs : NumPy array
        Array of spherical harmonic coefficients representing the ODF.
    ax : Matplotlib axis, optional
        Axis to plot the ODF on, by default None (creates a new figure).
    color : str, optional
        Color for ODF, by default 'blue'
    basis : str, optional
        Either "tournier" or "descoteaux", by default "tournier". (See https://docs.dipy.org/stable/theory/sh_basis.html).
    alpha : float, optional
        Opacity, by default 1
    linewidth : float, optional
        Linewidth for the ODF, by default 0.1
    separation : float, optional
        Separation between ODFs, by default 0.5

    Returns
    -------
    ax : Matplotlib axis
        Axis with the ODF plotted.
    """
    # if odf.ndim == 1:
    #     odf = odf[:, None]
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    sphere = unit_icosahedron.subdivide(n=4)
    x, y, z = sphere.vertices.T
    _, theta, phi = cart2sphere(x, y, z)
    l_max = order_from_ncoef(odfs.shape[1])
    if basis == "tournier":
        B = real_sh_tournier(sh_order_max=l_max, theta=theta, phi=phi)[0]
    else:
        B = real_sh_descoteaux(sh_order_max=l_max, theta=theta, phi=phi)[0]

    for i, odf in enumerate(odfs):
        odf_amp = B @ odf
        odf_verts = sphere.vertices * odf_amp[:, None]
        ax.plot_trisurf(
            odf_verts[:, 0] + i * separation,
            odf_verts[:, 1],
            odf_verts[:, 2],
            triangles=sphere.faces,
            color=color,
            shade=False,
            edgecolor="k",
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_aspect("equal")

    return ax


def plot_mesh(mesh, sphere, ax=None, cmap="cmc.batlow", alpha=1, clim=None):
    """Plot a hemispherical or spherical mesh with a scalar field.

    Parameters
    ----------
    mesh : (n_mesh,) array
        Scalar field on the mesh.
    sphere : Dipy Sphere
        Sphere to plot the mesh on.
    ax : Matplotlib axes, optional
        Axes to plot on, by default None
    cmap : str, optional
        Matplotlib or cmcrameri colormab, by default "cmc.batlow"
    alpha : int, optional
        Opacity, by default 1
    clim : tuple, optional
        Color limits, by default None

    Returns
    -------
    ax : Matplotlib axes
        Axes with the mesh plotted.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    if clim is None:
        clim = [0, mesh.max()]
    x, y, z = sphere.vertices.T

    triangles = Triangulation(x, y).triangles
    colors1 = np.mean(mesh[triangles], axis=1)
    triang = Triangulation(x, y, triangles)
    collec = ax.plot_trisurf(
        triang, z, cmap=cmap, shade=False, linewidth=0.0, alpha=alpha, antialiased=True
    )
    collec.set_array(colors1)
    collec.set_clim(clim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_aspect("equal")

    plt.colorbar(collec, ax=ax, fraction=0.04, orientation="horizontal")

    return ax


def wigner_d(l_max, m, m_prime, beta):
    """Calculate $$d_{m,m'}^l(beta)$$, the Wigner small-d matrix."""
    # See Rose, Elementary Theory of Angular Momentum, Eq. 4.13.
    # Compute the range of kappas
    kappa_min = max(0, m - m_prime)
    kappa_max = min(l_max + m, l_max - m_prime)
    kappas = np.arange(kappa_min, kappa_max + 1)

    # Initialize the sum
    d_sum = 0

    # Compute the sum
    for kappa in kappas:
        term = (
            (-1) ** kappa
            * np.cos(beta / 2) ** (2 * l_max + m - m_prime - 2 * kappa)
            * (-np.sin(beta / 2)) ** (m_prime - m + 2 * kappa)
            / (
                factorial(l_max - m_prime - kappa)
                * factorial(l_max + m - kappa)
                * factorial(kappa + m_prime - m)
                * factorial(kappa)
            )
        )
        d_sum += term

    # Compute the final value of d
    d = (
        np.sqrt(
            float(
                factorial(l_max + m)
                * factorial(l_max - m)
                * factorial(l_max + m_prime)
                * factorial(l_max - m_prime)
            )
        )
        * d_sum
    )
    return d


def wigner_D(l_max, m, m_prime, alpha, beta, gamma):
    """Calculate $D_{m,m'}^l(alpha, beta, gamma)$, the Wigner D matrix."""
    # See Rose, Elementary Theory of Angular Momentum, Eq. 4.12.
    # Compute the small d matrix
    d = wigner_d(l_max, m, m_prime, beta)

    # Compute the final value of D
    D = np.exp(-1j * m_prime * alpha) * d * np.exp(-1j * m * gamma)

    return D


def rotate_odf(sph_harm_coeff, rotation):
    """Rotate a function represented by spherical harmonic coefficients.

    Parameters
    ----------
    sph_harm_coeff : NumPy array
        Spherical harmonic coefficients represented in the mrtrix/Tournier07 basis (https://docs.dipy.org/stable/theory/sh_basis.html).
    rotation : Scipy Rotation object
        Rotation to apply to the function.

    Returns
    -------
    rotated_coeff : NumPy array
        Spherical harmonic coefficients of the rotated function in the mrtrix/Tournier07 basis.
    """
    # Get Euler angles
    alpha, beta, gamma = rotation.as_euler("zyz")
    l_max = order_from_ncoef(sph_harm_coeff.shape[0])

    # Convert coefficients by multiplying by the Wigner D matrix.
    # See Rose, Elementary Theory of Angular Momentum, Eq. 4.8.
    sph_harm_full = convert_sh_to_full_basis(sph_harm_coeff)
    rotated_coeff_full = np.zeros(sph_harm_full.shape, dtype=np.complex128)
    m_list_full, l_list_full = sph_harm_ind_list(l_max, full_basis=True)
    for i, (m, l) in enumerate(zip(m_list_full, l_list_full)):
        for m_prime in range(-l, l + 1):
            D = wigner_D(l, m, m_prime, alpha, beta, gamma)
            coeff = np.where((m_list_full == m_prime) & (l_list_full == l))[0]
            rotated_coeff_full[i] += D * np.complex128(sph_harm_full[coeff])

    # Convert back to the mrtrix/Tournier07 basis
    m_list, l_list = sph_harm_ind_list(l_max, full_basis=False)
    rotated_coeff = np.zeros(sph_harm_coeff.shape, dtype=np.complex128)
    for i, (m, l) in enumerate(zip(m_list, l_list)):
        coeff = rotated_coeff_full[np.where((m_list_full == m) & (l_list_full == l))[0]]
        if m != 0:
            coeff = coeff * np.sqrt(2)
        if m < 0:
            coeff = np.imag(coeff) * (-1) ** np.abs(m)
        elif m > 0:
            coeff = np.real(coeff)
        rotated_coeff[i] = coeff

    rotated_coeff = np.real(rotated_coeff)

    return rotated_coeff


def angular_corr_coeff(odf1, odf2):
    """Calculate the angular correlation coefficient between two ODFs.

    Parameters
    ----------
    odf1 : NumPy array
        Spherical harmonic coefficients representing the first ODF.
    odf2 : NumPy array
        Spherical harmonic coefficients representing the second ODF.

    Returns
    -------
    angular_corr : float
        Angular correlation coefficient between the two ODFs.
    """
    # De-mean
    odf1 = odf1[1:]
    odf2 = odf2[1:]

    # Calculate norm
    odf1_norm = np.sqrt(np.sum(odf1 * np.conj(odf1)))
    odf2_norm = np.sqrt(np.sum(odf2 * np.conj(odf2)))

    # Normalize
    odf1 = odf1 / odf1_norm
    odf2 = odf2 / odf2_norm

    angular_corr = np.real(np.sum(odf1 * np.conj(odf2)))
    return angular_corr


def match_odfs(true_odfs, est_odfs):
    """Match estimated ODFs to true ODFs.

    Parameters
    ----------
    true_odfs : NumPy array (Mx28)
        Array of true ODFs. Should be sorted by largest first for optimal.
    est_odfs : NumPy array (Nx28)
        Array of estimated ODFs.

    Returns
    -------
    matched_est_odfs : NumPy array (min(M, N)x28)
        Array of estimated ODFs matched to the true ODFs.
    index_array : NumPy array (m
        Array of indices of the matched estimated ODFs for indexing theta, phi, etc.
    """
    # # Are any of the true_odfs all zeros?
    # sum_coeff = np.sum(true_odfs, axis=1)
    # true_odfs = true_odfs[sum_coeff > 0.001]

    # Calculate ACC
    true_n_fibers = min(true_odfs.shape[0], est_odfs.shape[0])
    est_n_fibers = est_odfs.shape[0]

    # Initialize the index array
    index_array = np.zeros(true_n_fibers, dtype=int)

    # Create a working copy of estimated values
    est_odfs_copy = est_odfs.copy()

    # Iterate over each true value
    for i in range(true_n_fibers):
        # Calculate ACC to match
        weight = np.zeros(est_n_fibers)
        for j in range(est_n_fibers):
            acc = angular_corr_coeff(true_odfs[i, :], est_odfs_copy[j, :])
            weight[j] = acc  # (1-np.mean((true_odfs[i,:]-est_odfs_copy[j,:])**2))

            if np.isnan(weight[j]):
                weight[j] = -np.inf

        # Set ones in index array to -1
        weight[np.isinf(weight)] = -1

        # Find max ACC
        if np.all(np.isnan(weight)):
            break
        closest_index = np.argmax(weight)

        # Store the closest index
        index_array[i] = closest_index

        # Remove the matched element to prevent reuse
        est_odfs_copy[closest_index] = np.inf

    # Index array should have different numbers
    assert len(np.unique(index_array)) == len(index_array), "Matching failed."

    # Use the index array to reorder the estimated arrays if needed
    matched_est_odfs = est_odfs[index_array]

    return matched_est_odfs, index_array


def pdf2odfs(mesh, sphere, amp_threshold=0.5):
    """
    Estimate ODFs from spherical PDF

    Parameters
    ----------
    mesh : NumPy array
        Array of PDF values on the sphere mesh.
    sphere : Dipy Sphere
        Sphere mesh.
    amp_threshold : float, optional
        Amplitude threshold for maxima, by default 0.5

    Returns
    -------
    odfs : NumPy array (N, 28)
        Array of estimated ODFs.
    dirs : NumPy array
        Directions of maxima on the sphere mesh.
    vol_fracs : NumPy array
        Estimated volume fractions
    """

    theta = np.linspace(0, np.pi, 1000)
    phi = np.linspace(0, 2 * np.pi, 1000)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    points = np.array([sphere.theta, sphere.phi]).T
    pdf_mesh_interp = CloughTocher2DInterpolator(points, mesh, fill_value=0)
    rel_maxima = argrelmax(mesh, axis=0)

    # Get top 10 as initial points
    top_argmax = np.argsort(mesh[rel_maxima])[-10:]
    top_verts = sphere.vertices[rel_maxima[0][top_argmax]]
    top_theta, top_phi = (
        sphere.theta[rel_maxima[0][top_argmax]],
        sphere.phi[rel_maxima[0][top_argmax]],
    )

    # Minimize this
    initial_guesses = np.array([top_theta, top_phi]).T
    minima = []
    for initial_guess in initial_guesses:
        res = minimize(lambda x: -1 * pdf_mesh_interp(x), initial_guess)
        minima.append(res.x)

    # Get unique values and round to 4 decimals
    minima = np.array(minima)

    # Get values at minima
    minima_vals = pdf_mesh_interp(minima)
    minima_vals = minima_vals

    # Get round to 4 decimals and remove repeats for both
    minima = np.round(minima, 4)
    minima = np.unique(minima, axis=0)
    minima_vals = pdf_mesh_interp(minima)

    # Keep only minima > 0.1
    minima = minima[minima_vals > amp_threshold]
    minima_vals = minima_vals[minima_vals > amp_threshold]

    # Estimate volume fraction using ratio of amplitude at minima
    minima_vals = minima_vals / np.sum(minima_vals)

    # Now estimate ODF at these points with these volume fractions
    m_list, l_list = sph_harm_ind_list(6)
    odfs = []
    for min, min_val in zip(minima, minima_vals):
        odfs.append(
            convert_sh_descoteaux_tournier(
                gen_dirac(m_list, l_list, theta=min[0], phi=min[1])
            )
            * min_val
        )
    odfs = np.array(odfs)

    dirs = minima
    vol_fracs = minima_vals
    return odfs, dirs, vol_fracs


def angular_separation(angle1, angle2):
    """Calculate angular separation between two spherical angles.

    Parameters
    ----------
    angle1 : (theta, phi)
        First angle, with theta in [0, pi] and phi in [0, 2*pi].
    angle2 : (theta, phi)
        Second angle, with theta in [0, pi] and phi in [0, 2*pi].

    Returns
    -------
    delta : float
        Angular separation between the two angles.
    """
    theta1, phi1 = angle1
    theta2, phi2 = angle2
    cos_delta = np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + np.cos(
        theta1
    ) * np.cos(theta2)
    delta = np.arccos(cos_delta)
    return delta

def load_fissile_mat(path):
    """Load output from FISSILE and match estimated fibers to true fibers.

    Parameters
    ----------
    path : Path-like
        Path to the .mat file containing the output.

    Returns
    -------
    data_dict : list
        list of dict containing the data from the .mat file.
        - total_fod: 1x28 array sum of simulated fiber ODFs in lab frame
        - separate_fod: Mx28 array of estimated fiber ODFs in fiber frame
        - true_theta, true_phi, true_v: N array of true theta, phi, and volume fractions
        - est_theta, est_phi, est_v: M array of estimated theta, phi, and volume fractions
        - rotate_separate_fod: Mx28 array of rotated estimated ODFs in lab frame
        - true_fod: Mx28 array of simulated ODF in lab frame
        - acc: N array of angular correlation coefficients
        - est_theta_matched, est_phi_matched, est_v_matched: N array of estimated theta, phi, and volume fractions (matched to closest true fibers)
        - separate_fod_matched: Nx28 array of estimated fiber ODFs in fiber frame (matched to closest true fibers)
        - rotate_separate_fod_matched: Nx28 array of rotated estimated ODFs in lab frame (matched to closest true fibers)
        - true_fod_matched: Nx28 array of simulated ODF in lab frame (matched to closest true fibers)
    """
    # Load data
    data = loadmat(path)["data"]
    data_dict = [] * len(data[0])
    for num in range(len(data[0])):
        # Load data
        example = data[0][num]
        total_fod = example[0]
        separate_fod = example[1]
        true_theta = example[2][0] % np.pi
        true_phi = example[3][0] % (2 * np.pi)
        true_v = example[4][0]
        est_theta = example[5][0] % np.pi
        est_phi = example[6][0] % (2 * np.pi)
        est_v = example[7][0]

        # Only keeps ones where v > 0
        est_theta = est_theta[est_v > 0]
        est_phi = est_phi[est_v > 0]
        separate_fod = separate_fod[est_v > 0]
        est_v = est_v[est_v > 0]

        # Match true fibers to estimated fibers
        true_fod = np.zeros(true_theta.shape + (28,))
        for i in range(true_theta.shape[0]):
            m_list, l_list = sph_harm_ind_list(6)
            true_fod[i, :] = (
                convert_sh_descoteaux_tournier(
                    gen_dirac(m_list, l_list, theta=true_theta[i], phi=true_phi[i])
                )
                * true_v[i]
            )

        rotate_separate_fod = np.zeros(separate_fod.shape)
        for i in range(est_theta.shape[0]):
            rot = R.from_euler("ZYZ", [est_phi[i], -est_theta[i], 0])
            rotate_separate_fod[i, :] = rotate_odf(separate_fod[i, :], rot)

        _, index_array = match_odfs(true_fod, rotate_separate_fod)
        est_n_fibers = index_array.shape[0]
        est_theta_matched = est_theta[index_array]
        est_phi_matched = est_phi[index_array]
        est_v_matched = est_v[index_array]
        separate_fod_matched = separate_fod[index_array]
        rotate_separate_fod_matched = rotate_separate_fod[index_array]

        # Calculate ACC
        acc = np.zeros(est_n_fibers)
        true_fod_matched = true_fod[:est_n_fibers]
        for i in range(est_n_fibers):
            acc[i] = angular_corr_coeff(
                true_fod_matched[i, :], rotate_separate_fod_matched[i, :]
            )

        data_dict.append(
            {
                "total_fod": total_fod,
                "separate_fod": separate_fod,
                "true_theta": true_theta,
                "true_phi": true_phi,
                "true_v": true_v,
                "est_theta": est_theta,
                "est_phi": est_phi,
                "est_v": est_v,
                "rotate_separate_fod": rotate_separate_fod,
                "true_fod": true_fod,
                "acc": acc,
                "est_theta_matched": est_theta_matched,
                "est_phi_matched": est_phi_matched,
                "est_v_matched": est_v_matched,
                "separate_fod_matched": separate_fod_matched,
                "rotate_separate_fod_matched": rotate_separate_fod_matched,
                "true_fod_matched": true_fod_matched,
            }
        )

    return data_dict
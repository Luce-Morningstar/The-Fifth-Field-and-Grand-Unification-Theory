import cupy as cp
from cupyx.scipy.special import sph_harm
from legendre_cupy import generate_legendre_matrix  # our custom HELLBLAST P_lm builder

def alm2map(alm, theta, phi, lmax):
    """
    Compute the map on the sphere from a set of alm coefficients using CuPy.

    Parameters:
    - alm (cp.ndarray): 1D array of complex spherical harmonic coefficients.
    - theta (cp.ndarray): 1D array of theta angles.
    - phi (cp.ndarray): 1D array of phi angles.
    - lmax (int): Maximum multipole moment.

    Returns:
    - map (cp.ndarray): Real-valued collapse field on the sphere.
    """
    n_pix = len(theta)
    assert len(phi) == n_pix

    # Create alm index function (like healpy.getidx)
    def alm_index(l, m):
        return l * (l + 1) // 2 + m

    # Build associated Legendre matrix [l, m, pix]
    P_lm = generate_legendre_matrix(theta, lmax)  # shape: (l+1, m+1, npix)

    # Output field
    result = cp.zeros(n_pix, dtype=cp.complex64)

    for l in range(lmax + 1):
        for m in range(0, l + 1):
            idx = alm_index(l, m)
            coeff = alm[idx] if idx < len(alm) else 0.0

            exp_imphi = cp.exp(1j * m * phi)
            Y_lm = P_lm[l, m] * exp_imphi

            result += coeff * Y_lm

    return cp.real(result)

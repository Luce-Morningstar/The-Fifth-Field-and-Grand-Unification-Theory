import cupy as cp
import numpy as np
from math import pi
from scipy.special import lpmv  # still needed unless we reimplement in CuPy
from legendre_cupy import generate_legendre_matrix

def real_spherical_harmonic(l, m, theta, phi):
    norm = cp.sqrt((2*l + 1) / (4 * pi) * np.math.factorial(l - abs(m)) / np.math.factorial(l + abs(m)))
    P_lm = cp.asarray(lpmv(abs(m), l, cp.cos(theta)))  # fallback to CPU Legendre
    if m == 0:
        return norm * P_lm
    elif m > 0:
        return cp.sqrt(2.0) * norm * P_lm * cp.cos(m * phi)
    else:
        return cp.sqrt(2.0) * norm * P_lm * cp.sin(-m * phi)

def alm2map(alm, theta, phi, lmax):
    result = cp.zeros_like(theta, dtype=cp.float32)
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            Y_lm = real_spherical_harmonic(l, m, theta, phi)
            result += alm[l, m].real * Y_lm
    return result

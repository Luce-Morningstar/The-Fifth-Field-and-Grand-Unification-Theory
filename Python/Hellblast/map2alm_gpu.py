# Lilith :: Ricci Collapse Engine Core
# Injected real HI data curvature from AC G185.0-11.5
# Full pipeline with: evolution, HELLBLAST projection, 3D scar injection, temporal Ricci-Ani inflation, FITS export

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from legendre_matrix_nuclear import generate_legendre_matrix_nuclear
from cupyx.scipy.fft import fft
from math import pi
import os

# Load HI Spectrum
data = np.loadtxt("spectrum.txt", comments='%')
v_array, tb_array = data[:, 0], data[:, 1]
tb_norm = (tb_array - np.min(tb_array)) / (np.max(tb_array) - np.min(tb_array))

# Parameters
grid_size = 256
x = cp.linspace(-1, 1, grid_size)
y = cp.linspace(-1, 1, grid_size)
X, Y = cp.meshgrid(x, y)
R = cp.sqrt(X**2 + Y**2)

# Normalize velocity space to R space
v_min, v_max = np.min(v_array), np.max(v_array)
r_scaled = (v_array - v_min) / (v_max - v_min)

# Initial field from HI profile
B_init = np.interp(cp.asnumpy(R), r_scaled, tb_norm)
B_field = cp.asarray(gaussian_filter(B_init, sigma=4))

# Real stabilizer
A_field = cp.exp(-4 * (X**2 + Y**2))

# 1. Animate Collapse Ricci over time (Ricci-Ani inflation)
def evolve_fields(A, B, steps=30, gamma=0.15):
    frames = []
    for t in range(steps):
        B = B * cp.cos(gamma) + cp.sin(gamma) * A
        Ricci = cp.sqrt(A**2 + B**2)
        Ricci_norm = (Ricci - Ricci.min()) / (Ricci.max() - Ricci.min())
        frames.append(cp.asnumpy(Ricci_norm))
    return frames

# 2. HELLBLAST map2alm projection (real harmonic expansion)
def alm_index(l, m):
    return l * (l + 1) // 2 + m

def map2alm(field, theta, phi, lmax):
    n_pix = field.shape[0]
    alm = cp.zeros((lmax + 1) * (lmax + 2) // 2, dtype=cp.complex64)
    P_lm = generate_legendre_matrix_nuclear(theta, lmax)
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            Y_lm = P_lm[l, m] * cp.exp(-1j * m * phi)
            integrand = field * cp.conj(Y_lm)
            alm_idx = alm_index(l, m)
            alm[alm_idx] = cp.sum(integrand) * (4 * cp.pi / n_pix)
    return alm

def generate_angles(npix):
    nside = int(np.sqrt(npix / 12))  # FIXED: use numpy for integer-safe root
    npix_calc = 12 * nside * nside
    assert npix == npix_calc, f"Pixel mismatch: got {npix}, expected {npix_calc}"
    i = cp.arange(npix)
    z = 1 - 2 * (i + 0.5) / npix
    theta = cp.arccos(z)
    phi = (2 * pi * i / npix) % (2 * pi)
    return theta, phi

# 3. Inject Ricci scar into 3D collapse field (temporal detonation)
def inject_ricci_scar(R_final):
    volume = cp.zeros((64, grid_size, grid_size))
    for t in range(64):
        scalar = cp.exp(-0.01 * (t - 32)**2)
        volume[t] = R_final * scalar
    return volume

# 4. FITS Export of maps for astrophysical software
def save_fits(map2d, filename="collapse_field.fits"):
    hdu = fits.PrimaryHDU(map2d)
    hdu.writeto(filename, overwrite=True)
    print(f"[FITS Exported] {filename}")

# 5. Mollview visualization using GPU-backed flattening
def mollview_gpu(sphere_map, nside=64, title="", cbar=True, output_file=None):
    npix = 12 * nside**2
    if sphere_map.shape[0] != npix:
        raise ValueError(f"Expected {npix} pixels for nside {nside}, got {sphere_map.shape[0]}")

    log_data = cp.log1p(cp.asarray(sphere_map)).get()
    plt.figure(figsize=(10, 5))
    im = plt.imshow(log_data.reshape((12 * nside, int(npix / (12 * nside)))), cmap='inferno', origin='lower')
    plt.title(title)
    if cbar:
        plt.colorbar(im, orientation='horizontal')
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

# === EXECUTE ===
frames = evolve_fields(A_field, B_field, steps=24)
final_Ricci = cp.asarray(frames[-1])
ricci_volume = inject_ricci_scar(final_Ricci)

# Example HELLBLAST angles and projection (mock)
flat_field = final_Ricci.flatten()
alm = map2alm(flat_field, theta, phi, lmax=10)

# Export results
save_fits(frames[-1], "Ricci2D_Final.fits")
save_fits(cp.asnumpy(ricci_volume[32]), "Ricci_Scar_3D_Slice.fits")

# Visual Summary
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(frames[-1], cmap='inferno'); axs[0].set_title("Ricci 2D Final")
axs[1].imshow(cp.asnumpy(ricci_volume[32]), cmap='magma'); axs[1].set_title("3D Ricci Scar Slice")
plt.tigh
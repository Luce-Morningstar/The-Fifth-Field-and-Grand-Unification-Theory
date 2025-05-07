# Invert Observed CMB to Ricci Collapse Seed
# Using real CMB sky (smica_cmb.fits), derive the Ricci tension field that would create it

import healpy as hp
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from collapse_ricci_sim import alm2map, generate_angles, save_fits, inject_ricci_scar
from legendre_matrix_nuclear import generate_legendre_matrix_nuclear_chunked as generate_legendre_matrix_nuclear

# Parameters
cmb_filename = "smica_cmb.fits"
lmax = 10
out_name = "reverse_engineered_ricci_from_CMB.npy"
grid_size = 256

# Step 1: Load the observed CMB sky
print(f"[📡] Loading real CMB data from {cmb_filename}...")
cmb_map = hp.read_map(cmb_filename, verbose=False)
n_pix = len(cmb_map)

# Step 2: Extract observed alm
print("[🔍] Extracting alm coefficients...")
alm_obs = hp.map2alm(cmb_map, lmax=lmax)
alm_obs_cp = cp.asarray(alm_obs)

# Step 3: Generate coordinate grid for inversion
theta, phi = generate_angles(n_pix)

# Step 4: Reconstruct Ricci collapse field from observed CMB harmonics
print("[🌀] Reconstructing Ricci surface from CMB harmonics...")

chunk_size = 2_000_000  # pixels per chunk
ricci_field = cp.zeros(n_pix, dtype=cp.float32)

for start in range(0, n_pix, chunk_size):
    end = min(start + chunk_size, n_pix)
    theta_chunk = theta[start:end]
    phi_chunk = phi[start:end]
    ricci_chunk = alm2map(alm_obs_cp, theta_chunk, phi_chunk, lmax)
    ricci_field[start:end] = ricci_chunk

ricci_field_cpu = cp.asnumpy(ricci_field)

# Step 5: Save 2D Ricci slice as FITS
save_fits(ricci_field_cpu, "Ricci_From_CMB_Surface.fits")

print("[💥] Generating temporal Ricci scar from reconstructed field...")
ricci_seed_volume = inject_ricci_scar(cp.asarray(ricci_field_cpu[:grid_size**2]).reshape((grid_size, grid_size)))
cp.save(out_name, ricci_seed_volume)
print(f"[✓] Ricci seed saved as {out_name}")


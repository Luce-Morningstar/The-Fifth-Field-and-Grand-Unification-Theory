
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import correlate
from scipy.special import rel_entr

# === CONFIG ===
planck_file = "smica_cmb.fits"
collapse_dir = "lilith_turbo_output"
nside = 512
recent_n = 100

def volume_to_healpix(field_3d, nside=512):
    size = field_3d.shape[0]
    x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
    dx, dy, dz = x - size // 2, y - size // 2, z - size // 2
    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    theta = np.arccos(dz / r).flatten()
    phi = np.arctan2(dy, dx).flatten() % (2 * np.pi)
    values = field_3d.flatten()
    pix = hp.ang2pix(nside, theta, phi)
    return np.bincount(pix, weights=values, minlength=hp.nside2npix(nside))

def slice_to_healpix(slice_2d, size=256, nside=512):
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    z = np.full_like(x, size // 2)
    dx, dy, dz = x - size // 2, y - size // 2, z - size // 2
    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    theta = np.arccos(dz / r).flatten()
    phi = np.arctan2(dy, dx).flatten() % (2 * np.pi)
    values = slice_2d.flatten()
    pix = hp.ang2pix(nside, theta, phi)
    return np.bincount(pix, weights=values, minlength=hp.nside2npix(nside))

def load_and_flatten(fname):
    path = os.path.join(collapse_dir, fname)
    raw = np.load(path)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw -= np.min(raw)
    if np.max(raw) > 0:
        raw /= np.max(raw)
    if raw.ndim == 3:
        return volume_to_healpix(raw, nside)
    elif raw.ndim == 2:
        return slice_to_healpix(raw, size=raw.shape[0], nside=nside)
    elif raw.ndim == 1:
        return raw
    return None

print("Loading Planck SMICA map...")
planck_map = hp.read_map(planck_file, field=0)
planck_map = hp.ud_grade(planck_map, nside_out=nside)

npy_files = sorted([
    f for f in os.listdir(collapse_dir)
    if f.endswith(".npy") and any(c.isdigit() for c in f)
], key=lambda x: int(''.join(filter(str.isdigit, x))))[-recent_n:]


print(f"Loading and flattening {len(npy_files)} real field collapse maps...")
flattened_maps = []
weights = []

with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(load_and_flatten, npy_files), total=len(npy_files)))

for i, result in enumerate(results):
    if result is not None and np.std(result) > 1e-6:
        flattened_maps.append(result)
        weights.append(i + 1)

weights = np.array(weights)
weights = weights / np.sum(weights)

collapse_maps = {
    "FlatAverage": np.mean(flattened_maps, axis=0),
    "TimeWeighted": np.average(flattened_maps, axis=0, weights=weights)
}

for label, collapse_map in collapse_maps.items():
    print(f"\n--- Processing {label} Collapse Map ---")
    residual_map = collapse_map - planck_map

    hp.mollview(collapse_map, title=f"{label} Collapse Field", cmap="inferno")
    plt.savefig(f"{label.lower()}_collapse_map.png")

    hp.mollview(residual_map, title=f"{label} - Planck Residual", cmap="seismic")
    plt.savefig(f"{label.lower()}_vs_planck_diff.png")

    corr = np.corrcoef(collapse_map, planck_map)[0, 1]
    print(f"Correlation ({label} vs Planck): {corr:.6f}")

    Cl_planck = hp.anafast(planck_map)
    Cl_collapse = hp.anafast(collapse_map)
    Cl_cross = hp.anafast(collapse_map, planck_map)

    plt.figure()
    plt.plot(Cl_planck, label="Planck Cℓ")
    plt.plot(Cl_collapse, label=f"{label} Cℓ")
    plt.plot(Cl_cross, label="Cross Cℓ")
    plt.yscale("log")
    plt.legend()
    plt.title(f"{label} Power Spectrum Comparison")
    plt.savefig(f"{label.lower()}_power_spectrum_comparison.png")

    P = Cl_planck / np.sum(Cl_planck)
    Q = Cl_collapse / np.sum(Cl_collapse)
    kl_div = np.sum(rel_entr(P, Q))
    print(f"KL Divergence ({label} || Planck): {kl_div:.6f}")

    plt.figure()
    plt.plot(P, label="Planck P(ℓ)", alpha=0.7)
    plt.plot(Q, label=f"{label} Q(ℓ)", alpha=0.7)
    plt.title(f"{label} Normalized Power Spectra")
    plt.xlabel("ℓ")
    plt.ylabel("Normalized Cℓ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{label.lower()}_normalized_spectra.png")

    hp.gnomview(residual_map, rot=(45, 45), xsize=400, title=f"{label} Cold Spot Zoom", cmap="seismic")
    plt.savefig(f"{label.lower()}_zoomed_cold_spot.png")

    collapse_norm = (collapse_map - np.mean(collapse_map)) / np.std(collapse_map)
    planck_norm = (planck_map - np.mean(planck_map)) / np.std(planck_map)
    acf = correlate(collapse_norm, planck_norm, mode='same')

    plt.figure()
    plt.plot(acf, label="Angular Cross-Correlation")
    plt.title(f"{label} Angular Correlation")
    plt.xlabel("Pixel Shift")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{label.lower()}_angular_correlation.png")

    Cl_diff = np.abs(Cl_planck - Cl_collapse)
    l = np.arange(len(Cl_planck))
    bins = [(0, 50), (51, 200), (201, 500), (501, len(Cl_planck))]
    residuals = [np.mean(Cl_diff[start:end]) for start, end in bins]
    labels = [f"{start}-{end-1}" for start, end in bins]

    plt.figure()
    plt.bar(labels, residuals)
    plt.title(f"{label} ℓ-binned Mean |ΔCℓ|")
    plt.xlabel("ℓ-Range")
    plt.ylabel("Mean Absolute Residual")
    plt.tight_layout()
    plt.savefig(f"{label.lower()}_ell_binned_residuals.png")

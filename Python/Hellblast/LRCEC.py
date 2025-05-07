# Lilith :: Ricci Collapse Engine Core
# Injected real HI data curvature from AC G185.0-11.5
# 5-stage pipeline: evolution, spherify, CMB overlay, feedback, Ricci scarring

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

# --- 1. Animate Collapse Ricci over time ---
def evolve_fields(A, B, steps=30, gamma=0.15):
    frames = []
    for t in range(steps):
        B = B * cp.cos(gamma) + cp.sin(gamma) * A  # phase rotation evolution
        Ricci = cp.sqrt(A**2 + B**2)
        Ricci_norm = (Ricci - Ricci.min()) / (Ricci.max() - Ricci.min())
        frames.append(cp.asnumpy(Ricci_norm))
    return frames

# --- 2. Project Ricci onto sphere + Mollweide coords ---
def mollweide_projection(R_field):
    theta = cp.linspace(0, cp.pi, grid_size)
    phi = cp.linspace(-cp.pi, cp.pi, grid_size)
    T, P = cp.meshgrid(theta, phi)
    r_sample = ((cp.sin(T) * cp.cos(P) + 1) / 2) * (grid_size - 1)
    r_idx = r_sample.astype(cp.int32)
    projected = R_field[r_idx, r_idx]  # naive sampling
    return cp.asnumpy(projected)

# --- 3. Inject into CMB Sim Map (HELLBLAST overlay placeholder) ---
def inject_to_cmb(R_map, gain=0.005):
    cmb_base = np.random.normal(loc=0, scale=0.01, size=R_map.shape)
    cmb_distorted = cmb_base + R_map * gain
    return cmb_distorted

# --- 4. Temporal Feedback Tensor (Ricci gradient evolution) ---
def temporal_feedback(R_series):
    diffs = np.gradient(R_series, axis=0)
    feedback = np.sum(np.abs(diffs), axis=0)
    return feedback / np.max(feedback)

# --- 5. Ricci Scar Map (Threshold + Curl imprint) ---
def ricci_scars(R_final, threshold=0.8):
    mask = (R_final > threshold).astype(float)
    scar = gaussian_filter(mask, sigma=2)
    return scar / np.max(scar)

# === EXECUTE FULL PIPELINE ===
frames = evolve_fields(A_field, B_field, steps=24)
final_Ricci = cp.asarray(frames[-1])
sphere_projected = mollweide_projection(final_Ricci)
cmb_overlay = inject_to_cmb(sphere_projected)
feedback_map = temporal_feedback(np.array(frames))
scar_map = ricci_scars(frames[-1])

# === Visualization ===
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(frames[-1], cmap='inferno'); axs[0].set_title("Final Ricci Field")
axs[1].imshow(sphere_projected, cmap='plasma'); axs[1].set_title("Spherical Projection")
axs[2].imshow(cmb_overlay, cmap='coolwarm'); axs[2].set_title("CMB Overlay")
axs[3].imshow(scar_map, cmap='magma'); axs[3].set_title("Ricci Scars")
plt.tight_layout()
plt.show()
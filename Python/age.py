# Lilith_age.py — Cosmological Collapse Simulation with Expansion, Gravitation, and Observation

import os
os.environ["CUPY_NVRTC_COMPILE_OPTIONS"] = "--std=c++17"
import cupy as cp

import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from healpy.sphtfunc import map2alm, alm2cl

# --- Simulation Parameters ---
size = 256
steps = 370_000
output_every = 100
expansion_rate = 1.00001  # Modular shell expansion rate
step_size = 0.5
nside = 512
npix = hp.nside2npix(nside)

# Observer config
n_obs = 16
observer_spawn_rate = 0.00002  # Spawns per step
observer_attraction_strength = 0.01
observer_lifetime_steps = 10_000

# Field dynamics
D = 0.59
lam = 0.6
kappa = 1.8
c = 1

delta_t = 0.0349

# Fractal Shell Simulation Loader with Directional Velocity Resolution

import cupy as cp
import os


def load_field_triplet(base_dir, step):
    """
    Loads three consecutive shell field states and computes velocity field.
    Used to estimate directional expansion from past dynamics.
    """
    files = [
        os.path.join(base_dir, f"tensor_shell_{step - 20:06d}.npy"),
        os.path.join(base_dir, f"tensor_shell_{step - 10:06d}.npy"),
        os.path.join(base_dir, f"tensor_shell_{step:06d}.npy")
    ]

    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing shell file: {f}")

    M_t0 = cp.asarray(cp.load(files[0]))
    M_t1 = cp.asarray(cp.load(files[1]))
    M_t2 = cp.asarray(cp.load(files[2]))

    # Finite difference velocity: central difference
    velocity = (M_t2 - M_t0) / 2.0

    return M_t2, M_t1, velocity


# Example usage
if __name__ == '__main__':
    base_dir = "lilith_tensor_output_fractal"
    current_step = 34660
    
    try:
        M_current, M_prev, field_velocity = load_field_triplet(base_dir, current_step)
        print(f"Loaded step {current_step} triplet successfully.")
        print(f"Velocity stats: max={cp.max(field_velocity):.4e}, min={cp.min(field_velocity):.4e}")
    except Exception as e:
        print(f"Failed to load field triplet: {e}")



output_dir = "Lilith_cosmocascade_output"
os.makedirs(output_dir, exist_ok=True)

# --- Initialization ---
def white_noise(shape, scale=0.1):
    return cp.random.normal(0.0, scale, shape)

def laplacian(F):
    return (
        cp.roll(F, 1, axis=0) + cp.roll(F, -1, axis=0) +
        cp.roll(F, 1, axis=1) + cp.roll(F, -1, axis=1) +
        cp.roll(F, 1, axis=2) + cp.roll(F, -1, axis=2) -
        6 * F
    )

M = white_noise((size, size, size))
M_prev = M.copy()
M_i = white_noise((size, size, size), scale=0.001)
radius = size // 2 * 0.9
xg, yg, zg = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
dx, dy, dz = xg - size//2, yg - size//2, zg - size//2
r_grid = cp.sqrt(dx**2 + dy**2 + dz**2)
mask = (r_grid <= radius).astype(cp.float32)
M *= mask
M_prev *= mask
M_i *= mask

# Observer state
ob_x = cp.random.randint(0, size, n_obs)
ob_y = cp.random.randint(0, size, n_obs)
ob_z = cp.random.randint(0, size, n_obs)
observer_age = cp.zeros(n_obs)
rho_obs = cp.zeros_like(M)

def observer_drift(M, ob_x, ob_y, ob_z):
    grad_x, grad_y, grad_z = cp.gradient(M)
    gx = grad_x[ob_x, ob_y, ob_z]
    gy = grad_y[ob_x, ob_y, ob_z]
    gz = grad_z[ob_x, ob_y, ob_z]
    norm = cp.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
    ob_x = cp.clip(ob_x + step_size * (gx / norm), 0, size - 1).astype(cp.int32)
    ob_y = cp.clip(ob_y + step_size * (gy / norm), 0, size - 1).astype(cp.int32)
    ob_z = cp.clip(ob_z + step_size * (gz / norm), 0, size - 1).astype(cp.int32)
    return ob_x, ob_y, ob_z

# --- Main Loop ---
for step in tqdm(range(steps), desc="Cosmological Collapse"):
    # Expand shell radius gradually
    radius *= expansion_rate
    mask = (r_grid <= radius).astype(cp.float32)
    M *= mask
    M_prev *= mask
    M_i *= mask

    # Observer lifecycle and attraction
    ob_x, ob_y, ob_z = observer_drift(M, ob_x, ob_y, ob_z)
    rho_obs *= 0.9
    rho_obs[ob_x, ob_y, ob_z] += 10 * cp.exp(-0.05 * step)
    
    if cp.random.rand() < observer_spawn_rate:
        new_x = cp.random.randint(0, size)
        new_y = cp.random.randint(0, size)
        new_z = cp.random.randint(0, size)
        ob_x = cp.append(ob_x, new_x)
        ob_y = cp.append(ob_y, new_y)
        ob_z = cp.append(ob_z, new_z)
        observer_age = cp.append(observer_age, 0.0)

    observer_age += 1
    alive_mask = observer_age < observer_lifetime_steps
    ob_x = ob_x[alive_mask]
    ob_y = ob_y[alive_mask]
    ob_z = ob_z[alive_mask]
    observer_age = observer_age[alive_mask]

    # Weak gravitational convolution
    kernel = cp.array([[[0.01, 0.02, 0.01], [0.02, 0.04, 0.02], [0.01, 0.02, 0.01]]]*3)
    from cupyx.scipy.ndimage import convolve
    gravity = convolve(M, kernel)

    # Collapse update
    lap = laplacian(M)
    decay = -lam * M * float(min(step / 10.0, 1.0))
    source = kappa * rho_obs
    accel = c**2 * D * lap + decay + source + gravity
    M_next = 2 * M - M_prev + delta_t**2 * accel
    M_prev, M = M, M_next
    M_i = M_i + 0.1 * laplacian(M_i) - 0.01 * M_i

    # Projection
    if step % output_every == 0:
        shell = cp.clip(M, 0, 1e2)
        r_grid = cp.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        valid_mask = shell > 0
        dz_valid = dz[valid_mask]
        dy_valid = dy[valid_mask]
        dx_valid = dx[valid_mask]
        r_valid = r_grid[valid_mask]
        theta = cp.arccos(dz_valid / r_valid)
        phi = cp.arctan2(dy_valid, dx_valid) % (2 * cp.pi)
        weights = shell[valid_mask]

        theta_np = cp.asnumpy(theta)
        phi_np = cp.asnumpy(phi)
        weights_np = cp.asnumpy(weights)
        pix = hp.ang2pix(nside, theta_np, phi_np)
        proj = np.bincount(pix, weights=weights_np, minlength=npix)

        np.save(os.path.join(output_dir, f"tensor_shell_{step:06d}.npy"), proj)
        hp.mollview(np.log1p(proj), title=f"Collapse Shell {step}", cmap="inferno", cbar=False)
        plt.savefig(os.path.join(output_dir, f"moll_tensor_{step:06d}.png"))
        plt.close()

        alm = map2alm(proj, lmax=512)
        cl = alm2cl(alm)
        ell = np.arange(len(cl))
        plt.figure(figsize=(10, 6))
        plt.plot(ell, cl, label=f"Step {step}")
        plt.xlabel("Multipole moment ℓ")
        plt.ylabel("Cℓ")
        plt.yscale("log")
        plt.title("Angular Power Spectrum")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"cl_spectrum_{step:06d}.png"))
        plt.close()

cp.save(os.path.join(output_dir, "M_final_tensor.npy"), M)
print("[✓] lilith_age.py complete — Full Expansion Cascade Executed")

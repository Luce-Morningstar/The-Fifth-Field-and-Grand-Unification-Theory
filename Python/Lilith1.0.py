
import cupy as cp
import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.signal import correlate
from healpy.sphtfunc import map2alm, alm2cl

# Simulation Params
size = 256
steps = 50000
delta_t = 0.349
c = 1 
D = 0.25
lam = 8.5
kappa = 5
nside = 512
n_obs = 32
step_size = 0.5
max_layers = 2
shell_scale_factor = 0.5
observer_lifetime = 400
observer_function_limit = 20
observer_decay_rate = 0.85
observer_mobility_decay = 0.50
observer_replication_threshold = 0.25
observer_replication_rate = 0.0001
memory_decay = 0.990
boundary_threshold = 0.1

# Unique per-run directory
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"Lilith_tensor_output_fractal_{run_id}"
os.makedirs(output_dir, exist_ok=True)

def white_noise_field(shape, scale=0.1):
    noise = cp.random.normal(loc=0.0, scale=scale, size=shape)
    freq_noise = cp.fft.fftn(noise)
    random_phase = cp.exp(2j * cp.pi * cp.random.rand(*shape))
    filtered = cp.real(cp.fft.ifftn(freq_noise * random_phase))
    return filtered

# Per-layer storage
M_layers = []
M_prev_layers = []
M_i_layers = []
rho_obs_layers = []
shell_masks = []
shell_surfaces = []
radius_shells = []
observer_states = []
nucleation_fields = []
observer_counts = []
memory_fields = []

npix = hp.nside2npix(nside)

# Generate fractal layers
for i in range(max_layers):
    scale = shell_scale_factor ** i
    grid_size = size
    center = grid_size // 2
    xg, yg, zg = cp.meshgrid(cp.arange(grid_size), cp.arange(grid_size), cp.arange(grid_size), indexing='ij')
    dx, dy, dz = xg - center, yg - center, zg - center
    radius_grid = cp.sqrt(dx**2 + dy**2 + dz**2)
    radius_shell = radius_grid.astype(cp.int32)
    shell_max = int(radius_grid.max() * scale)
    mask = (radius_grid <= shell_max).astype(cp.float32)
    surface = ((radius_grid >= shell_max - 1.5) & (radius_grid <= shell_max)).astype(cp.float32)

    M = white_noise_field((grid_size, grid_size, grid_size)) * 0.1 * (1.0 / (1 + i))
    M_prev = M.copy()
    M_i = white_noise_field((grid_size, grid_size, grid_size), scale=0.001)
    rho_obs = cp.zeros_like(M)

    ob_x = cp.random.randint(0, grid_size, n_obs)
    ob_y = cp.random.randint(0, grid_size, n_obs)
    ob_z = cp.random.randint(0, grid_size, n_obs)
    ob_age = cp.zeros(n_obs, dtype=cp.int32)
    ob_fn = cp.zeros(n_obs, dtype=cp.int32)
    ob_alive = cp.ones(n_obs, dtype=cp.bool_)
    ob_mob = cp.ones(n_obs, dtype=cp.float32)

    M_layers.append(M * mask)
    M_prev_layers.append(M_prev * mask)
    M_i_layers.append(M_i * mask)
    rho_obs_layers.append(rho_obs)
    radius_shells.append(radius_shell)
    shell_masks.append(mask)
    shell_surfaces.append(surface)
    observer_states.append({"x": ob_x, "y": ob_y, "z": ob_z, "age": ob_age, "fn": ob_fn, "alive": ob_alive, "mobility": ob_mob})
    nucleation_fields.append(cp.zeros_like(M))
    observer_counts.append([])
    memory_fields.append(cp.zeros_like(M))

def laplacian_3d(F):
    return (
        cp.roll(F, 1, axis=0) + cp.roll(F, -1, axis=0) +
        cp.roll(F, 1, axis=1) + cp.roll(F, -1, axis=1) +
        cp.roll(F, 1, axis=2) + cp.roll(F, -1, axis=2) -
        6 * F
    )

def observer_drift(M, ob, radius_shell, shell_max, step_size=1):
    pot = M + 0.5 * laplacian_3d(M)
    grad_x, grad_y, grad_z = cp.gradient(pot)
    gx = grad_x[ob["x"], ob["y"], ob["z"]]
    gy = grad_y[ob["x"], ob["y"], ob["z"]]
    gz = grad_z[ob["x"], ob["y"], ob["z"]]
    norm = cp.sqrt(gx**2 + gy**2 + gz**2) + 1e-6

    ob["mobility"] *= observer_mobility_decay

    x_c, y_c, z_c = ob["x"], ob["y"], ob["z"]
    x_mean = cp.mean(x_c)
    y_mean = cp.mean(y_c)
    z_mean = cp.mean(z_c)
    cx = x_mean - x_c
    cy = y_mean - y_c
    cz = z_mean - z_c
    c_norm = cp.sqrt(cx**2 + cy**2 + cz**2) + 1e-6
    cx /= c_norm
    cy /= c_norm
    cz /= c_norm
    cohesion_weight = 0.9
    gx = (1 - cohesion_weight) * gx + cohesion_weight * cx
    gy = (1 - cohesion_weight) * gy + cohesion_weight * cy
    gz = (1 - cohesion_weight) * gz + cohesion_weight * cz

    ix = cp.gradient(M_i_layers[0], axis=0)[ob["x"], ob["y"], ob["z"]]
    iy = cp.gradient(M_i_layers[0], axis=1)[ob["x"], ob["y"], ob["z"]]
    iz = cp.gradient(M_i_layers[0], axis=2)[ob["x"], ob["y"], ob["z"]]
    i_norm = cp.sqrt(ix**2 + iy**2 + iz**2) + 1e-6
    imaginary_weight = 0.5  # Can be dynamically scaled
    gx = (1 - imaginary_weight) * gx + imaginary_weight * (ix / i_norm)
    gy = (1 - imaginary_weight) * gy + imaginary_weight * (iy / i_norm)
    gz = (1 - imaginary_weight) * gz + imaginary_weight * (iz / i_norm)

    # Optional: add jitter
    gx += 0.0001 * cp.random.normal(size=gx.shape)
    gy += 0.0001 * cp.random.normal(size=gy.shape)
    gz += 0.0001 * cp.random.normal(size=gz.shape)

    norm = cp.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
    x_new = cp.clip(ob["x"] + ob["mobility"] * step_size * (gx / norm), 0, size - 1).astype(cp.int32)
    y_new = cp.clip(ob["y"] + ob["mobility"] * step_size * (gy / norm), 0, size - 1).astype(cp.int32)
    z_new = cp.clip(ob["z"] + ob["mobility"] * step_size * (gz / norm), 0, size - 1).astype(cp.int32)

    r_obs = radius_shell[x_new, y_new, z_new]
    shell_hit = (r_obs >= shell_max)
    x_new[shell_hit] = size // 2
    y_new[shell_hit] = size // 2
    z_new[shell_hit] = size // 2

    return x_new, y_new, z_new


for step in tqdm(range(steps), desc="Fractal Tensor Cascade"):
    for i in range(len(M_layers)):
        M, M_prev, M_i, rho_obs = M_layers[i], M_prev_layers[i], M_i_layers[i], rho_obs_layers[i]
        ob = observer_states[i]
        ob_x, ob_y, ob_z = ob["x"], ob["y"], ob["z"]

        radius_shell = radius_shells[i]
        shell_max = int(radius_shell.max())

        # Drift with mobility decay
        ob_x, ob_y, ob_z = observer_drift(M, ob, radius_shell, shell_max)
        ob["x"], ob["y"], ob["z"] = ob_x, ob_y, ob_z

        # Observer replication in coherent zones
        coherence_zone = cp.abs(M - M_prev) < 0.01
        coherent_indices = cp.where(coherence_zone)
        if len(coherent_indices[0]) > 0:
            sampled = cp.random.choice(len(coherent_indices[0]), size=1)
            new_x = coherent_indices[0][sampled]
            new_y = coherent_indices[1][sampled]
            new_z = coherent_indices[2][sampled]
            ob["x"] = cp.concatenate((ob["x"], new_x))
            ob["y"] = cp.concatenate((ob["y"], new_y))
            ob["z"] = cp.concatenate((ob["z"], new_z))
            ob["age"] = cp.concatenate((ob["age"], cp.zeros(1, dtype=cp.int32)))
            ob["fn"] = cp.concatenate((ob["fn"], cp.zeros(1, dtype=cp.int32)))
            ob["alive"] = cp.concatenate((ob["alive"], cp.ones(1, dtype=cp.bool_)))
            ob["mobility"] = cp.concatenate((ob["mobility"], cp.ones(1, dtype=cp.float32)))

        observer_counts[i].append(len(ob["x"]))

        rho_obs *= 0.1
        rho_obs[ob["x"], ob["y"], ob["z"]] += 5 * cp.exp(-0.05 * step)

        lap = laplacian_3d(M)
        decay = -lam * M * float(min(step / 5.0, 1.0))
        source = kappa * rho_obs
        accel = c**2 * D * lap + decay + source

        M_next = 2 * M - M_prev + delta_t**2 * accel

        grad_mag = cp.sqrt(cp.sum(cp.stack(cp.gradient(M))**2, axis=0))
        coherence = cp.abs(M - M_prev)
        nucleation_fields[i] = cp.where((M > 0.05) & (coherence < 0.01), M, 0)

        if i + 1 < max_layers and i + 1 == len(M_layers):
            new_M = white_noise_field((size, size, size)) * 0.01
            shell_transfer = M[radius_shell == shell_max].mean()
            new_M[radius_shells[i] == 0] = shell_transfer
            M_layers.append(new_M)
            M_prev_layers.append(new_M.copy())
            M_i_layers.append(white_noise_field((size, size, size), scale=0.001))
            rho_obs_layers.append(cp.zeros_like(new_M))
            observer_states.append({
                "x": cp.random.randint(0, size, n_obs),
                "y": cp.random.randint(0, size, n_obs),
                "z": cp.random.randint(0, size, n_obs),
                "age": cp.zeros(n_obs, dtype=cp.int32),
                "fn": cp.zeros(n_obs, dtype=cp.int32),
                "alive": cp.ones(n_obs, dtype=cp.bool_),
                "mobility": cp.ones(n_obs, dtype=cp.float32)
            })
            nucleation_fields.append(cp.zeros_like(new_M))
            observer_counts.append([])
            memory_fields.append(cp.zeros_like(new_M))

        M_prev_layers[i] = M
        M_layers[i] = M_next
        M_i_layers[i] = M_i + 0.1 * laplacian_3d(M_i) - 0.01 * M_i

    if step % 10 == 0:
        combined_shell = cp.zeros((size, size, size))
        for i in range(len(M_layers)):
            combined_shell += M_layers[i] * shell_surfaces[i]

        shell_energy = cp.sum(combined_shell)
        #if shell_energy < 1e-6:
        #    continue

        r_grid = cp.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        valid_mask = combined_shell > 0
        dz_valid = dz[valid_mask]
        dy_valid = dy[valid_mask]
        dx_valid = dx[valid_mask]
        r_valid = r_grid[valid_mask]
        theta = cp.arccos(dz_valid / r_valid)
        phi = cp.arctan2(dy_valid, dx_valid) % (2 * cp.pi)
        weights = combined_shell[valid_mask]

        theta_np = cp.asnumpy(theta)
        phi_np = cp.asnumpy(phi)
        weights_np = cp.asnumpy(weights)

        pix = hp.ang2pix(nside, theta_np, phi_np)
        proj = np.bincount(pix, weights=weights_np, minlength=npix)

        np.save(os.path.join(output_dir, f"tensor_shell_{step:06d}.npy"), proj)
        hp.mollview(np.log1p(proj), title=f"Fractal Collapse Shell {step}", cmap="inferno", cbar=False)
        plt.savefig(os.path.join(output_dir, f"moll_tensor_{step:06d}.png"))
        plt.close()

        if step % 100 == 0:
         alm = map2alm(proj, lmax=2048)
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

if step % 500 == 0:
    try:
        # Load Planck Cl spectrum
        planck_cl = np.loadtxt("planck_2018_cls.txt")[:len(cl)]
        cl_norm = cl / (np.sum(cl) + 1e-12)
        planck_norm = planck_cl / (np.sum(planck_cl) + 1e-12)
        kl_val = entropy(cl_norm, planck_norm)
        corr_val = np.corrcoef(cl, planck_cl)[0, 1]
        ent_val = -np.sum(cl_norm * np.log(cl_norm + 1e-12))
        kl_log.append([step, kl_val, corr_val, ent_val])
        print(f"\n--- Step {step} KL ---\nKL: {kl_val:.6f}\nCorrelation: {corr_val:.6f}\nEntropy: {ent_val:.6f}")
    except Exception as e:
        print(f"[!] KL check failed at step {step}: {e}")

np.savetxt(os.path.join(output_dir, "kl_log.csv"), kl_log, delimiter=",", header="Step,KL,Correlation,Entropy")

cp.save(os.path.join(output_dir, "M_final_tensor.npy"), M_layers[0])
print("[✓] Lilith_tensordrive.py complete — Fractal Shell Topology Activated")

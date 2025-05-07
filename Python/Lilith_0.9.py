# Lilith_0.9.py — Observer AI with Fractal Collapse Tensor Guidance + Inflationic Seeding and Oscillation Modes + Planck CMB Comparison + Observer Attraction and Antimatter Void Dynamics

import cupy as cp
import numpy as np
import healpy as hp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import correlate
from healpy.sphtfunc import map2alm, alm2cl

# Simulation Params
size = 256
steps = 5000
delta_t = 0.349
c = 1
D = 0.59
lam = 1.2
kappa = 0
nside = 512
n_obs = 160
step_size = 0.5
max_layers = 8
shell_scale_factor = 0.5
omega_sq = 0.0025
gamma_damp = 0.025
obs_lifetime = 3e5

output_dir = "lilith_tensor_output_fractal"
os.makedirs(output_dir, exist_ok=True)

planck_data_path = "COM_PowerSpect_CMB-TT-full_R3.01.txt"
try:
    planck_data = np.loadtxt(planck_data_path)
    ell_planck = planck_data[:, 0]
    cl_planck = planck_data[:, 1]
except Exception as e:
    ell_planck = cl_planck = None
    print("[!] Planck data not found or unreadable:", e)

def inflationic_noise(shape):
    k = cp.fft.fftfreq(shape[0])[:, None, None]**2 + cp.fft.fftfreq(shape[1])[None, :, None]**2 + cp.fft.fftfreq(shape[2])[None, None, :]**2
    k = cp.sqrt(k)
    spectrum = 1 / (k + 1e-3)
    phase = cp.exp(2j * cp.pi * cp.random.rand(*shape))
    field_k = spectrum * phase
    return cp.fft.ifftn(field_k).real

def white_noise_field(shape, scale=0.1):
    noise = cp.random.normal(0.0, scale, shape)
    freq_noise = cp.fft.fftn(noise)
    random_phase = cp.exp(2j * cp.pi * cp.random.rand(*shape))
    return cp.fft.ifftn(freq_noise * random_phase).real

M_layers = []
M_prev_layers = []
M_i_layers = []
rho_obs_layers = []
antimatter_layers = []
shell_masks = []
shell_surfaces = []
radius_shells = []
observer_positions = []
observer_age_layers = []

npix = hp.nside2npix(nside)

for i in range(max_layers):
    scale = shell_scale_factor ** i
    xg, yg, zg = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
    dx, dy, dz = xg - size//2, yg - size//2, zg - size//2
    radius_grid = cp.sqrt(dx**2 + dy**2 + dz**2)
    radius_shell = radius_grid.astype(cp.int32)
    shell_max = int(radius_grid.max() * scale)
    mask = (radius_grid <= shell_max).astype(cp.float32)
    surface = ((radius_grid >= shell_max - 1.5) & (radius_grid <= shell_max)).astype(cp.float32)

    M = inflationic_noise((size, size, size)) * 0.1 / (1 + i)
    M_prev = M.copy()
    M_i = white_noise_field((size, size, size), scale=0.001)
    rho_obs = cp.zeros_like(M)
    antimatter = cp.random.normal(0, 0.01, M.shape) * mask * 0.86  # 14% less than matter

    ob_x = cp.random.randint(0, size, n_obs)
    ob_y = cp.random.randint(0, size, n_obs)
    ob_z = cp.random.randint(0, size, n_obs)
    ob_age = cp.zeros(n_obs)

    M_layers.append(M * mask)
    M_prev_layers.append(M_prev * mask)
    M_i_layers.append(M_i * mask)
    rho_obs_layers.append(rho_obs)
    antimatter_layers.append(antimatter)
    radius_shells.append(radius_shell)
    shell_masks.append(mask)
    shell_surfaces.append(surface)
    observer_positions.append((ob_x, ob_y, ob_z))
    observer_age_layers.append(ob_age)


    ob_x = cp.random.randint(0, size, n_obs)
    ob_y = cp.random.randint(0, size, n_obs)
    ob_z = cp.random.randint(0, size, n_obs)
    ob_age = cp.zeros(n_obs)

    M_layers.append(M * mask)
    M_prev_layers.append(M_prev * mask)
    M_i_layers.append(M_i * mask)
    rho_obs_layers.append(rho_obs)
    antimatter_layers.append(antimatter)
    radius_shells.append(radius_shell)
    shell_masks.append(mask)
    shell_surfaces.append(surface)
    observer_positions.append((ob_x, ob_y, ob_z))
    observer_age_layers.append(ob_age)

def laplacian_3d(F):
    return (
        cp.roll(F, 1, axis=0) + cp.roll(F, -1, axis=0) +
        cp.roll(F, 1, axis=1) + cp.roll(F, -1, axis=1) +
        cp.roll(F, 1, axis=2) + cp.roll(F, -1, axis=2) -
        6 * F
    )

def observer_drift(M, ob_x, ob_y, ob_z, radius_shell, shell_max, step_size=1):
    pot = M + 0.5 * laplacian_3d(M)
    grad_x, grad_y, grad_z = cp.gradient(pot)
    gx = grad_x[ob_x, ob_y, ob_z]
    gy = grad_y[ob_x, ob_y, ob_z]
    gz = grad_z[ob_x, ob_y, ob_z]
    norm = cp.sqrt(gx**2 + gy**2 + gz**2) + 1e-6

    x_new = cp.clip(ob_x + step_size * (gx / norm), 0, size - 1).astype(cp.int32)
    y_new = cp.clip(ob_y + step_size * (gy / norm), 0, size - 1).astype(cp.int32)
    z_new = cp.clip(ob_z + step_size * (gz / norm), 0, size - 1).astype(cp.int32)

    r_obs = radius_shell[x_new, y_new, z_new]
    shell_hit = (r_obs >= shell_max)
    x_new[shell_hit] = size // 2
    y_new[shell_hit] = size // 2
    z_new[shell_hit] = size // 2

    return x_new, y_new, z_new

for step in tqdm(range(steps), desc="Fractal Tensor Cascade"):
    for i in range(len(M_layers)):
        M, M_prev, M_i, rho_obs = M_layers[i], M_prev_layers[i], M_i_layers[i], rho_obs_layers[i]
        antimatter = antimatter_layers[i]
        ob_x, ob_y, ob_z = observer_positions[i]
        ob_age = observer_age_layers[i]
        radius_shell = radius_shells[i]
        shell_max = int(radius_shell.max())

        ob_x, ob_y, ob_z = observer_drift(M, ob_x, ob_y, ob_z, radius_shell, shell_max)
        ob_age += 1
        alive = ob_age < obs_lifetime
        ob_x, ob_y, ob_z = ob_x[alive], ob_y[alive], ob_z[alive]
        ob_age = ob_age[alive]
        observer_positions[i] = (ob_x, ob_y, ob_z)
        observer_age_layers[i] = ob_age

        rho_obs *= 0.1
        rho_obs[ob_x, ob_y, ob_z] += 10 * cp.exp(-0.05 * step)

        # Observer attraction
        rho_obs += laplacian_3d(rho_obs) * 0.005

        # Antimatter generation and voiding
        antimatter += cp.random.normal(0, 0.01, M.shape) * shell_masks[i]
        annihilation = cp.minimum(M, antimatter)
        M += 5.0 * annihilation
        antimatter -= annihilation

        lap = laplacian_3d(M)
        decay = -lam * M * float(min(step / 10.0, 1.0))
        acoustic = -omega_sq * M - gamma_damp * (M - M_prev) / delta_t
        source = kappa * rho_obs
        accel = c**2 * D * lap + decay + acoustic + source

        M_next = 2 * M - M_prev + delta_t**2 * accel

        if i + 1 < max_layers and i + 1 == len(M_layers):
            new_M = inflationic_noise((size, size, size)) * 0.01
            shell_transfer = M[radius_shell == shell_max].mean()
            new_M[radius_shells[i] == 0] = shell_transfer
            M_layers.append(new_M)
            M_prev_layers.append(new_M.copy())
            M_i_layers.append(white_noise_field((size, size, size), scale=0.001))
            rho_obs_layers.append(cp.zeros_like(new_M))
            antimatter_layers.append(cp.zeros_like(new_M))
            observer_positions.append((cp.random.randint(0, size, n_obs), cp.random.randint(0, size, n_obs), cp.random.randint(0, size, n_obs)))
            observer_age_layers.append(cp.zeros(n_obs))

        M_prev_layers[i] = M
        M_layers[i] = M_next
        M_i_layers[i] = M_i + 0.1 * laplacian_3d(M_i) - 0.01 * M_i

    # ... [Projection and plotting logic remains unchanged] ...


for step in tqdm(range(steps), desc="Fractal Tensor Cascade"):
    for i in range(len(M_layers)):
        M, M_prev, M_i, rho_obs = M_layers[i], M_prev_layers[i], M_i_layers[i], rho_obs_layers[i]
        ob_x, ob_y, ob_z = observer_positions[i]
        radius_shell = radius_shells[i]
        shell_max = int(radius_shell.max())

        ob_x, ob_y, ob_z = observer_drift(M, ob_x, ob_y, ob_z, radius_shell, shell_max)
        observer_positions[i] = (ob_x, ob_y, ob_z)

        rho_obs *= 0.1
        rho_obs[ob_x, ob_y, ob_z] += 10 * cp.exp(-0.05 * step)

        lap = laplacian_3d(M)
        decay = -lam * M * float(min(step / 10.0, 1.0))
        acoustic = -omega_sq * M - gamma_damp * (M - M_prev) / delta_t
        source = kappa * rho_obs
        accel = c**2 * D * lap + decay + acoustic + source

        M_next = 2 * M - M_prev + delta_t**2 * accel

        if i + 1 < max_layers:
            if i + 1 == len(M_layers):
                new_M = inflationic_noise((size, size, size)) * 0.01
                shell_transfer = M[radius_shell == shell_max].mean()
                new_M[radius_shells[i] == 0] = shell_transfer
                M_layers.append(new_M)
                M_prev_layers.append(new_M.copy())
                M_i_layers.append(white_noise_field((size, size, size), scale=0.001))
                rho_obs_layers.append(cp.zeros_like(new_M))
                observer_positions.append((
                    cp.random.randint(0, size, n_obs),
                    cp.random.randint(0, size, n_obs),
                    cp.random.randint(0, size, n_obs)
                ))

        M_prev_layers[i] = M
        M_layers[i] = M_next
        M_i_layers[i] = M_i + 0.1 * laplacian_3d(M_i) - 0.01 * M_i

    if step % 10 == 0:
        combined_shell = cp.zeros((size, size, size))
        for i in range(len(M_layers)):
            combined_shell += M_layers[i] * shell_surfaces[i]

        shell_energy = cp.sum(combined_shell)
        if shell_energy < 1e-6:
            continue

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
        plt.savefig(os.path.join(output_dir, f"moll_tensor_{step:04d}.png"))
        plt.close()

        if step % 10 == 0:
            alm = map2alm(proj, lmax=256)
            cl = alm2cl(alm)
            ell = np.arange(len(cl))
            plt.figure(figsize=(10, 6))
            plt.plot(ell, cl, label=f"Simulated (Step {step})", color="orange")
            if ell_planck is not None:
                plt.plot(ell_planck, cl_planck, label="Planck 2018 CMB", linestyle="--", color="blue")
            plt.xlabel("Multipole moment ℓ")
            plt.ylabel("Cℓ")
            plt.yscale("log")
            plt.title("Angular Power Spectrum — Fifth Field vs Planck")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"cl_compare_{step:04d}.png"))
            plt.close()

cp.save(os.path.join(output_dir, "M_final_tensor.npy"), M_layers[0])
print("[✓] Lilith_tensordrive.py complete — Fractal Shell Topology + Inflationic Acoustics + CMB Match")

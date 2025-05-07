import cupy as cp
import healpy as hp
import numpy as np


def evolve_collapse_field(M, M_i, rho_obs, D, lam, kappa, phi_boost, gamma_i, theta_field):
    """
    Core collapse field evolution update for CMB-like projections.
    Returns updated M and M_i.
    """
    def laplace(M):
        return (-6 * M + sum(cp.roll(M, shift, axis=ax) for shift in (1, -1) for ax in range(3)))

    def evolve_imaginary(M_i, theta):
        phase = cp.exp(1j * theta)
        return -gamma_i * M_i + cp.real(phase * M_i)

    lap_M = laplace(M)
    decay = -lam * M
    source = kappa * rho_obs
    M_i_update = evolve_imaginary(M_i, theta_field)

    M_i += M_i_update
    M += D * lap_M + source + decay + phi_boost * cp.tanh(cp.clip(M_i_update, -10, 10))

    M = cp.clip(M, -1e6, 1e6)
    M_i = cp.clip(M_i, -1e6, 1e6)

    return M, M_i


def project_to_healpix(M, size=128, nside=512, threshold=0.01):
    """
    Projects a 3D collapse field M to a HEALPix shell map.
    """
    collapse_mask = M > threshold
    collapse_coords = cp.argwhere(collapse_mask)
    if collapse_coords.shape[0] == 0:
        return np.zeros(hp.nside2npix(nside))

    coords_cpu = cp.asnumpy(collapse_coords)
    cx = cy = cz = size // 2
    dx = coords_cpu[:, 0] - cx
    dy = coords_cpu[:, 1] - cy
    dz = coords_cpu[:, 2] - cz
    r_vals = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    theta = np.arccos(dz / r_vals)
    phi = np.arctan2(dy, dx) % (2 * np.pi)
    pix = hp.ang2pix(nside, theta, phi)
    shell_map = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(np.float64)
    shell_map = shell_map / np.max(shell_map)  # Normalize to unit sphere projection

    return shell_map


def save_mollweide_projection(shell_map, filename="collapse_mollweide.png"):
    import matplotlib.pyplot as plt
    hp.mollview(shell_map, title="Collapse Field Mollweide Projection", cmap="inferno", norm="hist", min=0.0, max=1.0)
    plt.savefig(filename, dpi=400)
    plt.close()

    # Debug final cross-section slice
    plt.figure()
    plt.imshow(cp.asnumpy(M[:, :, size//2]), cmap='inferno')
    plt.colorbar()
    plt.title("Final M field (midplane)")
    plt.savefig("M_field_final_slice.png", dpi=300)
    plt.close()

# === TEST HARNESS ===
if __name__ == "__main__":
    # --- Simulation Grid & Projection Resolution ---
    size = 128  # 3D grid size (cube of size^3)
    nside = 512  # Healpix projection resolution
    steps = 100

    # --- Parameters ---
    params = {
        "D": 0.56,
        "lam": 0.0577,
        "kappa": 1.0,
        "phi_boost": 1.4,
        "gamma_i": 0.5,
        "theta_field": 0.4,
    }

    M = (cp.random.rand(size, size, size).astype(cp.float32) - 0.5) * 1e-2

    M_i = cp.random.rand(size, size, size).astype(cp.float32) * 0.01
    rho_obs = cp.zeros_like(M)

    def add_observer_density(rho, center, strength=1.0, sigma=3.0):
        x0, y0, z0 = center
        grid = cp.arange(-int(3 * sigma), int(3 * sigma) + 1)
        gx, gy, gz = cp.meshgrid(grid, grid, grid, indexing='ij')
        kernel = cp.exp(-(gx**2 + gy**2 + gz**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        kx, ky, kz = kernel.shape
        sx = slice(x0 - kx//2, x0 + kx//2 + 1)
        sy = slice(y0 - ky//2, y0 + ky//2 + 1)
        sz = slice(z0 - kz//2, z0 + kz//2 + 1)
        rho[sx, sy, sz] += strength * kernel

    shell_maps = []
    for t in range(steps):
        if t % 10 == 0:
            m_mean = float(cp.mean(M))
            m_max = float(cp.max(M))
            print(f"[{t:03d}] M: mean={m_mean:.3e}, max={m_max:.3e}")
        # inject observer density every 10 steps
        if t % 10 == 0:
            add_observer_density(rho_obs, (size//2, size//2, size//2), strength=5.0, sigma=6.0)

        # advance imaginary one step ahead
        _, M_i = evolve_collapse_field(M, M_i, rho_obs, **params)
        M, _ = evolve_collapse_field(M, M_i, rho_obs, **params)
        shell = project_to_healpix(M, size=size, nside=nside)
        shell_maps.append(shell)

    avg_shell = np.mean(shell_maps[-10:], axis=0)

    # Additional projection and comparison for imaginary field
    shell_imag = project_to_healpix(M_i, size=size, nside=nside)
    np.save("final_imaginary_shell_map.npy", shell_imag)
    save_mollweide_projection(shell_imag, filename="imaginary_mollweide.png")

    # Differential between real and imaginary shell maps
    diff_map = np.abs(avg_shell - shell_imag)
    np.save("collapse_shell_difference.npy", diff_map)
    save_mollweide_projection(diff_map, filename="collapse_diff_mollweide.png")
    np.save("avg_collapse_shell_map.npy", avg_shell)
    save_mollweide_projection(avg_shell, filename="collapse_mollweide.png")  # Rendered from average shell

    cl = hp.anafast(avg_shell)
    np.save("collapse_power_spectrum.npy", cl)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(cl, label="Collapse Spectrum")
    plt.plot(cl, alpha=0.5, label="Reference (50% opacity)")
    if np.any(cl > 0):
        plt.yscale('log')
    else:
        print("⚠️ Warning: No positive values in power spectrum. Skipping log scaling.")
    plt.title("Collapse Field Angular Power Spectrum")
    plt.xlabel("Multipole moment ℓ")
    plt.ylabel(r"$C_\ell$")
    plt.grid(True)
    plt.legend()
    plt.savefig("collapse_power_spectrum.png", dpi=400)
    plt.close()

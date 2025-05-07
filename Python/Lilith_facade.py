# CMB Comparison Collapse — Enhanced HEALPix Projector with Stepwise Output
import cupy as cp
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import numpy as np
import os

# Parameters
size = 256
D = .00002
lam = 50
kappa = .0000002
threshold = 5
steps = 700
nside = 128
npix = hp.nside2npix(nside)

output_dir = "cmb_projection_output"
os.makedirs(output_dir, exist_ok=True)

M = cp.random.rand(size, size, size) * 0.01
rho_obs = cp.zeros((size, size, size))

r = 10
n_obs = 20
angles = cp.linspace(0, 2 * cp.pi, n_obs, endpoint=False)
observer_thetas = cp.linspace(0, cp.pi, n_obs, endpoint=False)
observer_phis = angles.copy()

fig, ax = plt.subplots()
cmap = plt.get_cmap('inferno')
im = ax.imshow(M[:, :, size // 2].get(), cmap=cmap, vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_title("M(x, t) Cross-section at z=128")

# Laplace operator
def laplace_operator(M):
    return (
        cp.roll(M, 1, axis=0) + cp.roll(M, -1, axis=0) +
        cp.roll(M, 1, axis=1) + cp.roll(M, -1, axis=1) +
        cp.roll(M, 1, axis=2) + cp.roll(M, -1, axis=2) - 6 * M
    )

def spherical_to_cartesian(r, theta, phi):
    x = int(r * cp.sin(theta) * cp.cos(phi)) + size // 2
    y = int(r * cp.sin(theta) * cp.sin(phi)) + size // 2
    z = int(r * cp.cos(theta)) + size // 2
    return x % size, y % size, z % size

shell_map = cp.zeros(npix)

# Projection function
def project_collapse_shell(M, threshold, step):
    collapse_mask = M > threshold
    x, y, z = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
    dx = x - size // 2
    dy = y - size // 2
    dz = z - size // 2
    r = cp.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    theta = cp.arccos(dz / r).flatten()
    phi = cp.arctan2(dy, dx).flatten() % (2 * cp.pi)
    weights = M.flatten()
    mask = collapse_mask.flatten()

    theta = cp.asnumpy(theta[mask])
    phi = cp.asnumpy(phi[mask])
    weights = cp.asnumpy(weights[mask])

    pix = hp.ang2pix(nside, theta, phi)
    sphere_map = np.bincount(pix, weights=weights, minlength=npix)

    hp.write_map(os.path.join(output_dir, f"healpix_step_{step:04d}.fits"), sphere_map, overwrite=True)
    hp.mollview(np.log1p(sphere_map), title=f"Projection Step {step}", cmap="inferno", cbar=False)
    plt.savefig(os.path.join(output_dir, f"mollview_step_{step:04d}.png"))
    plt.close()

    cl = hp.anafast(sphere_map)
    ell = np.arange(len(cl))
    plt.plot(ell, cl)
    plt.yscale('log')
    plt.title(f"Power Spectrum Step {step}")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"spectrum_step_{step:04d}.png"))
    plt.close()

# Simulation loop
for step in tqdm(range(steps), desc="Simulating Collapse"):
    rho_obs[:] = 0
    for i in range(n_obs):
        phi = (observer_phis[i] + 0.05 * step) % (2 * cp.pi)
        theta = observer_thetas[i]
        x, y, z = spherical_to_cartesian(r, theta, phi)
        rho_obs[x % size, y % size, z % size] = 10

    diffusion = D * laplace_operator(M)
    decay = -lam * M
    source = kappa * rho_obs
    M += diffusion + decay + source

    if step % 2 == 0:
        project_collapse_shell(M, threshold, step)

    im.set_data(M[:, :, size // 2].get())

ani = FuncAnimation(fig, lambda f: [im], frames=steps, interval=100, blit=True)
ani.save(os.path.join(output_dir, "collapse_animation_gpu.mp4"), writer="ffmpeg")
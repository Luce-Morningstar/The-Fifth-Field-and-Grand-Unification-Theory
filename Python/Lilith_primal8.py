# Lilith_Primal_Turbo_Ultra — GPU Optimized Collapse Engine with Vectorized Observer Dynamics + GUI + Sphere Projection

import cupy as cp
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
import json
from tqdm import tqdm
from cupy.cuda import Stream

cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

# Default parameters
params = {
    'size': 128,
    'n_obs': 50000,
    'steps': 300,
    'dt': 0.1,
    'alpha': 0.01,
    'D': 0.2,
    'sigma': 2.0,
    'beta': 0.05,
    'gamma': 0.03,
    'kappa': 0.5,
    'r': 30.0,
    'epsilon': 1e-6,
    'eta': 0.02,
    'zeta': 0.05,
    'delta': 1e-5,
    'threshold': 0.3,
    'n': 2,
    'mu': 0.5,
    'lam': 10.0,
    'sieve_threshold': 0.65,
    'spawn_intensity': 0.015,
    'save_frames': False,
    'nside': 64,
    'gaussian_shell_sigma': 3.0,
    'feedback_strength': 0.05,
    'shockwave_decay': 0.01,
    'shockwave_strength': 5.0
}

CONFIG_PATH = "lilith_gui_config.json"

# GUI to set parameters

def launch_gui():
    root = tk.Tk()
    root.title("Lilith Turbo Config")
    entries = {}

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                params.update(json.load(f))
        except:
            pass

    for i, (key, value) in enumerate(params.items()):
        tk.Label(root, text=key).grid(row=i, column=0, sticky='w')
        entry = tk.Entry(root)
        entry.insert(0, str(value))
        entry.grid(row=i, column=1)
        entries[key] = entry

    def apply():
        for key in params:
            val = entries[key].get()
            try:
                if isinstance(params[key], bool):
                    params[key] = val.lower() in ["true", "1", "yes"]
                else:
                    params[key] = type(params[key])(eval(val))
            except Exception as e:
                print(f"Failed to parse {key}: {e}")
        with open(CONFIG_PATH, 'w') as f:
            json.dump(params, f)
        root.quit()

    tk.Button(root, text="Start", command=apply).grid(row=len(params), column=0, columnspan=2)
    root.mainloop()
    root.destroy()

launch_gui()

size = params['size']
output_dir = "lilith_turbo_output"
os.makedirs(output_dir, exist_ok=True)

# Init fields
M_real = cp.random.rand(size, size, size) * 0.01
M_img = cp.random.rand(size, size, size) * 0.01
M_prev_real = M_real.copy()
M_prev_img = M_img.copy()
observer_trail = cp.zeros((size, size, size))
rho_obs = cp.zeros((size, size, size))
rho_anti = cp.zeros((size, size, size))
shockwave_field = cp.zeros((size, size, size))
observer_positions = cp.random.rand(params['n_obs'], 3)
observer_momentum = cp.random.normal(0, 1, (params['n_obs'], 3))
observer_momentum /= cp.linalg.norm(observer_momentum, axis=1, keepdims=True)
observer_momentum *= 0.3
observer_positions += cp.random.normal(0, 0.05, observer_positions.shape)
observer_positions = cp.clip(observer_positions, 0, 1)

stream_main = Stream(non_blocking=True)

laplacian = lambda M: sum(cp.roll(M, s, axis=a) for s in (-1, 1) for a in range(3)) - 6 * M
sigmoid = lambda M: 1 / (1 + cp.exp(-params['lam'] * (M - params['mu'])))

def sieve_birth(M):
    grad = cp.gradient(M)
    foam = cp.sqrt(sum(g**2 for g in grad))
    spawn = (foam > params['sieve_threshold']) & (M < 0.3)
    return cp.where(spawn, cp.random.rand(*M.shape) * params['spawn_intensity'], 0.0)

def update_antimatter(rho_anti, coords):
    rho_anti[:] = 0
    for i in range(coords.shape[0]):
        x, y, z = coords[i]
        if np.random.rand() < 0.43:
            rho_anti[x, y, z] = 1
            shockwave_field[x, y, z] += params['shockwave_strength']


def evolve_field(M, M_prev, rho_obs, rho_anti, trail, feedback=None):
    S = sigmoid(M)
    decay = cp.exp(-params['sigma'] * S) * M
    annihilation = params['beta'] * M * rho_anti**params['n']
    observer_force = params['kappa'] * rho_obs / (params['r']**2 + params['epsilon'])
    entropy = params['eta'] * (M * laplacian(cp.log(M + params['delta'])) + sum((g**2)/(M + params['delta'])**2 for g in cp.gradient(M)))
    gate = params['zeta'] * S
    birth = sieve_birth(M)
    feedback_term = params['feedback_strength'] * (feedback - M) if feedback is not None else 0
    shockwave_term = shockwave_field * params['shockwave_strength']
    rhs = params['D'] * laplacian(M) - decay - annihilation + observer_force + entropy + gate + birth + feedback_term + shockwave_term
    rhs = cp.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)  # stabilize blowups
    accel = rhs - params['alpha'] * (M - M_prev) / params['dt']
    return 2 * M - M_prev + params['dt']**2 * accel

def update_observers():
    global observer_positions, observer_momentum
    anti_coords = cp.argwhere(rho_anti > 0)
    if anti_coords.shape[0] > 2000:
        anti_coords = anti_coords[:2000]  # throttle
    for i in range(observer_positions.shape[0]):
        pos = (observer_positions[i] * size).astype(int)
        for anti in anti_coords:
            d = cp.linalg.norm(pos - anti)
            if d < 0.1 * size:
                direction = (pos - anti).astype(cp.float32)
                if cp.linalg.norm(direction) > 0:
                    direction /= cp.linalg.norm(direction)
                    observer_momentum[i] += direction * 0.1
    observer_positions += observer_momentum
    observer_positions = cp.clip(observer_positions, 0, 1)
    coords = (observer_positions * size).astype(int)
    coords = cp.clip(coords, 0, size - 1)
    return coords

def place_fields(coords):
    global rho_obs, observer_trail
    rho_obs[:] = 0
    for i in range(coords.shape[0]):
        x, y, z = coords[i]
        rho_obs[x, y, z] = 10
        observer_trail[x, y, z] += 0.5
    observer_trail *= 0.99

def save_frame(M, step, label):
    if not params['save_frames']: return
    slice_ = M[:, :, size // 2]
    plt.imshow(cp.asnumpy(cp.log1p(slice_)), cmap='inferno')
    plt.title(f"{label} Field Step {step}")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f"{label.lower()}_step_{step:04d}.png"))
    plt.close()

def project_sphere(M, step, label):
    try:
        x, y, z = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
        dx = x - size // 2
        dy = y - size // 2
        dz = z - size // 2
        r = cp.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        theta = cp.arccos(dz / r)
        phi = cp.arctan2(dy, dx) % (2 * cp.pi)
        pix = hp.ang2pix(params['nside'], cp.asnumpy(theta).flatten(), cp.asnumpy(phi).flatten())
        values = cp.asnumpy(M.flatten())
        map_ = np.bincount(pix, weights=values, minlength=hp.nside2npix(params['nside']))
        hp.mollview(np.log1p(map_), cmap='inferno', title=f"{label} Sphere {step}", cbar=False)
        plt.savefig(os.path.join(output_dir, f"{label.lower()}_sphere_{step:04d}.png"))
        plt.close()
    except Exception as e:
        print(f"Projection failed at step {step}: {e}")

# Main loop
for step in tqdm(range(params['steps']), desc="Simulating Collapse"):
    with stream_main:
        shockwave_field *= (1 - params['shockwave_decay'])
        if step < 5:
            shockwave_field *= 0.1
        coords = update_observers()
        place_fields(coords)
        update_antimatter(rho_anti, coords)

        if step % 2 == 0:
            M_next_real = evolve_field(M_real, M_prev_real, rho_obs, rho_anti, observer_trail, feedback=M_img)
            M_prev_real, M_real = M_real, M_next_real
            if step % 10 == 0:
                save_frame(M_real, step, "Real")
                project_sphere(M_real, step, "Real")
        else:
            M_next_img = evolve_field(M_img, M_prev_img, rho_obs, rho_anti, observer_trail, feedback=M_real)
            M_prev_img, M_img = M_img, M_next_img
            if step % 10 == 0:
                save_frame(M_img, step, "Imag")
                project_sphere(M_img, step, "Imag")

print("Simulation complete. Lilith devoured another timeline.")

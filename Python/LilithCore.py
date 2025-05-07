## Lilith - Core Collapse Logic (v1.5 Final Purge Fix)
# Major Fixes: Duplicates, Laplacian, GUI-State, and evolve_imaginary merged
# Optimized for CuPy, real-imaginary coherence, memory, and chunk handling

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor
import json
import datetime
import scipy.ndimage
import random
import psutil
import traceback
import psutil

# Global GUI GPU label
gpu_label = None

def update_gpu_info():
    global gpu_label
    if gpu_label is not None:
        try:
            usage = psutil.cpu_percent(interval=0.5)
            gpu_label.config(text=f"System Load: {usage:.1f}%")
        except Exception as e:
            gpu_label.config(text="Load unavailable")


# Force GPU context and sync
cp.cuda.Device(0).use()
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    cp.zeros((1,))
    stream.synchronize()
print("[CUPY DEVICE] Using:", cp.cuda.runtime.getDevice())
print("[CUPY NAME] GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'])

# GPU memory logging for diagnostics
try:
    mempool = cp.get_default_memory_pool()
    print("[GPU MEMORY] Used:", mempool.used_bytes() / 1024**2, "MB")
    print("[GPU MEMORY] Total Alloc:", mempool.total_bytes() / 1024**2, "MB")
except Exception as e:
    print("[GPU STATS ERROR]", e)

# CONFIG LOAD
params = {
    'radius': 10,
    'mass_intensity': 5.0,
    'obs_intensity': 3.0,
    'anti_intensity': 3.0,
    'steps': 50,
    'grid_size': 32,
    'diffusion_rate': 0.01,
    'diffusion_base_speed': 1.0,
    'alpha': 0.005,
    'kappa': 0.5,
    'H_t': 0.001,
    'seed_interval': 5,
    'seed_repeats': 3
}
if os.path.exists("lilith_config.json"):
    try:
        with open("lilith_config.json", "r") as f:
            params.update(json.load(f))
            print("[CONFIG] Loaded lilith_config.json")
    except Exception as e:
        print(f"[CONFIG ERROR] Could not load config: {e}")

N = params['grid_size']

D_base = params['diffusion_rate'] * params['diffusion_base_speed']
alpha = params['alpha']
gamma = 0.1
kappa = params['kappa']
H_t = params['H_t']
g_const = 0.002
expansion_rate = math.e
seed_interval = params['seed_interval']
seed_repeats = params['seed_repeats']

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"collapse_output_{run_id}"
os.makedirs(output_dir, exist_ok=True)


# Add global chunk_manager declaration
chunk_manager = None

from your_chunk_module import CollapseChunkManager  # Replace with actual module if different

chunk_manager = CollapseChunkManager(chunk_size=32)
chunk_manager.initialize_origin()
chunk_manager.spawn_adjacent_chunks((0, 0, 0))
chunk_manager.spawn_adjacent_chunks((1, 0, 0))
chunk_manager.spawn_adjacent_chunks((-1, 0, 0))
chunk_manager.spawn_adjacent_chunks((0, 1, 0))
chunk_manager.spawn_adjacent_chunks((0, -1, 0))

def recursive_step(steps, progress_callback=None):
    global chunk_manager

    if chunk_manager is None:
        print("[ERROR] chunk_manager is not initialized.")
        return None, None

    active_chunks = chunk_manager.get_active_chunks()
    if not active_chunks:
        print("[ERROR] No active chunks to process. Aborting recursive step.")
        return None, None

    coords, batched_R, batched_I, batched_M, batched_rho, batched_antiM, batched_entropy, batched_void, batched_active, batched_lifetime = [], [], [], [], [], [], [], [], [], []

    for coord, chunk in active_chunks.items():
        try:
            coords.append(coord)
            batched_R.append(chunk['R'])
            batched_I.append(chunk['I'])
            batched_M.append(chunk['M'])
            batched_rho.append(chunk['rho'])
            batched_antiM.append(chunk['antiM'])
            batched_entropy.append(chunk['entropy'])
            batched_void.append(chunk['void'])
            batched_active.append(chunk['active'])
            batched_lifetime.append(chunk['lifetime'])
        except Exception as e:
            print(f"[CHUNK LOAD ERROR] Failed at chunk {coord}: {e}")
            continue

    try:
        batched_R = cp.stack(batched_R)
        batched_I = cp.stack(batched_I)
        batched_M = cp.stack(batched_M)
        batched_rho = cp.stack(batched_rho)
        batched_antiM = cp.stack(batched_antiM)
        batched_entropy = cp.stack(batched_entropy)
        batched_void = cp.stack(batched_void)
        batched_active = cp.stack(batched_active)
        batched_lifetime = cp.stack(batched_lifetime)
    except Exception as e:
        print(f"[BATCH ERROR] Could not stack tensors: {e}")
        return None, None

    def broadcast_death_mask(field):
        return cp.broadcast_to(field, batched_R.shape)

    for t in range(steps):
        try:
            print(f"[STEP {t}] Running evolution...")
            batched_lifetime -= 1.0
            death_mask = broadcast_death_mask(batched_lifetime <= 0.0)

            r = cp.sqrt(cp.sum(cp.square(cp.indices(batched_R.shape[1:]) - (N // 2)), axis=0)) + 1e-5
            r = r[None, ...]  # Broadcast to all batches
            collapse_potential = batched_I * batched_rho * (kappa / (r ** 2))
            annihilation = sanni_annihilation(batched_R, batched_antiM, gamma)
            update = 0.05 * collapse_potential - annihilation

            batched_R = cp.where(batched_active, batched_R + update, batched_R)
            batched_R = cp.where(death_mask, 0.0, batched_R)
            batched_active = cp.where(death_mask, False, batched_active)
            batched_active |= cp.abs(update) > 1e-6

            batched_I = evolve_imaginary(
                batched_I, batched_M, batched_rho, t, batched_active, batched_void, batched_entropy, batched_R
            )

            stream.synchronize()
            mem_used = cp.get_default_memory_pool().used_bytes() / (1024**2)
            print(f"[GPU USAGE CHECK] CuPy Memory Allocated: {mem_used:.2f} MB")
            print(f"[GPU STATUS] R max: {cp.max(batched_R).item():.4f}, mean: {cp.mean(batched_R).item():.4f}")
            print(f"[GPU STATUS] I max: {cp.max(batched_I).item():.4f}, mean: {cp.mean(batched_I).item():.4f}")
            print(f"[CHECKPOINT] Step {t} completed.")

            if progress_callback:
                progress_callback(t + 1, steps)

            try:
                save_slice(batched_R[0], t)
                save_volume(batched_R[0], t)
            except Exception as viz_err:
                print(f"[VISUAL ERROR] Step {t}: {viz_err}")

        except Exception as e:
            print(f"[SIM ERROR] Step {t}: {e}")
            return None, None

    def update_chunk_fields(chunk, idx):
        try:
            chunk['history'].append(batched_R[idx].copy())
            if len(chunk['history']) > 5:
                chunk['history'].pop(0)
            chunk['R'] = batched_R[idx]
            chunk['I'] = batched_I[idx]
            chunk['active'] = batched_active[idx]
            chunk['lifetime'] = batched_lifetime[idx]
        except Exception as e:
            print(f"[CHUNK UPDATE ERROR] Chunk {idx}: {e}")

    for idx, coord in enumerate(coords):
        if coord in active_chunks:
            update_chunk_fields(active_chunks[coord], idx)
            if cp.any(batched_active[idx][-1]) or cp.any(batched_active[idx][0]) or \
               cp.any(batched_active[idx][:, -1]) or cp.any(batched_active[idx][:, 0]) or \
               cp.any(batched_active[idx][:, :, -1]) or cp.any(batched_active[idx][:, :, 0]):
                chunk_manager.spawn_adjacent_chunks(coord)

    print("[RECURSION END] All chunks updated and outputs returned.")
    return batched_R[0], batched_I[0]

def spherical_observer_seed(shape, radius=10, n_obs=100, intensity=5.0):
    seed_field = cp.zeros(shape, dtype=cp.float32)
    z = cp.random.uniform(-1, 1, n_obs)
    t = cp.random.uniform(0, 2 * cp.pi, n_obs)
    r_sphere = cp.sqrt(1 - z**2)
    x = r_sphere * cp.cos(t)
    y = r_sphere * cp.sin(t)
    theta = cp.arccos(z)
    phi = cp.arctan2(y, x) % (2 * cp.pi)

    r = radius
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    ix = (r * cp.sin(theta) * cp.cos(phi)).astype(int) + cx
    iy = (r * cp.sin(theta) * cp.sin(phi)).astype(int) + cy
    iz = (r * cp.cos(theta)).astype(int) + cz

    seed_field[ix % shape[0], iy % shape[1], iz % shape[2]] = intensity
    return seed_field


def launch_gui():
    # Load saved GUI state if it exists
    gui_state_file = 'lilith_gui_state.json'
    saved_gui = {}
    if os.path.exists(gui_state_file):
        try:
            with open(gui_state_file, 'r') as f:
                saved_gui = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load GUI state: {e}")

    global root
    root = tk.Tk()
    root.title("Lilith Collapse Engine Control Panel")

    progress_var = tk.StringVar(value="Idle")

    def update_progress(step, total):
        progress_var.set(f"Step {step}/{total}")
        root.update_idletasks()

    from_this_chunk = chunk_manager.chunks[(0, 0, 0)]  # Ensure variable is defined before use

    # Seed observer field with quantum foam
    from_this_chunk['rho'] += cp.random.normal(loc=0.0, scale=0.02, size=from_this_chunk['rho'].shape).astype(cp.float32)
    safe_index = min(from_this_chunk['antiM'].shape[0] - 1, N // 2 + 1)
    clamp = lambda v, maxval: max(0, min(v, maxval - 1))
    x = clamp(N // 2 + 1, from_this_chunk['antiM'].shape[0])
    y = clamp(N // 2, from_this_chunk['antiM'].shape[1])
    z = clamp(N // 2, from_this_chunk['antiM'].shape[2])
    inject_antiphase(from_this_chunk['antiM'], (x, y, z), params['anti_intensity'])
    from_this_chunk['active'][(N//2)-1:(N//2)+2, (N//2)-1:(N//2)+2, (N//2)-1:(N//2)+2] = True

def sanni_annihilation(M_field, antiM_field, gamma):
    return gamma * M_field * antiM_field


def void_draw(rho_obs, void_field):
    return void_field * (1.0 - cp.clip(rho_obs, 0, 1))


def evolve_real(R_field, I_field_prev, rho_obs, antiM_field, active_mask, lifetime_field):
    # Collapse foam into matter
    collapse_threshold = 0.8
    condensation_rate = float(params.get('condensation_rate', 0.02))
    foam_to_matter = (rho_obs > collapse_threshold) & (R_field < 0.1)
    R_field = cp.where(foam_to_matter, R_field + condensation_rate * rho_obs, R_field)
    with stream:
        lifetime_field -= 1.0
        lifetime_field = cp.clip(lifetime_field, 0.0, 100.0)
        death_mask = lifetime_field <= 0.0
        r = cp.sqrt(cp.sum(cp.square(cp.indices(R_field.shape) - (N // 2)), axis=0)) + 1e-5
        collapse_potential = I_field_prev * rho_obs * (kappa / (r ** 2))
        annihilation = sanni_annihilation(R_field, antiM_field, gamma)
        update = 0.05 * collapse_potential - annihilation
        R_field = cp.where(active_mask, R_field + update, R_field)
        R_field = cp.where(death_mask, 0.0, R_field)
        active_mask = cp.where(death_mask, False, active_mask)
        active_mask |= cp.abs(update) > 1e-6
    return cp.clip(R_field, 0, 1250000)


def save_slice(field, step, axis=0):
    slice_2d = cp.asnumpy(cp.take(field, N//2, axis=axis))
    vmin, vmax = np.nanmin(slice_2d), np.nanmax(slice_2d)

    if np.isnan(slice_2d).any() or vmax - vmin < 1e-10:
        print(f"[WARN] Skipped frame {step}: degenerate data")
        return

    plt.imshow(slice_2d, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"Collapse Field Slice - Step {step}")
    plt.savefig(os.path.join(output_dir, f"collapse_step_{step:03d}.png"))
    plt.close()


def save_volume(field, step):
    np.save(os.path.join(output_dir, f"observer_Seed{step:03d}.npy"), cp.asnumpy(field))
    # Drop a 3D plane graph for visualization with intensity thresholding
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    data = cp.asnumpy(field)
    threshold = np.max(data) * 0.05
    x, y, z = np.indices(data.shape)
    mask = data > threshold

    if np.count_nonzero(mask) == 0:
        print(f"[WARN] Skipped 3D plot at step {step}: degenerate volume")
        plt.close()
        return

    xf, yf, zf = x[mask], y[mask], z[mask]
    norm_vals = (data[mask] - np.min(data[mask])) / (np.max(data[mask]) - np.min(data[mask]) + 1e-9)
    ax.scatter(xf, yf, zf, c=norm_vals, cmap='inferno', alpha=0.4, s=2)
    ax.set_title(f"3D Collapse Field - Step {step}")
    plt.savefig(os.path.join(output_dir, f"collapse_volume_3d_step_{step:03d}.png"))
    plt.close()


def collapse_score(R_field):
    grad = cp.gradient(R_field)
    lap = laplacian(R_field)
    return float(cp.mean(cp.abs(grad[0])) + cp.mean(cp.abs(lap)))


def _evolve_chunk(coord, chunk, steps, progress_callback):
    # Secondary observer spawn logic (Quantum Hydra Mode)
    def check_spawn_observer(chunk):
        spawn_mask = chunk['rho'] > 1.0
        if cp.any(spawn_mask):
            coords = cp.argwhere(spawn_mask)
            for index, (x, y, z) in enumerate(coords):
                inject_sphere(chunk['rho'], (int(x), int(y), int(z)), 2, 0.5, spiral_index=index, time_index=chunk.get('frame', 0))
                inject_sphere(chunk['M'], (int(x), int(y), int(z)), 1, 0.2, spiral_index=index, time_index=chunk.get('frame', 0))
                inject_sphere(chunk['rho'], (int(x), int(y), int(z)), 2, 0.5)
                inject_sphere(chunk['M'], (int(x), int(y), int(z)), 1, 0.2)
                chunk['active'][int(x), int(y), int(z)] = True
    substeps = 4
    R = chunk['R']
    I = chunk['I']
    M = chunk['M']
    rho = chunk['rho']
    antiM = chunk['antiM']
    entropy = chunk['entropy']
    void = chunk['void']
    active = chunk['active']
    lifetime = chunk['lifetime']

    for t in range(steps):
        for _ in range(substeps):
            R = evolve_real(R, I, rho, antiM, active, lifetime)
            I = evolve_imaginary(I, M, rho, t, active, void, entropy, R)
            active |= cp.abs(R) > 1e-6

        check_spawn_observer(chunk)
        chunk['frame'] = chunk.get('frame', 0) + 1
        chunk['R'] = R
        chunk['I'] = I
        chunk['active'] = active

        if cp.any(active[-1]) or cp.any(active[0]) or cp.any(active[:, -1]) or cp.any(active[:, 0]) or cp.any(active[:, :, -1]) or cp.any(active[:, :, 0]):
            chunk_manager.spawn_adjacent_chunks(coord)

        if progress_callback:
            progress_callback(t + 1, steps)
          

## Lilith - Core Collapse Logic (v1.5 Final Purge Fix)
# ... [rest of unchanged code above remains] ...


    def run_sim():
        try:
            config = {k: float(v.get()) if '.' in v.get() else int(v.get()) for k, v in entries.items()}
            print("[CONFIG] Running with:", config)
            progress_var.set("Running...")
            root.update_idletasks()

            # Unpack config
            params.update(config)
            N = int(config['grid_size'])
            chunk_manager = CollapseChunkManager(chunk_size=N)
            chunk_manager.initialize_origin()
            chunk_manager.spawn_adjacent_chunks((0, 0, 0))

            origin_chunk = chunk_manager.chunks[(0, 0, 0)]
            center = (N // 2, N // 2, N // 2)
            inject_sphere(origin_chunk['M'], center, config['radius'], config['mass_intensity'])
            inject_sphere(origin_chunk['rho'], center, config['radius'], config['obs_intensity'])
            inject_antiphase(origin_chunk['antiM'], (N // 2 + 1, N // 2, N // 2), config['anti_intensity'])
            origin_chunk['active'][(N // 2) - 1:(N // 2) + 2, (N // 2) - 1:(N // 2) + 2, (N // 2) - 1:(N // 2) + 2] = True

            mem_used = cp.get_default_memory_pool().used_bytes() / (1024**2)
            print(f"[GPU USAGE CHECK] CuPy Memory Allocated: {mem_used:.2f} MB")

            R_out, I_out = recursive_step(config['steps'], update_progress)
            progress_var.set(f"Collapse Score: {collapse_score(R_out):.4f}")
        except Exception as e:
            print("[ERROR] Simulation failed:", e)
            progress_var.set("Error")
            root.update_idletasks()

import tkinter as tk
from tkinter import ttk

params = {
    'radius': 10,
    'mass_intensity': 5.0,
    'obs_intensity': 3.0,
    'anti_intensity': 3.0,
    'steps': 50,
    'grid_size': 32,
    'diffusion_rate': 0.01,
    'diffusion_base_speed': 1.0,
    'alpha': 0.005,
    'kappa': 0.5,
    'H_t': 0.001,
    'observer_count': 6,
    'condensation_rate': 0.02
}

progress_var = None
gpu_label = None


def launch_gui():
    global gpu_label, progress_var

    root = tk.Tk()
    root.title("Lilith Collapse Engine")
    progress_var = tk.StringVar(value="Idle")

    entries = {}

    def add_field(label, key, row):
        ttk.Label(root, text=label).grid(row=row, column=0)
        e = tk.Entry(root)
        e.insert(0, str(params.get(key, "")))
        e.grid(row=row, column=1)
        entries[key] = e

    field_defs = [
        ("Sphere Radius", "radius"),
        ("Mass Intensity", "mass_intensity"),
        ("Observer Intensity", "obs_intensity"),
        ("Antiphase Intensity", "anti_intensity"),
        ("Steps", "steps"),
        ("Grid Size", "grid_size"),
        ("Diffusion Coeff. D(t)", "diffusion_rate"),
        ("Base Speed Coeff.", "diffusion_base_speed"),
        ("Alpha (decay)", "alpha"),
        ("Kappa (obs force)", "kappa"),
        ("H_t (entropy)", "H_t"),
        ("Number of Observers", "observer_count"),
        ("Condensation Rate", "condensation_rate")
    ]
# After activating the chunk and before starting mainloop
    R_out, I_out = recursive_step(params['steps'],)

    for i, (label, key) in enumerate(field_defs):
        add_field(label, key, i)

    def run_sim():
        try:
            config = {k: float(v.get()) if '.' in v.get() else int(v.get()) for k, v in entries.items()}
            print("[CONFIG] Running with:", config)
            progress_var.set("Running...")
            # TODO: Add simulation logic
        except Exception as e:
            print("[ERROR] Simulation failed:", e)
            progress_var.set("Error")

    ttk.Button(root, text="Run Simulation", command=run_sim).grid(row=len(field_defs), column=0, columnspan=2)

    ttk.Label(root, textvariable=progress_var).grid(row=len(field_defs)+1, column=0, columnspan=2)
    gpu_label = ttk.Label(root, text="System Load: 0%")
    gpu_label.grid(row=len(field_defs)+2, column=0, columnspan=2)

    root.mainloop()

launch_gui()

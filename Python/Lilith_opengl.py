# Lilith_opengl_viewer.py — Real-Time 3D Particle Viewer (WSL + Vispy Compatible)

import numpy as np
import os
import time
from vispy import app, scene
from vispy.scene.visuals import Markers
import tkinter as tk
from tkinter import simpledialog, filedialog

# GUI GRAVITY + FOLDER SETUP
root = tk.Tk()
root.withdraw()
default_gravity = 1e5
user_gravity = simpledialog.askfloat("Lilith Viewer Config", "Set gravitational constant (suggest 1e5 to 1e8):", initialvalue=default_gravity)
sim_dir = filedialog.askdirectory(title='Select Simulation Output Folder')
root.destroy()

G_CONSTANT = user_gravity if user_gravity else default_gravity
print(f"[Config] Gravitational Constant Set: {G_CONSTANT}")

if not sim_dir or not os.path.isdir(sim_dir):
    raise SystemExit("No valid simulation directory selected. Exiting.")

# CONFIG
PARTICLE_COLOR = (1.0, 0.5, 0.1, 0.8)
PARTICLE_SIZE = 2.0
FRAME_DELAY = 0.1  # seconds

# VISPY INIT
app.use_app('glfw')  # Safe WSL-compatible backend
canvas = scene.SceneCanvas(keys='interactive', show=True, title='Lilith OpenGL Viewer', bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=45, distance=2.5, elevation=30, azimuth=30)
scatter = Markers()
view.add(scatter)

# STATE
current_step = [0]
paused = [False]
frames_by_step = {}
max_step = 0

# KEY EVENT
@canvas.events.key_press.connect
def on_key(event):
    if event.key.name.lower() == 'space':
        paused[0] = not paused[0]
        print(f"[Toggle] Paused: {paused[0]}")

# INDEXING
step_cache = {}
def index_step(step):
    if step in step_cache:
        return step_cache[step]
    posfile = os.path.join(sim_dir, f"collapse_volume_step_{step:03d}.npy")
    typfile = os.path.join(sim_dir, f"types_step_{step:04d}.npy")
    if os.path.exists(posfile):
        step_cache[step] = (posfile, typfile if os.path.exists(typfile) else None)
    else:
        step_cache[step] = (None, None)
    return step_cache[step]

# UPDATE FUNC
@canvas.connect
def on_draw(ev):
    if paused[0]:
        return
    step = current_step[0]
    posfile, typfile = index_step(step)
    if not posfile:
        print(f"[Step {step}] No file found, looping to step 0...")
        current_step[0] = 0
        return
    try:
        pts = np.load(posfile)
        if pts.shape[1] >= 3:
            pts = pts[:, :3] / (np.max(np.abs(pts)) + 1e-9)
            scatter.set_data(pts, face_color=PARTICLE_COLOR, size=PARTICLE_SIZE, edge_width=0)
            print(f"[Step {step}] Rendered {len(pts)} particles")
    except Exception as e:
        print(f"[Error loading {posfile}] {e}")
    current_step[0] += 1

# TIMER LOOP
update_timer = app.Timer(interval=FRAME_DELAY, connect=lambda ev: canvas.update(), start=True)

if __name__ == '__main__':
    app.run()
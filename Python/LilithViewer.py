# LilithViewer.py
# OpenGL Shell Viewer for Lilith Collapse Field - 3D Rendering + Formic Logic Edition

import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from vispy import app, scene
app.use_app('pyqt5')
from vispy.color import Colormap

class Lilith3DViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.geometry("720x160")
        self.root.resizable(False, False)
        self.data_folder = filedialog.askdirectory(title="Select Collapse Output Folder")
        self.root.deiconify()
        if not self.data_folder or not os.path.isdir(self.data_folder):
            print("No valid folder selected. Exiting.")
            return
        self.current_volume = None
        self.step = 0

        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(up='z', fov=60)

        self.volume = scene.visuals.Volume(np.zeros((64, 64, 64)), parent=self.view.scene, method='mip')
        self.volume.cmap = Colormap(['black', 'purple', 'red', 'orange', 'yellow', 'white'])

        self.formic_overlay = scene.visuals.Markers(parent=self.view.scene)
        self.formic_overlay.set_gl_state('translucent', depth_test=True)

        if self.data_folder:
            self.volume_steps = self.get_volume_steps()
            if not self.volume_steps:
                print("[Lilith Viewer] No volume steps found in selected folder. Viewer will remain idle.")
                self.volume_steps = [0]
                self.setup_gui()
                return
            self.setup_gui()
            self.load_volume(step=self.volume_steps[0])

    def setup_gui(self):
        self.gain = tk.DoubleVar(value=5000.0)
        # Avoid duplicating Tk window setup

        frame = ttk.Frame(self.root)
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=10)
        frame.columnconfigure(2, weight=1)

        ttk.Button(frame, text="Play/Pause", command=self.toggle_play).grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(frame, text="Select Folder", command=self.select_folder).grid(row=0, column=0, padx=5, pady=5)
        self.volume_steps = self.get_volume_steps()
        max_step = max(self.volume_steps) if self.volume_steps else 0
        self.step_slider = ttk.Scale(frame, from_=0, to=max_step, orient='horizontal')
        self.step_slider.bind("<ButtonRelease-1>", self.slider_release)
        self.step_slider.bind("<B1-Motion>", self.slider_motion)
        self.step_update_job = None
        self.step_slider.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        ttk.Label(frame, text="Gain").grid(row=1, column=0, padx=5, pady=5)
        gain_slider = ttk.Scale(frame, from_=100, to=10000, orient='horizontal', variable=self.gain, command=self.update_gain_label)
        gain_slider.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.gain_label = ttk.Label(frame, text=f"{self.gain.get():.0f}")
        self.gain_label.grid(row=1, column=2, padx=5, pady=5)

        self.root.after(0, self.step_loop)
        self.canvas.show()

    def get_volume_steps(self):
        steps = []
        for fname in os.listdir(self.data_folder):
            if fname.startswith("collapse_volume_step_") and fname.endswith(".npy"):
                try:
                    step = int(fname.split("_")[-1].split(".")[0])
                    steps.append(step)
                except ValueError:
                    pass
        return sorted(steps)

    def select_folder(self):
        self.data_folder = filedialog.askdirectory()
        if self.data_folder:
            self.load_volume(step=0)

    def load_volume(self, step=0):
        path = os.path.join(self.data_folder, f"collapse_volume_step_{step:03d}.npy")
        gain = self.gain.get()
        if os.path.exists(path):
            vol = np.load(path)
            if vol.dtype != np.float32:
                vol = vol.astype(np.float32)
            print(f"[Step {step}] Volume Stats — min: {np.min(vol):.6e}, max: {np.max(vol):.6e}, mean: {np.mean(vol):.6e}")
            norm_vol = (vol * gain / np.max(vol)).astype(np.float32) if np.max(vol) > 0 else vol.astype(np.float32)
            self.volume.set_data(norm_vol)
            self.apply_formic_logic(vol)
            self.step = step

    def apply_formic_logic(self, vol):
        """
        Detect and highlight 'formic' emergent patterns: local maxima that could represent aggregation or convergence nodes.
        """
        threshold = np.percentile(vol, 98)
        indices = np.argwhere(vol > threshold)
        if indices.size > 0:
            positions = indices.astype(np.float32)
            colors = np.repeat([[1.0, 1.0, 1.0, 1.0]], positions.shape[0], axis=0)
            self.formic_overlay.set_data(positions, face_color=colors, size=5)
        else:
            self.formic_overlay.set_data(np.zeros((0, 3), dtype=np.float32))

    def update_step(self, val):
        self.load_volume(int(float(val)))

    def step_loop(self):
        if not hasattr(self, 'paused'):
            self.paused = False
        if not self.paused:
            self.step = self.step + 1
            if self.step not in self.volume_steps:
                self.step = self.volume_steps[0]
            self.step_slider.set(self.step)
            self.load_volume(self.step)
        self.root.after(100, self.step_loop)

    def start_timer_loop(self):
        pass  # No longer needed; mainloop is managed directly

    def slider_release(self, event):
        val = self.step_slider.get()
        self.update_step(val)

    def slider_motion(self, event):
        if self.step_update_job is not None:
            self.root.after_cancel(self.step_update_job)
        self.step_update_job = self.root.after(300, lambda: self.update_step(self.step_slider.get()))

    def update_gain_label(self, val):
        self.gain_label.config(text=f"{float(val):.0f}")

    def toggle_play(self):
        self.paused = not getattr(self, 'paused', False)
        print(f"[Toggle] Playback Paused: {self.paused}")


if __name__ == '__main__':
    viewer = Lilith3DViewer()
    viewer.root.mainloop()

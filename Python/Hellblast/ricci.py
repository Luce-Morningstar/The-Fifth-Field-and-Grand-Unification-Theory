import numpy as np
import cupy as cp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Load your pre-extracted spectrum
v_array, tb_array = np.loadtxt("spectrum.txt", unpack=True, comments='%')

# Normalize the HI brightness field
tb_norm = (tb_array - np.min(tb_array)) / (np.max(tb_array) - np.min(tb_array))

# Setup 2D radial grid
grid_size = 256
x = cp.linspace(-1, 1, grid_size)
y = cp.linspace(-1, 1, grid_size)
X, Y = cp.meshgrid(x, y)
R = cp.sqrt(X**2 + Y**2)

# Normalize velocity range to match R range
v_min, v_max = np.min(v_array), np.max(v_array)
r_scaled = (v_array - v_min) / (v_max - v_min)

# Interpolate brightness into radial grid
B_radial = np.interp(cp.asnumpy(R), r_scaled, tb_norm)
B_field = cp.asarray(gaussian_filter(B_radial, sigma=5))  # Diffuse it

# Real collapse stabilizer (mock A_ij)
A_field = cp.exp(-4 * (X**2 + Y**2))

# Ricci Tensor calculation
Ricci = cp.sqrt(A_field**2 + B_field**2)

# Normalize for plotting
Ricci_norm = (Ricci - Ricci.min()) / (Ricci.max() - Ricci.min())

# Convert to NumPy for rendering
X_np = cp.asnumpy(X)
Y_np = cp.asnumpy(Y)
Ricci_np = cp.asnumpy(Ricci_norm)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_np, Y_np, Ricci_np, cmap='inferno', edgecolor='none')

ax.set_title('Collapse Ricci Surface from Real HI Spectrum (AC G185.0–11.5)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Ricci Magnitude')
fig.colorbar(surf, label='Normalized Collapse Ricci')
plt.tight_layout()
plt.show()

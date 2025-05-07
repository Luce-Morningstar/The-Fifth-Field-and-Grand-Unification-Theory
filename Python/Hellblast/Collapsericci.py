import cupy as cp
import matplotlib.pyplot as plt

# Parameters
grid_size = 256
x = cp.linspace(-1, 1, grid_size)
y = cp.linspace(-1, 1, grid_size)
X, Y = cp.meshgrid(x, y)

# Real deformation field A_ij — collapse stabilizing structure
A = cp.exp(-4 * (X**2 + Y**2))  # centered density

# Imaginary HI-phase curvature B_ij (mocked from radial velocity arcs)
R = cp.sqrt(X**2 + Y**2)
B_theta = cp.sin(4 * cp.pi * R) * cp.exp(-4 * R**2)  # mimics HI-phase interference

# Collapse Ricci approximation
Ricci = cp.sqrt(A**2 + B_theta**2)

# Normalize for rendering
Ricci_norm = (Ricci - Ricci.min()) / (Ricci.max() - Ricci.min())

# Convert to NumPy for Matplotlib visualization
X_np = cp.asnumpy(X)
Y_np = cp.asnumpy(Y)
Ricci_np = cp.asnumpy(Ricci_norm)

# Plot Ricci field surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_np, Y_np, Ricci_np, cmap='inferno', edgecolor='none')

ax.set_title('Ricci Curvature Heightmap w/ HI Phase Overlay')
ax.set_xlabel('X (Mpc)')
ax.set_ylabel('Y (Mpc)')
ax.set_zlabel('Ricci Magnitude')
fig.colorbar(surf, label='Normalized Ricci Magnitude')
plt.tight_layout()
plt.show()

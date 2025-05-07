import cupy as cp
import numpy as np
import time

def generate_legendre_matrix_nuclear(theta_cp, lmax):
    N = theta_cp.size
    x = cp.cos(theta_cp)
    sqrt1mx2 = cp.sqrt(1.0 - x**2 + 1e-12)

    P = cp.zeros((lmax + 1, lmax + 1, N), dtype=cp.float32)
    P[0, 0, :] = 1.0
    if lmax >= 1:
        P[1, 0, :] = x
        P[1, 1, :] = sqrt1mx2

    for l in range(2, lmax + 1):
        for m in range(0, l + 1):
            if m == l:
                P[l, m, :] = (2 * l - 1)**0.5 * sqrt1mx2 * P[l - 1, m - 1, :]
            elif m == l - 1:
                P[l, m, :] = (2 * l - 1) * x * P[l - 1, m, :]
            else:
                P[l, m, :] = ((2 * l - 1) * x * P[l - 1, m, :] - (l + m - 1) * P[l - 2, m, :]) / (l - m + 1e-6)

    return P

def generate_legendre_matrix_nuclear_chunked(theta, lmax, chunk_size=10000):
    """
    Chunked version of the GPU-heavy Legendre matrix generator.
    Avoids full allocation overload by generating P_l^m(theta) in blocks.
    """
    N = theta.shape[0]
    P = cp.zeros((lmax + 1, lmax + 1, N), dtype=cp.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        theta_chunk = theta[start:end].get()  # bring to CPU
        for l in range(lmax + 1):
            for m in range(0, l + 1):
                Plm_chunk = scipy.special.lpmv(m, l, np.cos(theta_chunk))
                P[l, m, start:end] = cp.asarray(Plm_chunk, dtype=cp.float32)

    return P

# Run benchmark
theta = np.linspace(0, np.pi, 100000, dtype=np.float32)
theta_cp = cp.asarray(theta)
lmax = 256

start_gpu = time.time()
P_gpu = generate_legendre_matrix_nuclear(theta_cp, lmax)
cp.cuda.Device(0).synchronize()
end_gpu = time.time()

print(f"[🔥] Nuclear Legendre Matrix Generated — Time: {end_gpu - start_gpu:.4f} s")
print(f"[✓] Matrix shape: {P_gpu.shape}")

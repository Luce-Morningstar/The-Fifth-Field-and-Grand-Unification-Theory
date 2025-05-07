import os
os.environ["NUMBA_CPU_FEATURES"] = "+avx512"
os.environ["NUMBA_NUM_THREADS"] = "14"
import numpy as np
import cupy as cp
from scipy.special import lpmv
import time
from numba import njit, prange, config
from numba import config, __version__ as numba_version
print("Numba Version:", numba_version)
print("AVX-512 Enabled:", config.CPU_NAME)

theta = np.linspace(0, np.pi, 10000)
theta_cp = cp.asarray(theta)
lmax = 256
x = np.cos(theta)

# Precompute full (l, m) coefficient matrix using lpmv outside the jit scope
P_lm_values = np.empty((lmax + 1, lmax + 1, len(x)), dtype=np.float64)
for l in range(lmax + 1):
    for m in range(0, l + 1):
        P_lm_values[l, m, :] = lpmv(m, l, x)

@njit(parallel=True, fastmath=True)
def copy_legendre_matrix(P_src, P_dst):
    lmax, _, N = P_src.shape
    for l in prange(lmax):
        for m in range(l + 1):
            for i in range(N):
                P_dst[l, m, i] = P_src[l, m, i]

    return P_dst

def generate_legendre_matrix_cpu_avx(theta, lmax):
    N = len(theta)
    P_pre = P_lm_values[:, :, :N]
    P_out = np.zeros_like(P_pre)
    return copy_legendre_matrix(P_pre, P_out)

def generate_legendre_matrix_gpu(theta_cp, lmax):
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


# === BENCHMARK ===
start_cpu = time.time()
P_cpu = generate_legendre_matrix_cpu_avx(theta, lmax)
end_cpu = time.time()

start_gpu = time.time()
P_gpu = generate_legendre_matrix_gpu(theta_cp, lmax)
cp.cuda.Device(0).synchronize()
end_gpu = time.time()

print(f"[⛓️ AVX-512 Multithreaded CPU Time] {end_cpu - start_cpu:.4f} s")
print(f"[⚡ GPU Time] {end_gpu - start_gpu:.4f} s")

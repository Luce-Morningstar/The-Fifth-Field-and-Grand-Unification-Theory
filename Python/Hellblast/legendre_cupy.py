import cupy as cp

def generate_legendre_matrix(theta, lmax):
    """
    Compute associated Legendre polynomials P_l^m(cosθ) for all l, m up to lmax on GPU.
    
    Parameters:
    - theta (cp.ndarray): 1D array of theta angles (radians)
    - lmax (int): Maximum degree l

    Returns:
    - P (cp.ndarray): CuPy array of shape (lmax+1, lmax+1, N) where N = len(theta)
                      P[l, m, i] = P_l^m(cosθ_i)
    """
    N = theta.size
    x = cp.cos(theta)
    P = cp.zeros((lmax + 1, lmax + 1, N), dtype=cp.float32)

    # Initial values
    P[0, 0] = 1.0
    if lmax > 0:
        P[1, 0] = x
        P[1, 1] = cp.sqrt(1.0 - x**2)

    for l in range(2, lmax + 1):
        for m in range(0, l + 1):
            if m == l:
                P[l, m] = (2 * l - 1)**0.5 * cp.sqrt(1 - x**2) * P[l - 1, m - 1]
            elif m == l - 1:
                P[l, m] = (2 * l - 1) * x * P[l - 1, m]
            else:
                num = (2 * l - 1) * x * P[l - 1, m] - (l + m - 1) * P[l - 2, m]
                denom = l - m
                P[l, m] = num / denom

    return P

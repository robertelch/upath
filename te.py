import numpy as np
import time
from numba import njit, prange

# -----------------------------
# Original nested-loop function
# -----------------------------
def generate_lattice_points_loops(basis, limit):
    points = []
    for i in range(-limit, limit+1):
        for j in range(-limit, limit+1):
            for k in range(-limit, limit+1):
                p = i*basis[0] + j*basis[1] + k*basis[2]
                points.append(p.astype(float))
    return np.array(points, dtype=float)

# -----------------------------
# Vectorized NumPy version
# -----------------------------
def generate_lattice_points_vectorized(basis, limit):
    rng = np.arange(-limit, limit + 1)
    I, J, K = np.meshgrid(rng, rng, rng, indexing='ij')
    coeffs = np.vstack([I.ravel(), J.ravel(), K.ravel()]).T
    points = coeffs @ basis
    return points.astype(float)

# -----------------------------
# Numba parallel optimized version
# -----------------------------
@njit(parallel=True)
def generate_lattice_points_numba(basis, limit):
    n = 2*limit + 1
    num_points = n**3
    points = np.empty((num_points, 3), dtype=np.float64)
    idx = 0
    for i in prange(-limit, limit+1):
        for j in range(-limit, limit+1):
            for k in range(-limit, limit+1):
                points[idx, 0] = i*basis[0,0] + j*basis[1,0] + k*basis[2,0]
                points[idx, 1] = i*basis[0,1] + j*basis[1,1] + k*basis[2,1]
                points[idx, 2] = i*basis[0,2] + j*basis[1,2] + k*basis[2,2]
                idx += 1
    return points

# -----------------------------
# Benchmarking function
# -----------------------------
def stress_test_all(basis, max_limit=6):
    print(f"{'Limit':>5} | {'Loops (s)':>12} | {'Vectorized (s)':>15} | {'Numba (s)':>12}")
    print("-"*55)
    
    for limit in range(50, max_limit+1):
        # Loops
        start = time.perf_counter()
        generate_lattice_points_loops(basis, limit)
        loop_time = time.perf_counter() - start

        # Vectorized
        start = time.perf_counter()
        generate_lattice_points_vectorized(basis, limit)
        vec_time = time.perf_counter() - start

        # Numba
        start = time.perf_counter()
        generate_lattice_points_numba(basis, limit)
        numba_time = time.perf_counter() - start

        print(f"{limit:>5} | {loop_time:>12.6f} | {vec_time:>15.6f} | {numba_time:>12.6f}")

# -----------------------------
# Example usage
# -----------------------------
basis = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)  # cubic lattice

generate_lattice_points_numba(basis, 1)
print("second now")
start = time.perf_counter()
generate_lattice_points_vectorized(basis, 1000)
print(time.perf_counter() - start)
#stress_test_all(basis, max_limit=1000)


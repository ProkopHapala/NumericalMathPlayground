"""
Test GPU GEMM kernels (tall-skinny and skinny-transpose) against NumPy reference.

These two GEMM kernels are the building blocks of the GPU similarity transform
H_proj = V^T H V used in Ritz correction (see nested_solver.py). They are
specialized for the shapes that arise in block-diagonal solvers:

1. Tall-skinny GEMM: C = A^T @ B
   A is [K, N] (large, square-ish), B is [K, m] (tall-skinny, m << K).
   Output C is [N, m]. Used for the first step: temp = H @ V.

2. Skinny-transpose GEMM: C = A^T @ B
   A and B are both [N, m] (tall-skinny). Output C is [m, m] (small square).
   Used for the second step: H_proj = V^T @ temp.

Both kernels use 2D tiled local-memory GEMM for GPU efficiency.
This script verifies numerical correctness (float32) against NumPy double
precision for various N, K, m combinations.

Usage:
    python test_gpu_gemm.py
"""

import sys, time
sys.path.insert(0, '.')
import numpy as np
from nested_solver import OpenCLBlockJacobi


def test_tall_skinny():
    print("=== Tall-skinny GEMM: C = A @ B ===")
    print("A: [K, N] column-major, B: [K, m], C: [N, m]")

    solver = OpenCLBlockJacobi()

    for N, K, m in [(492, 492, 8), (492, 492, 16), (492, 492, 32), (810, 810, 32), (492, 492, 64), (492, 492, 128), (492, 492, 256)]:
        # A_colmaj shape = (K, N) -- A[k, n] stored contiguously in n
        A = np.random.randn(K, N).astype(np.float32)
        B = np.random.randn(K, m).astype(np.float32)

        # CPU reference: C = A.T @ B  (A.T is [N, K], B is [K, m])
        C_ref = A.T @ B

        # GPU kernel
        t0 = time.time()
        C_gpu = solver.gemm_tall_skinny(A, B, N, K, m)
        t_gpu = time.time() - t0

        err = np.max(np.abs(C_gpu - C_ref))
        rel = err / (np.max(np.abs(C_ref)) + 1e-12)

        print(f"  N={N:4d} K={K:4d} m={m:2d}: max_abs_err={err:.2e}, max_rel_err={rel:.2e}, time={t_gpu*1000:.2f}ms")
        assert rel < 1e-4, f"Tall-skinny GEMM failed: rel_err={rel}"


def test_skinny_transpose():
    print("\n=== Skinny-transpose GEMM: C = A.T @ B ===")
    print("A, B: [N, m], C: [m, m]")

    solver = OpenCLBlockJacobi()

    for N, m in [(492, 8), (492, 16), (492, 32), (810, 32), (492, 64), (492, 128), (492, 256)]:
        A = np.random.randn(N, m).astype(np.float32)
        B = np.random.randn(N, m).astype(np.float32)

        C_ref = A.T @ B

        t0 = time.time()
        C_gpu = solver.gemm_skinny_transpose(A, B, N, m)
        t_gpu = time.time() - t0

        err = np.max(np.abs(C_gpu - C_ref))
        rel = err / (np.max(np.abs(C_ref)) + 1e-12)

        print(f"  N={N:4d} m={m:2d}: max_abs_err={err:.2e}, max_rel_err={rel:.2e}, time={t_gpu*1000:.2f}ms")
        assert rel < 1e-4, f"Skinny-transpose GEMM failed: rel_err={rel}"


if __name__ == "__main__":
    test_tall_skinny()
    test_skinny_transpose()
    print("\nAll GEMM kernel tests passed.")

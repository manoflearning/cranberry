import os
os.environ['OMP_NUM_THREADS'] = '1'

from cranberry import Tensor
import numpy as np
import time

N = 1024

def gemm_numpy():
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s_np = et - st
    return s_np

def gemm_cranberry():
    A = Tensor.randn(N, N)
    B = Tensor.randn(N, N)
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s_cr = et - st
    return s_cr

if __name__ == '__main__':
    flop = N*N*2*N
    print(f'{flop * 1e-9:.2f} GFLOP')

    # numpy
    s_np = gemm_numpy()
    print(f'NumPy:\t\t{flop / s_np * 1e-9:.2f} GFLOP/S')

    # cranberry
    s_cr = gemm_cranberry()
    print(f'Cranberry:\t{flop / s_cr * 1e-9:.2f} GFLOP/S')

    # speed check
    print(f'Speedup:\t{s_np / s_cr:.2f}x')
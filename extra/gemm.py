import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import cranberry as cr
import time

N = 1024

if __name__ == '__main__':
    flop = N*N*2*N
    print(f'{flop / 1e9:.2f} GFLOP')

    # numpy
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s_numpy = et - st
    print(f'NumPy:\t\t{flop / s_numpy * 1e-9:.2f} GFLOP/S')

    # cranberry
    P = cr.Tensor(A)
    Q = cr.Tensor(B)
    st = time.monotonic()
    R = P @ Q
    et = time.monotonic()
    s_cranberry = et - st
    print(f'cranberry:\t{flop / s_cranberry * 1e-9:.2f} GFLOP/S')

    # check
    print(f'Speedup:\t{s_numpy / s_cranberry:.2f}x')
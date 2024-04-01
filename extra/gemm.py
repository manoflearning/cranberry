import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import cranberry as cr
import time

N = 1024

if __name__ == '__main__':
    flop = N*N*2*N
    print(f'{flop * 1e-9:.2f} GFLOP')

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    P = cr.Tensor(A)
    Q = cr.Tensor(B)

    # numpy
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s_np = et - st
    print(f'NumPy:\t\t{flop / s_np * 1e-9:.2f} GFLOP/S')

    # cranberry
    st = time.monotonic()
    R = P @ Q
    et = time.monotonic()
    s_cr = et - st
    print(f'Cranberry:\t{flop / s_cr * 1e-9:.2f} GFLOP/S')

    # speed check
    print(f'Speedup:\t{s_np / s_cr:.2f}x')
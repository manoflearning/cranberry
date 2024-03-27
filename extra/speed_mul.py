import cranberry as cb
import numpy as np
import time

N = 1024

def speed_mul():
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # numpy
    st = time.monotonic()
    for _ in range(10000):
        C = A * B
    et = time.monotonic()
    s_np = et - st
    print(f'NumPy:\t\t {s_np:.2f} s')

    # cranberry
    A = cb.Tensor(A)
    B = cb.Tensor(B)

    st = time.monotonic()
    for _ in range(10000):
        C = A * B
    et = time.monotonic()
    s_cr = et - st
    print(f'Cranberry:\t {s_cr:.2f} s')

    # speed check
    print(f'Speedup:\t {s_np / s_cr:.2f}x')

if __name__ == '__main__':
    speed_mul()
import cranberry as cr
from cranberry import nn
from cranberry import SGD
import torch
import numpy as np

A = np.ones((2, 3))
B = np.ones((3))

P = cr.Tensor(A)
Q = cr.Tensor(B)

print(A + B)
print((P + Q).numpy())

optimizer = cr.SGD([P, Q], lr=0.1)

optimizer.step()
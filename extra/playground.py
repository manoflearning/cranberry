import cranberry as cr
import numpy as np

a = cr.Tensor(np.array([1, 2, 3], dtype=np.float32))
b = cr.Tensor(np.array([3, 2, 1], dtype=np.float32))

c = a + b
d = c * a
e = d / a

print(d)
print(e)

d.backward()
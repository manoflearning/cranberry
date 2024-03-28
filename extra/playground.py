import cranberry as cb
import numpy as np

list = [[1, 1, 1]]

a = cb.Tensor(np.array(list))
b = cb.Tensor([3, 2, 2])

c = a * b

print(c)
print(c.shape)
print(c.grad)

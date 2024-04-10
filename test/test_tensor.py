import cranberry as cr
import numpy as np
import unittest

np.random.seed(1337)

A_np = np.random.randn(3, 3).astype(np.float32)
B_np = np.random.randn(3, 3).astype(np.float32)
C_np = np.random.randn(1, 3).astype(np.float32)

class TestCranberry(unittest.TestCase):
    def test_zerodim_initialization(self):
        a = cr.Tensor(55)
        b = cr.Tensor(156.342)

        # TODO: if data is scalar, a.shape == () not == []
        self.assertEqual(a.shape, [])
        self.assertEqual(b.shape, [])

    def test_add(self):
        A = cr.Tensor(A_np)
        B = cr.Tensor(B_np)
        np.testing.assert_allclose(A_np + B_np, (A + B).numpy())

    def test_add_broadcast(self):
        A = cr.Tensor(A_np)
        C = cr.Tensor(C_np)
        np.testing.assert_allclose(A_np + C_np, (A + C).numpy())

    def test_matmul(self):
        A = cr.Tensor(A_np)
        B = cr.Tensor(B_np)
        np.testing.assert_allclose(A_np @ B_np, (A @ B).numpy(), rtol=1e-7, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
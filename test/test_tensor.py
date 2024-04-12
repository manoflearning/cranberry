import numpy as np
import cranberry as cr
import torch
import unittest

np.random.seed(1337)

A_np = np.random.randn(5, 5).astype(np.float32)
B_np = np.random.randn(5, 5).astype(np.float32)
C_np = np.random.randn(5, 5).astype(np.float32)

Z_np = np.random.randn(1, 1).astype(np.float32)

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
        Z = cr.Tensor(Z_np)
        np.testing.assert_allclose(A_np + Z_np, (A + Z).numpy())

    def test_matmul(self):
        A = cr.Tensor(A_np)
        B = cr.Tensor(B_np)
        np.testing.assert_allclose(A_np @ B_np, (A @ B).numpy(), rtol=1e-7, atol=1e-7)

    def test_backward_pass_1(self):
        def run_cranberry():
            A = cr.Tensor(A_np, requires_grad=True)
            B = cr.Tensor(B_np, requires_grad=True)
            C = cr.Tensor(C_np)
            Z = cr.Tensor(Z_np)
            out = A.matmul(B)
            out = out.relu()
            out = out * C
            out = out - Z
            out = out.pow(4)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        def run_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            C = torch.tensor(C_np)
            Z = torch.tensor(Z_np)
            out = A.matmul(B)
            out = out.relu()
            out = out * C
            out = out - Z
            out = out.pow(4)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        for x, y in zip(run_cranberry(), run_pytorch()):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)

    def test_backward_pass_2(self):
        def run_cranberry():
            x = cr.Tensor(-4.0, requires_grad=True)
            z = cr.Tensor(2.0) * x + cr.Tensor(2.0) + x
            q = z.relu() + z * x
            h = (z * z).relu()
            y = h + q + q * x
            y.backward()
            return y.detach().numpy(), x.grad

        def run_pytorch():
            x = torch.tensor(-4.0, requires_grad=True, dtype=torch.float32)
            z = 2.0 * x + 2.0 + x
            q = z.relu() + z * x
            h = (z * z).relu()
            y = h + q + q * x
            y.backward()
            return y.detach().numpy(), x.grad
        
        for x, y in zip(run_cranberry(), run_pytorch()):
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
import numpy as np
from cranberry import Tensor
import torch
import unittest

np.random.seed(1337)

A_np = np.random.randn(5, 5).astype(np.float32)
B_np = np.random.randn(5, 5).astype(np.float32)
C_np = np.random.randn(5, 5).astype(np.float32)

# for broadcasting
Z_np = np.random.randn(1, 1).astype(np.float32)

rtol, atol = 1e-6, 1e-6

class TestCranberry(unittest.TestCase):
    def test_zerodim_initialization(self):
        a = Tensor(55)
        b = Tensor(156.342)

        self.assertEqual(a.shape, ())
        self.assertEqual(b.shape, ())

    def test_add(self):
        A = Tensor(A_np)
        B = Tensor(B_np)
        np.testing.assert_allclose(A_np + B_np, (A + B).numpy())

    def test_add_broadcast(self):
        A = Tensor(A_np)
        Z = Tensor(Z_np)
        np.testing.assert_allclose(A_np + Z_np, (A + Z).numpy())

    def test_matmul(self):
        A = Tensor(A_np)
        B = Tensor(B_np)
        np.testing.assert_allclose(A_np @ B_np, (A @ B).numpy(), rtol, atol)

    def test_broadcast_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            Z = Tensor(Z_np, requires_grad=True)
            out = A + Z
            out = out.sum()
            out.backward()
            print(out.grad)
            return out.detach().numpy(), A.grad, Z.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            Z = torch.tensor(Z_np, requires_grad=True)
            p = A + Z
            out = p.sum()
            out.backward()
            return out.detach().numpy(), A.grad, Z.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    def test_add_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = A + B
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = A + B
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    def test_mul_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = A * B
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = A * B
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    def test_matmul_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = A @ B
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = A @ B
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_relu_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.relu()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.relu()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    def test_exp_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.exp()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.exp()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_log_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.log()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.log()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_neg_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = -A
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = -A
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    def test_sum_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    # reshape op doesn't do actual backprop
    # but it shares the same storage as the original tensor
    def test_reshape_backward(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.reshape(1, 25)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.reshape(1, 25)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        for (x, y) in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_backward_pass_0(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            C = Tensor(C_np)
            Z = Tensor(Z_np)
            out = A.matmul(B)
            out = out.relu()
            out = out * C
            out = out - Z
            out = out * out * out * out
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            C = torch.tensor(C_np)
            Z = torch.tensor(Z_np)
            out = A.matmul(B)
            out = out.relu()
            out = out * C
            out = out - Z
            out = out * out * out * out
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_backward_pass_1(self):
        def test_cranberry():
            x = Tensor(-4.0, requires_grad=True)
            z = 2.0 * x + 2.0 + x
            q = z.relu() + z * x
            h = (z * z).relu()
            y = h + q + q * x
            y.backward()
            return y.detach().numpy(), x.grad

        def test_pytorch():
            x = torch.tensor(-4.0, requires_grad=True, dtype=torch.float32)
            z = 2.0 * x + 2.0 + x
            q = z.relu() + z * x
            h = (z * z).relu()
            y = h + q + q * x
            y.backward()
            return y.detach().numpy(), x.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

if __name__ == '__main__':
    unittest.main()
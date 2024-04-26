import numpy as np
from cranberry import Tensor
import torch
import unittest

Tensor.manual_seed(1337)
np.random.seed(1337)

N, M, K = 54, 29, 17 # do not change these values
A_np = np.random.randn(N, M).astype(np.float32)
B_np = np.random.randn(N, M).astype(np.float32)
C_np = np.random.randn(M, K).astype(np.float32)
D_np = np.random.randn(1).astype(np.float32)

rtol, atol = 1e-4, 1e-4 # is this acceptable?

class TestCranberry(unittest.TestCase):
    def test_zerodim_initialization(self):
        a = Tensor(55)
        b = Tensor(156.342)

        self.assertEqual(a.shape, ())
        self.assertEqual(b.shape, ())

    # unary ops: neg, sqrt, relu, exp, log, sigmoid

    def test_neg(self):
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
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sqrt(self): # include negative values
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.sqrt()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.sqrt()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_relu(self):
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
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_exp(self):
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
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_log(self): # include nan/inf tests
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
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)
    
    def test_sigmoid(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.sigmoid()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.sigmoid()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # binary ops: add, sub, mul, div

    def test_add(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = (A + B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = (A + B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sub(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = (A - B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = (A - B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_mul(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = (A * B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = (A * B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_div(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = (A / B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = (A / B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # reduce ops: sum, mean

    def test_sum(self):
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
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sum_axis(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.sum(dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.sum(dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_mean(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.mean()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.mean()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_mean_axis(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.mean(dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.mean(dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # movement ops: reshape, expand, permute, flatten, transpose

    def test_reshape(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.reshape(M, -1, N)
            out = out * out + out
            # out = out.reshape(29, 27, -1, 1, 2)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.reshape(M, -1, N)
            out = out * out + out
            # out = out.reshape(29, 27, -1, 1, 2)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_expand(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            p = A.expand(2, 3, 1, 5, N, M)
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            p = A.expand(2, 3, 1, 5, N, M)
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_permute(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            p = A.permute(1, 0)
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            p = A.permute(1, 0)
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_flatten(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            p = A.flatten()
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            p = A.flatten()
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_transpose(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            p = A.transpose(1, 0)
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            p = A.transpose(1, 0)
            out = p * p + p
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # processing ops: matmul

    def test_matmul_2d(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            C = Tensor(C_np, requires_grad=True)
            out = A.matmul(C)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, C.grad
        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            C = torch.tensor(C_np, requires_grad=True)
            out = A.matmul(C)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, C.grad
        
        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # functional nn ops: linear, sparse_categorical_crossentropy

    # def test_add(self):
    #     A = Tensor(A_np)
    #     B = Tensor(B_np)
    #     np.testing.assert_allclose(A_np + B_np, (A + B).numpy())

    # def test_add_broadcast(self):
    #     A = Tensor(A_np)
    #     Z = Tensor(Z_np)
    #     np.testing.assert_allclose(A_np + Z_np, (A + Z).numpy())

    # def test_matmul(self):
    #     A = Tensor(A_np)
    #     B = Tensor(B_np)
    #     np.testing.assert_allclose(A_np @ B_np, (A @ B).numpy(), rtol, atol)

    # def test_broadcast_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         Z = Tensor(Z_np, requires_grad=True)
    #         out = A + Z
    #         out = out.sum()
    #         out.backward()
    #         print(out.grad)
    #         return out.detach().numpy(), A.grad, Z.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         Z = torch.tensor(Z_np, requires_grad=True)
    #         p = A + Z
    #         out = p.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, Z.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)
    
    # def test_add_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         B = Tensor(B_np, requires_grad=True)
    #         out = A + B
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         B = torch.tensor(B_np, requires_grad=True)
    #         out = A + B
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)
    
    # def test_mul_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         B = Tensor(B_np, requires_grad=True)
    #         out = A * B
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         B = torch.tensor(B_np, requires_grad=True)
    #         out = A * B
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)
    
    # def test_matmul_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         B = Tensor(B_np, requires_grad=True)
    #         out = A @ B
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         B = torch.tensor(B_np, requires_grad=True)
    #         out = A @ B
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)

    # def test_relu_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         out = A.relu()
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         out = A.relu()
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)
    
    # def test_exp_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         out = A.exp()
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         out = A.exp()
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)

    # def test_log_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         out = A.log()
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         out = A.log()
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)

    # def test_neg_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         out = -A
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         out = -A
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)
    
    # def test_sum_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         out = A.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         out = A.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)
    
    # # reshape op doesn't do actual backprop
    # # but it shares the same storage as the original tensor
    # def test_reshape_backward(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         out = A.reshape(1, 25)
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         out = A.reshape(1, 25)
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad

    #     for (x, y) in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)

    # def test_backward_pass_0(self):
    #     def test_cranberry():
    #         A = Tensor(A_np, requires_grad=True)
    #         B = Tensor(B_np, requires_grad=True)
    #         C = Tensor(C_np)
    #         Z = Tensor(Z_np)
    #         out = A.matmul(B)
    #         out = out.relu()
    #         out = out * C
    #         out = out - Z
    #         out = out * out * out * out
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad
        
    #     def test_pytorch():
    #         A = torch.tensor(A_np, requires_grad=True)
    #         B = torch.tensor(B_np, requires_grad=True)
    #         C = torch.tensor(C_np)
    #         Z = torch.tensor(Z_np)
    #         out = A.matmul(B)
    #         out = out.relu()
    #         out = out * C
    #         out = out - Z
    #         out = out * out * out * out
    #         out = out.sum()
    #         out.backward()
    #         return out.detach().numpy(), A.grad, B.grad

    #     for x, y in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)

    # def test_backward_pass_1(self):
    #     def test_cranberry():
    #         x = Tensor(-4.0, requires_grad=True)
    #         y = 2.0 * x + 2.0 + x
    #         # q = z.relu() + z * x
    #         # y = (z * z).relu()
    #         # y = h + q + q * x
    #         y.backward()
    #         return y.detach().numpy(), x.grad

    #     def test_pytorch():
    #         x = torch.tensor(-4.0, requires_grad=True, dtype=torch.float32)
    #         y = 2.0 * x + 2.0 + x
    #         # q = z.relu() + z * x
    #         # y = (z * z).relu()
    #         # y = h + q + q * x
    #         y.backward()
    #         return y.detach().numpy(), x.grad
        
    #     for x, y in zip(test_cranberry(), test_pytorch()):
    #         np.testing.assert_allclose(x, y, rtol, atol)

if __name__ == '__main__':
    unittest.main()
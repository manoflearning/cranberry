import numpy as np
from cranberry import Tensor
import torch
import unittest

np.random.seed(1337)
Tensor.manual_seed(1337)
torch.manual_seed(1337)

N, M, K = 54, 29, 17  # do not change these values
A_np = np.random.randn(N, M).astype(np.float32)
Y_np = np.random.randint(0, 10, N)
B_np = np.random.randn(N, M).astype(np.float32)
C_np = np.random.randn(M, K).astype(np.float32)
D_np = np.random.randn(1).astype(np.float32)
E_np = np.random.randn(K).astype(np.float32)

A_1d = np.random.randn(N).astype(np.float32)
B_1d = np.random.randn(M).astype(np.float32)

rtol, atol = 1e-4, 1e-4  # is this acceptable?


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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sqrt(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = (A.relu() + 1e-3).sqrt()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = (A.relu() + 1e-3).sqrt()
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_log(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = (A.relu() + 1e-3).log()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = (A.relu() + 1e-3).log()
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

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
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

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
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

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
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # reduce ops: sum, max, mean, softmax, log_softmax

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sum_dim_0(self):
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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sum_dim_1(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.sum(dim=1, keepdim=True)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.sum(dim=1, keepdim=True)
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_max(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.max()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.max()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_max_dim_0(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.max(dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.max(dim=1).values.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_max_dim_1(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.max(dim=1, keepdim=True)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.max(dim=1, keepdim=True).values.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_softmax_0(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.softmax()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.softmax(dim=-1)
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_softmax_1(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.softmax(dim=-2)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.softmax(dim=-2)
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_log_softmax_0(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.log_softmax()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.log_softmax(dim=-1)
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_log_softmax_1(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            out = A.log_softmax(dim=-2)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            out = A.log_softmax(dim=-2)
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_expand(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            p = A.expand(2, 3, 1, 5, N, M)
            out = p * p + p
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            p = A.expand(2, 3, 1, 5, N, M)
            out = p * p + p
            out = out.sum()
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

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
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # processing ops: matmul

    def test_matml_1d_1d(self):
        def test_cranberry():
            A = Tensor(A_1d, requires_grad=True)
            B = Tensor(A_1d, requires_grad=True)
            out = A.matmul(B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        def test_pytorch():
            A = torch.tensor(A_1d, requires_grad=True)
            B = torch.tensor(A_1d, requires_grad=True)
            out = A.matmul(B).sum()
            out.backward()
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_matmul_2d_2d(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            C = Tensor(C_np, requires_grad=True)
            out = A.matmul(C).sum()
            out.backward()
            return out.detach().numpy(), A.grad, C.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            C = torch.tensor(C_np, requires_grad=True)
            out = A.matmul(C).sum()
            out.backward()
            assert A.grad is not None and C.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                C.grad.detach().numpy(),
            )

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_matmul_1d_2d(self):
        def test_cranberry():
            A = Tensor(A_1d, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            out = A.matmul(B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        def test_pytorch():
            A = torch.tensor(A_1d, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            out = A.matmul(B).sum()
            out.backward()
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_matmul_2d_1d(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_1d, requires_grad=True)
            out = A.matmul(B).sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_1d, requires_grad=True)
            out = A.matmul(B).sum()
            out.backward()
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # functional nn ops: linear, sparse_categorical_crossentropy

    def test_linear(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            W = Tensor(C_np)
            b = Tensor(E_np)
            out = A.linear(W, b).sum()
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            W = torch.tensor(C_np)
            b = torch.tensor(E_np)
            out = torch.nn.functional.linear(
                A, W.transpose(1, 0), b
            ).sum()  # transpose to match cranberry
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    def test_sparse_categorical_crossentropy(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            Y = Tensor(Y_np)
            out = A.sparse_categorical_crossentropy(Y)
            out.backward()
            return out.detach().numpy(), A.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            Y = torch.tensor(Y_np)
            out = torch.nn.CrossEntropyLoss(reduction="mean")(A, Y)
            out.backward()
            assert A.grad is not None
            return out.detach().numpy(), A.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)

    # others

    def test_backward_pass_0(self):
        def test_cranberry():
            A = Tensor(A_np, requires_grad=True)
            B = Tensor(B_np, requires_grad=True)
            D = Tensor(D_np)
            out = A.matmul(B.transpose(1, 0))
            out = out.relu()
            out = out - D
            out = out * out
            out = out.sum()
            out.backward()
            return out.detach().numpy(), A.grad, B.grad

        def test_pytorch():
            A = torch.tensor(A_np, requires_grad=True)
            B = torch.tensor(B_np, requires_grad=True)
            D = torch.tensor(D_np)
            out = A.matmul(B.transpose(1, 0))
            out = out.relu()
            out = out - D
            out = out * out
            out = out.sum()
            out.backward()
            assert A.grad is not None and B.grad is not None
            return (
                out.detach().numpy(),
                A.grad.detach().numpy(),
                B.grad.detach().numpy(),
            )

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
            assert x.grad is not None
            return y.detach().numpy(), x.grad.detach().numpy()

        for x, y in zip(test_cranberry(), test_pytorch()):
            np.testing.assert_allclose(x, y, rtol, atol)


if __name__ == "__main__":
    unittest.main()

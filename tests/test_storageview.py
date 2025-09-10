import numpy as np
import unittest

from cranberry import StorageView


np.random.seed(1337)

N, M = 54, 29  # keep in sync with test_tensor.py
A_np = np.random.randn(N * M).astype(np.float32)
B_np = np.random.randn(N * M).astype(np.float32)

rtol, atol = 1e-5, 1e-5


class TestStorageView(unittest.TestCase):
    def test_from_vec_len_to_vec(self):
        v = StorageView.from_vec([1.0, 2.0, 3.5], "cpu")
        self.assertEqual(v.len(), 3)
        self.assertEqual(v.to_vec(), [1.0, 2.0, 3.5])

    def test_full(self):
        v = StorageView.full(2.5, 4, "cpu")
        self.assertEqual(v.len(), 4)
        self.assertEqual(v.to_vec(), [2.5, 2.5, 2.5, 2.5])

    # unary ops: neg, sqrt, exp, log

    def test_neg_1d(self):
        def test_cranberry():
            a = StorageView.from_vec(A_np.tolist(), "cpu")
            out = a.neg()
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return -A_np

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_sqrt_1d(self):
        # ensure positivity
        data = (np.abs(A_np) + 1e-3).astype(np.float32)

        def test_cranberry():
            a = StorageView.from_vec(data.tolist(), "cpu")
            out = a.sqrt()
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return np.sqrt(data)

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_exp_1d(self):
        # clamp to avoid inf in exp
        data = np.clip(A_np, -10, 10)

        def test_cranberry():
            a = StorageView.from_vec(data.tolist(), "cpu")
            out = a.exp()
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return np.exp(data)

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_log_1d(self):
        data = (np.abs(A_np) + 1e-3).astype(np.float32)

        def test_cranberry():
            a = StorageView.from_vec(data.tolist(), "cpu")
            out = a.log()
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return np.log(data)

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    # binary ops: add, sub, mul, div (1D and 2D contiguous)

    def test_add_1d(self):
        def test_cranberry():
            a = StorageView.from_vec(A_np.tolist(), "cpu")
            b = StorageView.from_vec(B_np.tolist(), "cpu")
            out = a.add(b)
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return A_np + B_np

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_sub_1d(self):
        def test_cranberry():
            a = StorageView.from_vec(A_np.tolist(), "cpu")
            b = StorageView.from_vec(B_np.tolist(), "cpu")
            out = a.sub(b)
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return A_np - B_np

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_mul_1d(self):
        def test_cranberry():
            a = StorageView.from_vec(A_np.tolist(), "cpu")
            b = StorageView.from_vec(B_np.tolist(), "cpu")
            out = a.mul(b)
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return A_np * B_np

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_div_1d(self):
        bnp = B_np.copy()
        bnp[bnp == 0] = 1.0

        def test_cranberry():
            a = StorageView.from_vec(A_np.tolist(), "cpu")
            b = StorageView.from_vec(bnp.tolist(), "cpu")
            out = a.div(b)
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            return A_np / bnp

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_unary_2d_then_flatten_compare(self):
        data = np.arange(N * M, dtype=np.float32)

        def test_cranberry():
            a = StorageView.from_vec(data.tolist(), "cpu").reshape([N, M])
            out = a.neg().exp().log()  # identity-ish
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.log(np.exp(-data.reshape(N, M))).reshape(-1)
            return out

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_unary_random_shapes(self):
        rng = np.random.default_rng(1337)
        shapes = [(N * M,), (10, 10), (3, 4, 5)]

        for shape in shapes:
            data = rng.standard_normal(np.prod(shape)).astype(np.float32)
            # prepare variants for stability per op
            pos = (np.abs(data) + 1e-3).astype(np.float32)  # for sqrt/log
            clip = np.clip(data, -10, 10)  # for exp

            # neg
            a = StorageView.from_vec(data.tolist(), "cpu")
            if len(shape) > 1:
                a = a.reshape(list(shape))
            out = np.array(a.neg().to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, -data.reshape(-1), rtol, atol)

            # sqrt
            a = StorageView.from_vec(pos.tolist(), "cpu")
            if len(shape) > 1:
                a = a.reshape(list(shape))
            out = np.array(a.sqrt().to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, np.sqrt(pos).reshape(-1), rtol, atol)

            # exp
            a = StorageView.from_vec(clip.tolist(), "cpu")
            if len(shape) > 1:
                a = a.reshape(list(shape))
            out = np.array(a.exp().to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, np.exp(clip).reshape(-1), rtol, atol)

            # log
            a = StorageView.from_vec(pos.tolist(), "cpu")
            if len(shape) > 1:
                a = a.reshape(list(shape))
            out = np.array(a.log().to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, np.log(pos).reshape(-1), rtol, atol)

    def test_binary_random_shapes(self):
        rng = np.random.default_rng(2024)
        shapes = [(N * M,), (17, 11), (2, 3, 5)]

        for shape in shapes:
            a_np = rng.standard_normal(np.prod(shape)).astype(np.float32)
            b_np = rng.standard_normal(np.prod(shape)).astype(np.float32)
            b_np_div = b_np.copy()
            b_np_div[np.isclose(b_np_div, 0.0)] = 1.0

            def make_sv(arr):
                sv = StorageView.from_vec(arr.reshape(-1).tolist(), "cpu")
                return sv.reshape(list(shape)) if len(shape) > 1 else sv

            # add
            a = make_sv(a_np)
            b = make_sv(b_np)
            out = np.array(a.add(b).to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, (a_np + b_np).reshape(-1), rtol, atol)

            # sub
            a = make_sv(a_np)
            b = make_sv(b_np)
            out = np.array(a.sub(b).to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, (a_np - b_np).reshape(-1), rtol, atol)

            # mul
            a = make_sv(a_np)
            b = make_sv(b_np)
            out = np.array(a.mul(b).to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, (a_np * b_np).reshape(-1), rtol, atol)

            # div
            a = make_sv(a_np)
            b = make_sv(b_np_div)
            out = np.array(a.div(b).to_vec(), dtype=np.float32)
            np.testing.assert_allclose(out, (a_np / b_np_div).reshape(-1), rtol, atol)

    def test_slice_then_unary(self):
        base = np.linspace(-3, 3, 101, dtype=np.float32)
        v = StorageView.from_vec(base.tolist(), "cpu")
        s = v.slice(5, 77)
        out = np.array(s.exp().to_vec(), dtype=np.float32)
        np.testing.assert_allclose(out, np.exp(base[5:82]), rtol, atol)

    def test_slice_then_binary(self):
        a_base = np.linspace(-2, 2, 111, dtype=np.float32)
        b_base = np.linspace(3, -3, 111, dtype=np.float32)
        a = StorageView.from_vec(a_base.tolist(), "cpu").slice(7, 55)
        b = StorageView.from_vec(b_base.tolist(), "cpu").slice(9, 55)
        out = np.array(a.mul(b).to_vec(), dtype=np.float32)
        np.testing.assert_allclose(out, (a_base[7:62] * b_base[9:64]), rtol, atol)

    def test_numpy_input(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = StorageView.from_vec(arr, "cpu")
        self.assertEqual(v.to_vec(), [1.0, 2.0, 3.0])

    def test_unary_simd_remainder(self):
        # length intentionally not divisible by SIMD width (64)
        data = np.arange(65, dtype=np.float32)
        v = StorageView.from_vec(data.tolist(), "cpu")
        out = np.array(v.neg().to_vec(), dtype=np.float32)
        np.testing.assert_allclose(out, -data, rtol, atol)

    def test_binary_simd_remainder(self):
        a = np.arange(129, dtype=np.float32)
        b = np.arange(129, dtype=np.float32) * -0.25
        va = StorageView.from_vec(a.tolist(), "cpu")
        vb = StorageView.from_vec(b.tolist(), "cpu")
        out = np.array(va.add(vb).to_vec(), dtype=np.float32)
        np.testing.assert_allclose(out, a + b, rtol, atol)

    def test_binary_2d(self):
        a_np = A_np.reshape(N, M)
        b_np = B_np.reshape(N, M)

        def test_cranberry():
            a = StorageView.from_vec(a_np.reshape(-1).tolist(), "cpu").reshape([N, M])
            b = StorageView.from_vec(b_np.reshape(-1).tolist(), "cpu").reshape([N, M])
            out = a.mul(b).add(a).sub(b).div(a)
            return np.array(out.to_vec(), dtype=np.float32)

        def test_numpy():
            out = (a_np * b_np + a_np - b_np) / a_np
            return out.reshape(-1)

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    # movement semantics and error paths

    def test_slice_1d(self):
        base = np.arange(100, dtype=np.float32)
        off, size = 7, 23

        def test_cranberry():
            v = StorageView.from_vec(base.tolist(), "cpu")
            s = v.slice(off, size)
            return np.array(s.to_vec(), dtype=np.float32)

        def test_numpy():
            return base[off : off + size]

        np.testing.assert_allclose(test_cranberry(), test_numpy(), rtol, atol)

    def test_reshape_contiguous_to_vec(self):
        v = StorageView.from_vec([1.0, 2.0, 3.0, 4.0], "cpu")
        r = v.reshape([2, 2])
        self.assertEqual(r.to_vec(), [1.0, 2.0, 3.0, 4.0])

    def test_expand_non_contiguous_to_vec_raises(self):
        v = StorageView.from_vec([1.0, 2.0, 3.0], "cpu").reshape([1, 3])
        e = v.expand([2, 3])
        with self.assertRaises(ValueError):
            _ = e.to_vec()

    def test_permute_non_contiguous_to_vec_raises(self):
        v = StorageView.from_vec([1.0, 2.0, 3.0, 4.0], "cpu").reshape([2, 2])
        p = v.permute([1, 0])
        with self.assertRaises(ValueError):
            _ = p.to_vec()

    def test_unary_on_non_contiguous_raises(self):
        v = StorageView.from_vec([1.0, 4.0, 9.0, 16.0], "cpu").reshape([2, 2]).permute([1, 0])
        with self.assertRaises(RuntimeError):
            _ = v.neg()
        with self.assertRaises(RuntimeError):
            _ = v.sqrt()
        with self.assertRaises(RuntimeError):
            _ = v.exp()
        with self.assertRaises(RuntimeError):
            _ = v.log()

    def test_binary_on_non_contiguous_raises(self):
        a = StorageView.from_vec([1.0, 2.0, 3.0, 4.0], "cpu").reshape([2, 2]).permute([1, 0])
        b = StorageView.from_vec([1.0, 2.0, 3.0, 4.0], "cpu").reshape([2, 2])
        with self.assertRaises(RuntimeError):
            _ = a.add(b)
        with self.assertRaises(RuntimeError):
            _ = a.sub(b)
        with self.assertRaises(RuntimeError):
            _ = a.mul(b)
        with self.assertRaises(RuntimeError):
            _ = a.div(b)

    def test_binary_shape_mismatch_raises(self):
        a = StorageView.from_vec([1.0, 2.0, 3.0], "cpu")
        b = StorageView.from_vec([4.0, 5.0], "cpu")
        with self.assertRaises(RuntimeError):
            _ = a.add(b)

    def test_add_device_mismatch_raises(self):
        a = StorageView.from_vec([1.0, 2.0], "cpu")
        b = StorageView.from_vec([1.0, 2.0], "metal")
        with self.assertRaises(ValueError):
            _ = a.add(b)

    def test_ops_not_implemented_other_device(self):
        v = StorageView.from_vec([1.0, 2.0], "metal")
        with self.assertRaises(NotImplementedError):
            _ = v.neg()
        with self.assertRaises(NotImplementedError):
            _ = v.exp()


if __name__ == "__main__":
    unittest.main()

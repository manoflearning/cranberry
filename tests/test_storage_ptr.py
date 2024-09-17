import unittest
import numpy as np
from cranberry import StoragePtr

np.random.seed(1337)

MAX_VEC_SIZE = 4000
DEFAULT_TEST_COUNT = 10
DEVICE = "cpu"

rtol, atol = 1e-4, 1e-4


class TestStoragePtr(unittest.TestCase):
    def test_storage_ptr_neg(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.zeros(vec_size, dtype=np.float32)
            y[idx_2 : idx_2 + size] = -x[idx_1 : idx_1 + size]

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.neg(a, b, idx_1, idx_2, size)

            np.testing.assert_allclose(y, b.to_vec(), rtol, atol)

    def test_storage_ptr_sqrt(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.zeros(vec_size, dtype=np.float32)
            y[idx_2 : idx_2 + size] = np.sqrt(x[idx_1 : idx_1 + size])

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.sqrt(a, b, idx_1, idx_2, size)

            np.testing.assert_allclose(y, b.to_vec(), rtol, atol)

    def test_storage_ptr_exp(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.zeros(vec_size, dtype=np.float32)
            y[idx_2 : idx_2 + size] = np.exp(x[idx_1 : idx_1 + size])

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.exp(a, b, idx_1, idx_2, size)

            np.testing.assert_allclose(y, b.to_vec(), rtol, atol)

    def test_storage_ptr_log(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.zeros(vec_size, dtype=np.float32)
            y[idx_2 : idx_2 + size] = np.log(x[idx_1 : idx_1 + size])

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.log(a, b, idx_1, idx_2, size)

            np.testing.assert_allclose(y, b.to_vec(), rtol, atol)

    def test_storage_ptr_add(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            idx_3 = np.random.randint(0, vec_size)

            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2, vec_size - idx_3) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.random.rand(vec_size).astype(np.float32)
            z = np.zeros(vec_size, dtype=np.float32)
            z[idx_3 : idx_3 + size] = x[idx_1 : idx_1 + size] + y[idx_2 : idx_2 + size]

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.from_vec(y, DEVICE)
            c = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.add(a, b, c, idx_1, idx_2, idx_3, size)

            np.testing.assert_allclose(z, c.to_vec(), rtol, atol)

    def test_storage_ptr_sub(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            idx_3 = np.random.randint(0, vec_size)

            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2, vec_size - idx_3) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.random.rand(vec_size).astype(np.float32)
            z = np.zeros(vec_size, dtype=np.float32)
            z[idx_3 : idx_3 + size] = x[idx_1 : idx_1 + size] - y[idx_2 : idx_2 + size]

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.from_vec(y, DEVICE)
            c = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.sub(a, b, c, idx_1, idx_2, idx_3, size)

            np.testing.assert_allclose(z, c.to_vec(), rtol, atol)

    def test_storage_ptr_mul(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            idx_3 = np.random.randint(0, vec_size)

            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2, vec_size - idx_3) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.random.rand(vec_size).astype(np.float32)
            z = np.zeros(vec_size, dtype=np.float32)
            z[idx_3 : idx_3 + size] = x[idx_1 : idx_1 + size] * y[idx_2 : idx_2 + size]

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.from_vec(y, DEVICE)
            c = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.mul(a, b, c, idx_1, idx_2, idx_3, size)

    def test_storage_ptr_div(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            idx_3 = np.random.randint(0, vec_size)

            size = np.random.randint(1, min(vec_size - idx_1, vec_size - idx_2, vec_size - idx_3) + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.random.rand(vec_size).astype(np.float32)
            z = np.zeros(vec_size, dtype=np.float32)
            z[idx_3 : idx_3 + size] = x[idx_1 : idx_1 + size] / y[idx_2 : idx_2 + size]

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.from_vec(y, DEVICE)
            c = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.div(a, b, c, idx_1, idx_2, idx_3, size)

            np.testing.assert_allclose(z, c.to_vec(), rtol, atol)

    def test_storage_ptr_sum(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            size = np.random.randint(1, vec_size - idx_1 + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.zeros(vec_size, dtype=np.float32)
            y[idx_2] = np.sum(x[idx_1 : idx_1 + size])

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.sum(a, b, idx_1, idx_2, size)

            np.testing.assert_allclose(y, b.to_vec(), rtol, atol)

    def test_storage_ptr_max(self):
        for _ in range(DEFAULT_TEST_COUNT):
            vec_size = np.random.randint(1, MAX_VEC_SIZE + 1)

            idx_1 = np.random.randint(0, vec_size)
            idx_2 = np.random.randint(0, vec_size)
            size = np.random.randint(1, vec_size - idx_1 + 1)

            x = np.random.rand(vec_size).astype(np.float32)
            y = np.zeros(vec_size, dtype=np.float32)
            y[idx_2] = np.max(x[idx_1 : idx_1 + size])

            a = StoragePtr.from_vec(x, DEVICE)
            b = StoragePtr.full(0.0, vec_size, DEVICE)
            StoragePtr.max(a, b, idx_1, idx_2, size)

            np.testing.assert_allclose(y, b.to_vec(), rtol, atol)


if __name__ == "__main__":
    unittest.main()

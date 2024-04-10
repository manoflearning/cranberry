import cranberry as cr
import numpy as np
import unittest

class TestCranberry(unittest.TestCase):
    def test_zerodim_initialization(self):
        a = cr.Tensor(55)
        b = cr.Tensor(156.342)

        # TODO: if data is scalar, a.shape == () not == []
        self.assertEqual(a.shape, [])
        self.assertEqual(b.shape, [])

    def test_plus_equals(self):
        a = cr.Tensor.uniform([10, 10], -100, 100)
        b = cr.Tensor.uniform([10, 10], -100, 100)

        c = a + b
        val1 = np.array(c.data)
        a += b
        val2 = np.array(a.data)

        np.testing.assert_allclose(val1, val2)

    def test_matmul_equals(self):
        a = np.random.rand(2, 2).astype(np.float32)
        b = np.random.rand(2, 2).astype(np.float32)
        c = a @ b

        p = cr.Tensor(a)
        q = cr.Tensor(b)
        r = p @ q

        
if __name__ == '__main__':
    unittest.main()
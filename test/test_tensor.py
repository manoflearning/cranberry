import cranberry as cb
import numpy as np
import unittest

class TestCranberry(unittest.TestCase):
    def test_zerodim_initialization(self):
        a = cb.Tensor(55)
        b = cb.Tensor(156.342)

        # TODO: if data is scalar, a.shape == () not == []
        self.assertEqual(a.shape, [])
        self.assertEqual(b.shape, [])

    def test_plus_equals(self):
        a = cb.Tensor.uniform([10, 10], -100, 100)
        b = cb.Tensor.uniform([10, 10], -100, 100)

        c = a + b
        val1 = np.array(c.data)
        a += b
        val2 = np.array(a.data)

        np.testing.assert_allclose(val1, val2)        

if __name__ == '__main__':
    unittest.main()
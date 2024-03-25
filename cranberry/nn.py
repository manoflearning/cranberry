import math
from cranberry import Tensor

class Linear:

    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.weight = Tensor.kaiming_uniform([out_features, in_features], math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = Tensor.uniform([out_features], -bound, bound) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.linear(self.weight, self.bias)
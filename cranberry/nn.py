import math
from cranberry import Tensor

class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.weight = Tensor.kaiming_uniform(shape=[out_features, in_features], a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = Tensor.uniform(shape=[out_features, in_features], low=-bound, high=bound) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.linear(weight=self.weight, bias=self.bias)

class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()
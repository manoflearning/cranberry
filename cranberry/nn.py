import math
import cranberry as cb

class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.weight = cb.Tensor.kaiming_uniform(shape=[in_features, out_features], a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = cb.Tensor.uniform(shape=[out_features], low=-bound, high=bound) if bias else None

    def __call__(self, x: cb.Tensor) -> cb.Tensor:
        return x.linear(weight=self.weight, bias=self.bias)

    def __str__(self):
        return f"Linear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None})"        

class ReLU:
    def __call__(self, x: cb.Tensor) -> cb.Tensor:
        return x.relu()
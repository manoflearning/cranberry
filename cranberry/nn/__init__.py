from abc import abstractmethod
from cranberry import Tensor

import math
from typing import List


class Module:
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def parameters(self) -> List[Tensor]:
        pass


class ReLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> List[Tensor]:
        return []


class Linear(Module):
    # https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/__init__.py#L72-L80
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = (
            Tensor.uniform(out_features, low=-bound, high=bound) if bias else None
        )

    def __call__(self, x: Tensor) -> Tensor:
        return x.linear(weight=self.weight.transpose(0, 1), bias=self.bias)

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        out = []
        for layer in self.layers:
            out += layer.parameters()
        return out

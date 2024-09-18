from __future__ import annotations
from typing import Tuple


class Shape:
  def __init__(self, shape: Tuple):
    self.dims = shape
    assert all(isinstance(i, int) and i > 0 for i in self.dims)

  def __eq__(self, other) -> bool:
    assert isinstance(other, Shape)
    return self.dims == other.dims

  def __repr__(self) -> str:
    return f"{self.dims}"

  def __iter__(self):
    return iter(self.dims)

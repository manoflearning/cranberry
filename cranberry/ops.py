from enum import Enum, auto
from typing import Union


class UnaryOps(Enum):
  NEG = auto()
  SQRT = auto()
  RELU = auto()
  EXP = auto()
  LOG = auto()

  def __repr__(self):
    return f"{self.name.lower()}"


class BinaryOps(Enum):
  ADD = auto()
  SUB = auto()
  MUL = auto()
  DIV = auto()

  def __repr__(self):
    return f"{self.name.lower()}"


class ReduceOps(Enum):
  SUM = auto()
  MAX = auto()

  def __repr__(self):
    return f"{self.name.lower()}"


class MovementOps(Enum):
  RESHAPE = auto()
  EXPAND = auto()
  PERMUTE = auto()

  def __repr__(self):
    return f"{self.name.lower()}"


Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, None]

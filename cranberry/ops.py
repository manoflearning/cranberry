from enum import Enum, auto
from typing import Union


class UnaryOps(Enum):
  NEG = auto(); SQRT = auto(); EXP = auto(); LOG = auto() # noqa: E702
  def __repr__(self): return f"{self.name.lower()}"

class BinaryOps(Enum):
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto() # noqa: E702
  CMPLT = auto()
  def __repr__(self): return f"{self.name.lower()}"

class ReduceOps(Enum):
  SUM = auto(); MAX = auto() # noqa: E702
  def __repr__(self): return f"{self.name.lower()}"

class MetaOps(Enum):
  RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto() # noqa: E702
  def __repr__(self): return f"{self.name.lower()}"

Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, None]

from enum import Enum, auto
from typing import Union

class UnaryOps(Enum): NEG = auto(); SQRT = auto(); RELU = auto(); EXP = auto(); LOG = auto()
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
class ReduceOps(Enum): SUM = auto()
class MovementOps(Enum): RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto()

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, None]
from typing import Union
import numpy as np

class StoragePtr:
  @staticmethod
  def full(value: float, size: int, device: str) -> StoragePtr: ...
  @staticmethod
  def from_vec(vec: Union[list[float], np.ndarray], device: str) -> StoragePtr: ...
  @staticmethod
  def neg(a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int): ...
  @staticmethod
  def sqrt(a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int): ...
  @staticmethod
  def exp(a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int): ...
  @staticmethod
  def log(a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int): ...
  @staticmethod
  def add(a: StoragePtr, b: StoragePtr, c: StoragePtr, idx_a: int, idx_b: int, idx_c: int, size: int): ...
  @staticmethod
  def sub(a: StoragePtr, b: StoragePtr, c: StoragePtr, idx_a: int, idx_b: int, idx_c: int, size: int): ...
  @staticmethod
  def mul(a: StoragePtr, b: StoragePtr, c: StoragePtr, idx_a: int, idx_b: int, idx_c: int, size: int): ...
  @staticmethod
  def div(a: StoragePtr, b: StoragePtr, c: StoragePtr, idx_a: int, idx_b: int, idx_c: int, size: int): ...
  @staticmethod
  def sum(a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int): ...
  @staticmethod
  def max(a: StoragePtr, b: StoragePtr, idx_a: int, idx_b: int, size: int): ...
  def to_vec(self) -> list[float]: ...

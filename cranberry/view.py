from __future__ import annotations
from dataclasses import dataclass
import functools
import itertools
from math import prod
import operator
from typing import Optional, Tuple

@functools.lru_cache(maxsize=None)
def canonicalize_strides(shape: Tuple[int], strides: Tuple[int]) -> Tuple[int]:
    return tuple(0 if s == 1 else st for s, st in zip(shape, strides))

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape: Tuple[int]) -> Tuple[int]:
    if not shape: return ()
    strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))
    return canonicalize_strides(shape, strides)

@dataclass(frozen=True)
class View:
    shape: Tuple[int]
    strides: Tuple[int]
    # TODO: add offset and mask
    # offset: int
    # mask: Optional[Tuple[Tuple[int, int]]]
    contiguous: bool

    @functools.lru_cache(maxsize=None)
    def size(self) -> int: return prod(self.shape)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def create(shape: Tuple[int], strides: Optional[Tuple[int]] = None):
        strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)
        contiguous = strides == strides_for_shape(shape)
        return View(shape, strides, contiguous)
    
    @functools.lru_cache(maxsize=None)
    def reshape(self, new_shape: Tuple[int]) -> Optional[View]:
        if new_shape == self.shape: return self

        assert all(s >= 0 for s in new_shape), f"shape can't contain negative numbers {new_shape}"
        assert prod(self.shape) == prod(new_shape), f"cannot reshape {self.shape} to {new_shape}"
        if 0 in self.shape:
            assert 0 in new_shape, f"cannot reshape 0 size shape {self.shape} to {new_shape}"
            return View.create(new_shape)
        
        if self.contiguous: return View.create(new_shape)
        else: assert False, "non-contiguous reshape not implemented"

    @functools.lru_cache(maxsize=None)
    def expand(self, new_shape: Tuple[int]) -> View:
        assert len(new_shape) == len(self.shape), f"cannot expand {self.shape} to {new_shape}"
        assert all((ns == s or ns == 1) or (s == 1 and st == 0) for ns, s, st in zip(new_shape, self.shape, self.strides)), f"invalid expansion {new_shape} from {self.shape=} {self.strides=}"
        return View.create(new_shape, self.strides)

    @functools.lru_cache(maxsize=None)
    def permute(self, dims: Tuple[int]) -> View:
        assert set(dims) == set(range(len(dims))), f"invalid permutation {dims}"
        return View.create(tuple(self.shape[d] for d in dims), tuple(self.strides[d] for d in dims))
    
    # TODO: flip, shrink, pad

    @functools.lru_cache(maxsize=None)
    def indices(self) -> Tuple[int, int]:
        # have no idea whether this is the right way or not
        if self.contiguous: return (0, self.size())

        def _indices(shape: Tuple[int], strides: Tuple[int], start: int):
            if not shape: return [start, 1]
            out = []
            for i in range(shape[0]):
                out += _indices(shape[1:], strides[1:], start + i % strides[0] * strides[0])
            return out
        out = _indices(self.shape, self.strides, 0)

        new_out = [out[0]]
        for i in range(1, len(out)):
            assert out[i-1] <= out[i], f"invalid indices {out}"
            if out[i-1] + 1 == out[i]: new_out[-1][1] += 1
            else: new_out.append(out[i])
        return tuple(new_out)
import gzip
import hashlib
from os import getenv
import os
import pathlib
import tempfile
from typing import Optional, Union
import urllib.request
import sys

from typing import TYPE_CHECKING, Any

from cranberry import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
  pass

try:  # pragma: no cover - optional dependency
  import numpy as np  # type: ignore[assignment]
except ImportError:  # pragma: no cover
  np = None  # type: ignore[assignment]


def _require_numpy() -> Any:
  if np is None:  # pragma: no cover - optional path
    raise RuntimeError("NumPy is required for dataset utilities. Install 'cranberry[numpy]' to enable them.")
  return np


# Platform detection without psutil
OSX = sys.platform == "darwin"

# Optional tqdm progress bar
try:
  from tqdm import tqdm as _tqdm  # type: ignore

  tqdm = _tqdm  # noqa: N802 - keep name compatibility
except Exception:  # pragma: no cover - fallback path

  class _NoopTqdm:
    def __init__(self, *a, **kw):
      pass

    def update(self, n: int):
      pass

  # Keep the same callable interface as tqdm
  tqdm = _NoopTqdm  # noqa: N802 - keep name compatibility

_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))


def fetch(
  url: str,
  name: Optional[Union[pathlib.Path, str]] = None,
  allow_caching=not getenv("DISABLE_HTTP_CACHE"),
) -> pathlib.Path:
  if url.startswith(("/", ".")):
    return pathlib.Path(url)
  fp = (
    pathlib.Path(name)
    if name is not None and (isinstance(name, pathlib.Path) or "/" in name)
    else pathlib.Path(_cache_dir) / "cranberry" / "downloads" / (name or hashlib.md5(url.encode("utf-8")).hexdigest())
  )
  if not fp.is_file() or not allow_caching:
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get("content-length", 0))
      progress_bar = tqdm(total=total_length, unit="B", unit_scale=True, desc=url)
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384):
          progress_bar.update(f.write(chunk))
        f.close()
        if (file_size := os.stat(f.name).st_size) < total_length:
          raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp


def _fetch_mnist(file, offset):
  np_mod = _require_numpy()
  return Tensor(
    np_mod.frombuffer(
      gzip.open(fetch("https://storage.googleapis.com/cvdf-datasets/mnist/" + file)).read()[offset:],
      dtype=np_mod.uint8,
    )
  )


def mnist():
  return (
    _fetch_mnist("train-images-idx3-ubyte.gz", 0x10).reshape(60000, 1, 28, 28),
    _fetch_mnist("train-labels-idx1-ubyte.gz", 8),
    _fetch_mnist("t10k-images-idx3-ubyte.gz", 0x10).reshape(10000, 1, 28, 28),
    _fetch_mnist("t10k-labels-idx1-ubyte.gz", 8),
  )

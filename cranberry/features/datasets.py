import gzip
import hashlib
from os import getenv
import os
import pathlib
import tempfile
from typing import Optional, Union
import urllib.request

import numpy as np
from psutil import OSX
from tqdm import tqdm
from cranberry import Tensor

_cache_dir: str = getenv(
    "XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache")
)


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
        else pathlib.Path(_cache_dir)
        / "cranberry"
        / "downloads"
        / (name if name else hashlib.md5(url.encode("utf-8")).hexdigest())
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
                    raise RuntimeError(
                        f"fetch size incomplete, {file_size} < {total_length}"
                    )
                pathlib.Path(f.name).rename(fp)
    return fp


def _fetch_mnist(file, offset):
    return Tensor(
        np.frombuffer(
            gzip.open(
                fetch("https://storage.googleapis.com/cvdf-datasets/mnist/" + file)
            ).read()[offset:],
            dtype=np.uint8,
        )
    )


def mnist():
    return (
        _fetch_mnist("train-images-idx3-ubyte.gz", 0x10).reshape(60000, 1, 28, 28),
        _fetch_mnist("train-labels-idx1-ubyte.gz", 8),
        _fetch_mnist("t10k-images-idx3-ubyte.gz", 0x10).reshape(10000, 1, 28, 28),
        _fetch_mnist("t10k-labels-idx1-ubyte.gz", 8),
    )

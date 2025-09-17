<h1 align="center">Cranberry</h1>

<p align="center">
A small deep learning framework in Rust and Python
</p>

<p align="center">
  <a href="https://github.com/manoflearning/cranberry/actions/workflows/main.yaml">
    <img src="https://github.com/manoflearning/cranberry/actions/workflows/main.yaml/badge.svg" alt="Unit Tests" height="22">
  </a>
  <a href="https://github.com/manoflearning/cranberry/stargazers">
    <img src="https://img.shields.io/github/stars/manoflearning/cranberry" alt="GitHub Stars" height="22">
  </a>
</p>

## Overview

Cranberry is an educational project exploring how a tensor library, automatic differentiation, and a Rust-backed storage layer fit together. The Python front-end intentionally stays simple while the Rust extension supplies fast contiguous kernels and view manipulation utilities. Everything targets 32-bit floating point tensors and now offers optional CUDA acceleration for pointwise kernels alongside the CPU path.

## Highlights

- Python-first `Tensor` API backed by the `StorageView` PyO3 module.
- Reverse-mode autograd with topological traversal, supporting gradient tracking through broadcasting and reshape/expand/permute transforms.
- Contiguous CPU kernels for unary/binary ops plus sum/max reductions, with broadcasting handled in Python.
- Optional CUDA backend for contiguous unary and binary operations when an NVIDIA GPU and toolkit are available.
- Basic neural-network building blocks (`nn.Linear`, `nn.ReLU`, `nn.Sequential`) and stochastic gradient descent in `optim.SGD`.
- Visualization helpers for autograd graphs (`cranberry.features.visualize`) and an MNIST downloader with caching (`cranberry.features.datasets`).

## Current Status

**Tensor & Autograd**
- `Tensor` stores data in a Rust `StorageView` and exposes `.requires_grad`, `.grad`, `.backward()`.
- `backward()` runs on scalar outputs; higher-rank tensors need manual reduction to a scalar loss.
- Broadcasting, chaining, and reshape/expand/permute operations participate in autograd; gradients are accumulated in contiguous buffers.
- Optional NumPy interoperability via `Tensor.numpy()` and `Tensor.grad` when the `numpy` extra is installed.

**Operations**
- Unary: `neg`, `sqrt`, `relu`, `exp`, `log` plus derived helpers (`sigmoid`, `tanh`, `gelu`).
- Binary: `add`, `sub`, `mul`, `div` with broadcasting semantics.
- Reductions: `sum`, `max`, `mean` (derived from `sum`), `softmax`, `log_softmax`.
- Movement: `reshape`, `expand`, `permute`, `flatten`, `transpose`, `view`.
- Other helpers: 1D/2D `matmul`, `linear`, and `sparse_categorical_crossentropy`.

**Random & Initialization**
- Deterministic RNG via `Tensor.manual_seed`.
- Initializers: `Tensor.randn`, `Tensor.uniform`, `Tensor.kaiming_uniform`.

**Neural Network Utilities**
- Modules: `nn.Linear`, `nn.ReLU`, `nn.Sequential`.
- Optimizer: `optim.SGD` with in-place parameter updates and `zero_grad()` convenience.

**Data & Visualization**
- `features.datasets.fetch` caches downloads under `$XDG_CACHE_HOME` (or `~/Library/Caches` / `~/.cache`) and falls back gracefully when caching is disabled.
- `features.datasets.mnist()` returns tensors shaped `(N, 1, 28, 28)` for images and `(N,)` for labels.
- `features.visualize.plot_graph` renders autograd graphs via Graphviz when the `viz` extra is installed.

**Rust Extension**
- `StorageView` exposes contiguous tensor storage, reshaping, expanding, permuting, and random fills.
- CPU backend implements SIMD-accelerated unary/binary kernels and reduction routines.
- CUDA backend (via `cudarc`) mirrors the contiguous unary/binary kernels when a CUDA device is detected.
- Views currently support up to rank-4 tensors; non-contiguous reshape/permute paths are under construction.

## Limitations & Work in Progress

- CUDA backend currently covers only contiguous unary/binary kernels; Metal remains stubbed out.
- Autograd requires scalar losses and does not yet handle slicing/indexing/in-place mutations.
- Views must be contiguous for most kernels; slicing and advanced indexing are not implemented.
- Only `float32` tensors are supported; dtype promotion and mixed precision are future work.
- Batched matrix multiplication, convolutions, and additional operators are not yet implemented.
- `optim.SGD` is the only optimizer; schedulers, Adam, and other training utilities are on the roadmap.

## Installation

Requirements: Python 3.11 and a Rust toolchain (see `rust-toolchain.toml`).

Using uv:

```bash
git clone https://github.com/manoflearning/cranberry.git
cd cranberry

uv python install 3.11
uv sync --dev

# Build the native extension in editable mode
uv run maturin develop
```

Using pip (requires Rust for the build step):

```bash
git clone https://github.com/manoflearning/cranberry.git
cd cranberry
pip install -e .[numpy]
```

Optional extras:

- `pip install -e .[viz]` for autograd visualization (requires Graphviz system binary).
- `pip install -e .[datasets]` for download progress via `tqdm`.
- `pip install -e .[all]` to include every extra.

## CUDA backend

- Requires an NVIDIA driver and CUDA toolkit (NVRTC must be discoverable via `CUDA_HOME`, `CUDA_PATH`, or the default `/usr/local/cuda`).
- No separate `nvcc` build step is needed—the crate compiles its kernels at runtime using NVRTC.
- At runtime pass `device="cuda"` when creating tensors/storage; contiguous unary and binary ops will execute on the GPU and fall back with a runtime error if no device is present.

## Quickstart

```python
import numpy as np
from cranberry import nn, optim, Tensor
from cranberry.features import datasets

# Download and reshape MNIST
X_train, Y_train, X_test, Y_test = datasets.mnist()
X_train, X_test = X_train.flatten(1), X_test.flatten(1)

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10),
)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
batch_size, epochs = 128, 1
N = X_train.shape[0]

X_train_np, Y_train_np = X_train.numpy(), Y_train.numpy()

for epoch in range(epochs):
  perm = np.random.permutation(N)
  for start in range(0, N, batch_size):
    end = min(start + batch_size, N)
    inputs = Tensor(X_train_np[perm[start:end]], requires_grad=False)
    labels = Tensor(Y_train_np[perm[start:end]], requires_grad=False)

    optimizer.zero_grad()
    logits = model(inputs)
    loss = logits.sparse_categorical_crossentropy(labels)
    loss.backward()
    optimizer.step()
```

More examples live in `examples/`.

## Development

- `uv run pytest` runs the Python test suite (requires the Rust extension to be built).
- `cargo test` exercises the Rust core components.

## License

MIT License (see `LICENSE`).

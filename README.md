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

Cranberry is an educational project that provides basic automatic differentiation and tensor operations. The high‑level API is implemented in Python for clarity, and a Rust extension provides vectorized kernels for a low‑level storage API (`StoragePtr`). Currently only CPU is supported.

## Status (Sep 2025)

- Tensor and autograd: forward/backward works for common cases; `backward()` is limited to scalars.
- Ops
  - Unary: neg, sqrt, relu, exp, log, sigmoid, tanh, gelu
  - Binary: add, sub, mul, div (with broadcasting)
  - Reductions: sum, max, mean, softmax, log_softmax
  - Movement: reshape, expand, permute, flatten, transpose
  - Linear algebra: matmul (no batched matmul yet)
- Simple NN modules: `nn.Linear`, `nn.ReLU`, `nn.Sequential`
- Optimizer: `optim.SGD` (others like Adam are not implemented)
- Data/visualization: MNIST loader; computation graph visualization utility
- Rust extension: `StoragePtr` exposed via PyO3; CPU backend implements unary/binary/reduce kernels

Limitations
- GPU/Metal backends are stubs.
- Tensor slicing/indexing is not implemented.
- Batched matmul, additional optimizers, and more kernels are in progress.
- APIs and internals may change.

## Installation

Requirements: Python 3.11; Rust nightly toolchain for building the native extension.

Using uv
```bash
git clone https://github.com/manoflearning/cranberry.git
cd cranberry

# Set up Python and sync dependencies
uv python install 3.11
uv sync --dev

# (Optional) build the native extension
uv run maturin develop
```

Using pip (requires Rust toolchain)
```bash
git clone https://github.com/manoflearning/cranberry.git
cd cranberry
pip install -e .
```

Note: Depending on your platform, a local build may be required.

## Quickstart

Minimal example training MNIST with mini‑batches. Tensor slicing is not available yet, so batching uses NumPy views. See `examples/mnist.py` for a complete script.

```python
import numpy as np
from cranberry import nn, optim, Tensor
from cranberry.features.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
X_train, X_test = X_train.flatten(1), X_test.flatten(1)

X_train_np, Y_train_np = X_train.numpy(), Y_train.numpy()
X_test_np, Y_test_np = X_test.numpy(), Y_test.numpy()

model = nn.Sequential(
  nn.Linear(784, 128), nn.ReLU(),
  nn.Linear(128, 64), nn.ReLU(),
  nn.Linear(64, 10),
)

optimizer = optim.SGD(model.parameters(), lr=0.001)
batch_size, epochs = 128, 1

N = X_train_np.shape[0]
steps = (N + batch_size - 1) // batch_size
for epoch in range(epochs):
  perm = np.random.permutation(N)
  for step in range(steps):
    s, e = step * batch_size, min((step + 1) * batch_size, N)
    Xb = Tensor(X_train_np[perm[s:e]])
    Yb = Tensor(Y_train_np[perm[s:e]])

    optimizer.zero_grad()
    loss = model(Xb).sparse_categorical_crossentropy(Yb)
    loss.backward()
    optimizer.step()
```

More examples are available in the [examples](./examples) directory.

## Roadmap (high level)

- Remove NumPy entirely: make Cranberry's `Shape` and `View` abstractions and the Rust `StoragePtr` backend the primary tensor storage; complete end‑to‑end integration across ops, autograd, and NN modules.
- Slicing/views and batched matmul: first‑class slicing/indexing, views without copies, and batched linear algebra.
- More optimizers (e.g., Adam) and training utilities.
- GPU/Metal backends.
- Packaging, release automation, and documentation.

## License

MIT License (see `LICENSE`)

<h1 align="center">Cranberry</h1>

<p align="center">
A lightweight deep learning framework in Rust and Python
</p>

<p align="center">
    <a href="https://discord.gg/DKqZGpPJDV">
        <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/DKqZGpPJDV" alt="discord server">
    </a>
    <a href="https://github.com/manoflearning/cranberry/actions/workflows/main.yaml">
        <img src="https://github.com/manoflearning/cranberry/actions/workflows/main.yaml/badge.svg" alt="unit tests">
    </a>
    <a href="https://github.com/manoflearning/cranberry/stargazers">
        <img src="https://img.shields.io/github/stars/manoflearning/cranberry" alt="github repo stars">
    </a>
</p>

Cranberry is a deep learning framework designed for efficiency and simplicity. It combines the performance of Rust with the flexibility of Python to create a powerful yet concise tool for machine learning practitioners and researchers.

### Current Status

Cranberry is in its early stages of development but is already functional for basic deep learning tasks. We are actively working on expanding its capabilities to make it a robust and practical framework for a wide range of applications.

### Key Features
- Minimal set of kernels implemented in Rust
- Majority of the framework logic implemented in Python for flexibility and ease of development
- Extremely concise codebase, making it easy to understand and extend
- PyTorch-like API for familiar and intuitive usage
- Designed for real-world applications while maintaining simplicity

## Getting Started

### Installation

```bash
git clone https://github.com/manoflearning/cranberry.git
cd cranberry
pip install poetry
poetry install
```

### Usage and Examples

```python
from cranberry import nn, optim
from cranberry.features.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
X_train, X_test = X_train.flatten(1), X_test.flatten(1)

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

optimizer = optim.SGD(model.parameters(), lr=0.001)

for i in range(100):
    optimizer.zero_grad()
    # forward
    loss = model(X_train).sparse_categorical_crossentropy(Y_train)
    # backward
    loss.backward()
    # update
    optimizer.step()

    print(f"epoch {i}, loss {loss.item():.4f}")
```

For more examples, see the [examples](./examples) directory.

## Contributing and Community

We welcome contributions to help Cranberry grow into a fully-featured, production-ready deep learning framework.
Also, join our [Discord server](https://discord.gg/DKqZGpPJDV) to discuss development, get help, or just hang out with us.

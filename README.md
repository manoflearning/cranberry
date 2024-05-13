# Cranberry

<div align="center">
<img src="cranberry.png" alt="profile" style="width:40%; height:auto; padding:12px">
</div>

A one-person developing deep learning framework.
Cranberry provides a PyTorch-like API, making it easy to learn and use.
For now, it focuses on writing code that runs fast and stable on top of the CPU. 
Multiple accelerators, including NVIDIA GPU (CUDA), will be supported soon.

## Usage

```py
from cranberry import Tensor, nn, optim
from cranberry.features.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
X_train, X_test = X_train.flatten(1), X_test.flatten(1)

model = nn.Sequential(
    nn.Linear(784, 128), Tensor.relu,
    nn.Linear(128, 64), Tensor.relu,
    nn.Linear(64, 10)
)

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

See [examples/demo.ipynb](/examples/demo.ipynb) and [examples/mnist.py](/examples/mnist.py) for additional examples.
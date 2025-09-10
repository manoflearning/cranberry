import numpy as np
from cranberry import nn, optim, Tensor
from cranberry.features.datasets import mnist

# Train MNIST with mini-batches to avoid huge intermediates in matmul.

X_train, Y_train, X_test, Y_test = mnist()
X_train, X_test = X_train.flatten(1), X_test.flatten(1)

# Cache numpy views for batching (Tensor currently lacks slicing)
X_train_np, Y_train_np = X_train.numpy(), Y_train.numpy()
X_test_np, Y_test_np = X_test.numpy(), Y_test.numpy()

model = nn.Sequential(
  nn.Linear(784, 128), nn.ReLU(),
  nn.Linear(128, 64), nn.ReLU(),
  nn.Linear(64, 10),
)

optimizer = optim.SGD(model.parameters(), lr=0.001)  # TODO: Adam

batch_size = 128
epochs = 1

N = X_train_np.shape[0]
steps_per_epoch = (N + batch_size - 1) // batch_size

for epoch in range(epochs):
  perm = np.random.permutation(N)
  running_loss = 0.0
  for step in range(steps_per_epoch):
    s = step * batch_size
    e = min(s + batch_size, N)
    idx = perm[s:e]
    Xb = Tensor(X_train_np[idx])
    Yb = Tensor(Y_train_np[idx])

    optimizer.zero_grad()
    loss = model(Xb).sparse_categorical_crossentropy(Yb)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if (step + 1) % 50 == 0 or step + 1 == steps_per_epoch:
      avg = running_loss / (step + 1)
      print(f"epoch {epoch+1}/{epochs} step {step+1}/{steps_per_epoch} - loss {avg:.4f}")

# Simple test accuracy
def accuracy(X_np: np.ndarray, Y_np: np.ndarray) -> float:
  total, correct = 0, 0
  for s in range(0, X_np.shape[0], batch_size):
    e = min(s + batch_size, X_np.shape[0])
    logits = model(Tensor(X_np[s:e]))
    pred = np.argmax(logits.numpy(), axis=1)
    correct += (pred == Y_np[s:e].astype(np.int64)).sum()
    total += e - s
  return correct / total

test_acc = accuracy(X_test_np, Y_test_np)
print(f"test accuracy: {test_acc*100:.2f}%")

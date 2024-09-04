from cranberry import nn, optim
from cranberry.features.datasets import mnist

# TODO: implement a model with conv2d, maxpool2d, etc.
# https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

X_train, Y_train, X_test, Y_test = mnist()
X_train, X_test = X_train.flatten(1), X_test.flatten(1)

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

optimizer = optim.SGD(model.parameters(), lr=0.001)  # TODO: use Adam

for i in range(100):
    optimizer.zero_grad()
    # forward
    loss = model(X_train).sparse_categorical_crossentropy(Y_train)
    # backward
    loss.backward()
    # update
    optimizer.step()

    print(f"epoch {i}, loss {loss.item():.4f}")

# TODO: evaluate on test set

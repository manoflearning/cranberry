from cranberry import Tensor

class SGD:
    def __init__(self, params: list, lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params: 
            if isinstance(p, Tensor): p.zero_grad()

    def step(self):
        for p in self.params: p.step(self.lr)

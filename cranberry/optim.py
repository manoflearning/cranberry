from typing import Tuple, Union
from cranberry import Tensor

# TODO: implement more detailed SGD
class SGD:
    def __init__(self, params: list, lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params: p.zero_grad()

    def step(self):
        for p in self.params: p.step(self.lr)

# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# TODO: implement more detailed Adam
# class Adam:
#     def __init__(self, 
#                  params: list, # TODO: implement Parameter class
#                  lr: Union[float, Tensor] = 1e-3,
#                  betas: Tuple[float, float] = (0.9, 0.999), 
#                  eps: float = 1e-8, 
#                  weight_decay: float = 0, 
#                  amsgrad: bool = False):
#         self.params = params
#         self.lr = lr
#         self.betas = betas
#         self.eps = eps
#         self.weight_decay = weight_decay
#         self.amsgrad = amsgrad

#     def zero_grad(self):
#         for p in self.params: 
#             if isinstance(p, Tensor): p.zero_grad()

#     def step(self):


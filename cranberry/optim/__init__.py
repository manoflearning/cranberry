from cranberry import StorageView, Tensor
from typing import List


class SGD:
  def __init__(self, params: List[Tensor], lr: float):
    # if it's False, but being put into an optimizer, set it to True
    for x in params:
      if x._requires_grad is False:
        x._requires_grad = True
    self._params, self._lr = params, lr

  def zero_grad(self):
    for p in self._params:
      p.zero_grad()

  def step(self):
    for p in self._params:
      grad_storage = p.grad_storage()
      if grad_storage is None:
        continue
      grad_view = grad_storage.contiguous()
      shape_list = list(p.shape)
      scale = StorageView.full(float(self._lr), max(p.num_elements(), 1), p.device)
      scale = scale.reshape(shape_list) if shape_list else scale.reshape([])
      scaled_grad = grad_view.mul(scale)
      new_data = p.data_storage().contiguous().sub(scaled_grad)
      p.set_data_storage(new_data)

  @property
  def lr(self):
    return self._lr

  @lr.setter
  def lr(self, lr: float):
    self._lr = lr


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

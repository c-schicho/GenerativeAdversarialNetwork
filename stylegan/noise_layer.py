from torch import Tensor
from torch.nn import Module


class GaussianNoise(Module):

    def __init__(self):
        super(GaussianNoise, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass

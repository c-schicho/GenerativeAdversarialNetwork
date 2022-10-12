from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU


class MappingNetwork(Module):

    def __init__(self, n_layers: int = 8, n_units: int = 512):
        super(MappingNetwork, self).__init__()
        self.network = Sequential(
            *[[Linear(n_units, n_units), LeakyReLU(.2, inplace=True)] for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

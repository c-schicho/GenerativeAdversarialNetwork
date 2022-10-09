import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d, Sigmoid


class DCGANDiscriminator(Module):

    def __init__(self, in_dims: int):
        super(DCGANDiscriminator, self).__init__()
        self.discriminator = Sequential(
            Conv2d(3, in_dims, 4, 2, 1, bias=False),
            LeakyReLU(.2, inplace=True),
            Conv2d(in_dims, in_dims * 2, 4, 2, 1, bias=False),
            BatchNorm2d(in_dims * 2),
            LeakyReLU(.2, inplace=True),
            Conv2d(in_dims * 2, in_dims * 4, 4, 2, 1, bias=False),
            BatchNorm2d(in_dims * 4),
            LeakyReLU(.2, inplace=True),
            Conv2d(in_dims * 4, in_dims * 8, 4, 2, 1, bias=False),
            BatchNorm2d(in_dims * 8),
            LeakyReLU(.2, inplace=True),
            Conv2d(in_dims * 8, 1, 4, 1, 0, bias=False),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.discriminator(x)

    def save(self, path: str):
        torch.save(self.discriminator.state_dict(), path)

    def load(self, path: str):
        self.discriminator.load_state_dict(torch.load(path))

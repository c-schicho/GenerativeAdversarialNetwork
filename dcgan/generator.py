import torch
from torch import Tensor
from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, LeakyReLU, Tanh


class DCGANGenerator(Module):

    def __init__(self, in_dims: int, out_dims: int):
        super(DCGANGenerator, self).__init__()
        self.generator = Sequential(
            ConvTranspose2d(in_dims, out_dims * 8, 4, 1, 0, bias=False),
            BatchNorm2d(out_dims * 8),
            LeakyReLU(.1, inplace=True),
            ConvTranspose2d(out_dims * 8, out_dims * 4, 4, 2, 1, bias=False),
            BatchNorm2d(out_dims * 4),
            LeakyReLU(.1, inplace=True),
            ConvTranspose2d(out_dims * 4, out_dims * 2, 4, 2, 1, bias=False),
            BatchNorm2d(out_dims * 2),
            LeakyReLU(.1, inplace=True),
            ConvTranspose2d(out_dims * 2, out_dims, 4, 2, 1, bias=False),
            BatchNorm2d(out_dims),
            LeakyReLU(.1, inplace=True),
            ConvTranspose2d(out_dims, 3, 4, 2, 1, bias=False),
            Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)

    def save(self, path: str):
        torch.save(self.generator.state_dict(), path)

    def load(self, path: str):
        self.generator.load_state_dict(torch.load(path))

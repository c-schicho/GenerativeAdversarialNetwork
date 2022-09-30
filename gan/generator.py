import torch
from torch.nn import Module, Sequential, Linear, LeakyReLU, Tanh


class FNNGenerator(Module):

    def __init__(self, in_dims: int, out_dims: int):
        super(FNNGenerator, self).__init__()
        self.generator = Sequential(
            Linear(in_dims, 256),
            LeakyReLU(.2),
            Linear(256, 512),
            LeakyReLU(.2),
            Linear(512, 1024),
            LeakyReLU(.2),
            Linear(1024, out_dims),
            Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def save(self, path: str):
        torch.save(self.generator.state_dict(), path)

    def load(self, path: str):
        self.generator.load_state_dict(torch.load(path))

import torch
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid, Dropout


class FNNDiscriminator(Module):

    def __init__(self, in_dims: int):
        super(FNNDiscriminator, self).__init__()
        self.discriminator = Sequential(
            Linear(in_dims, 1024),
            LeakyReLU(.2),
            Dropout(.3),
            Linear(1024, 512),
            LeakyReLU(.2),
            Dropout(.3),
            Linear(512, 256),
            LeakyReLU(.2),
            Dropout(.3),
            Linear(256, 1),
            Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def save(self, path: str):
        torch.save(self.discriminator.state_dict(), path)

    def load(self, path: str):
        self.discriminator.load_state_dict(torch.load(path))

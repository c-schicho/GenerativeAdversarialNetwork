from torch import Tensor
from torch.nn import Module


class StyleGANGenerator(Module):

    def __init__(self):
        super(StyleGANGenerator, self).__init__()

    def forward(self) -> Tensor:
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

import torch
from torch import Tensor
from torch.nn import Module


class ADAIn(Module):

    def __init__(self, idx_h_dim: int = 2, idx_w_dim: int = 3):
        super(ADAIn, self).__init__()
        self.h_dim = idx_h_dim
        self.w_dim = idx_w_dim

    def get_n_pixels(self, x: Tensor) -> int:
        return x.size(dim=self.h_dim) * x.size(dim=self.w_dim)

    def mu(self, x: Tensor) -> Tensor:
        n = torch.sum(x, (2, 3))
        d = self.get_n_pixels(x)
        return n / d

    def sigma(self, x: Tensor) -> Tensor:
        n = torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3))
        d = self.get_n_pixels(x)
        return torch.sqrt(n / d)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return (
                self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + self.mu(y)
        ).permute([2, 3, 0, 1])

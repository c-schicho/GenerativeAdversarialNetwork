import glob
import os.path
from typing import Optional, Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

DATA_PATH = "data"
MNIST_PATH = os.path.join(DATA_PATH, "mnist")
POKEMON_PATH = os.path.join(DATA_PATH, "pokemon")


class CustomImageDataset(Dataset):

    def __init__(self, path: str, transform: Optional[Callable] = None):
        super(CustomImageDataset, self).__init__()
        self.img_paths = sorted(glob.glob(os.path.join(path, "**", "*.png"), recursive=True))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.img_paths[idx])

        if self.transform:
            item = self.transform(img)
        else:
            item = img

        return item


class MNISTDataset(CustomImageDataset):

    def __init__(self, transform: Optional[Callable] = ToTensor):
        super().__init__(MNIST_PATH, transform=transform)


class PokemonImageDataset(CustomImageDataset):

    def __init__(self, transform: Optional[Callable] = ToTensor):
        super().__init__(POKEMON_PATH, transform=transform)

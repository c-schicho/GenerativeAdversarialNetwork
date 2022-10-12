from torch.utils.data import DataLoader

from data.datasets import *


def get_dataloader(
        dataset: str,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        flatten: bool = False,
        transform: Optional[Callable] = ToTensor()
) -> DataLoader:
    if dataset == "mnist":
        dataset = MNISTDataset(transform=transform, flatten=flatten)

    elif dataset == "pokemon":
        dataset = PokemonImageDataset(transform=transform, flatten=flatten)

    else:
        raise NotImplementedError("dataset does not exist")

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

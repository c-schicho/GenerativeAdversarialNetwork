from torch.utils.data import DataLoader

from datasets import *


def get_dataloader(
        dataset: str,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4
) -> DataLoader:
    if dataset == "mnist":
        dataset = MNISTDataset()

    elif dataset == "pokemon":
        dataset = PokemonImageDataset()

    else:
        raise NotImplementedError("dataset does not exist")

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

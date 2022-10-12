import os

import torch
from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stylegan.discriminator import StyleGANDiscriminator
from stylegan.generator import StyleGANGenerator
from utils import write_train_stats


class Trainer:

    def __init__(self, l_rate: float, z_dim: int, image_dim: int, generator_path: str, discriminator_path: str,
                 tensorboard_path: str):
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path
        self.tensorboard_path = tensorboard_path
        self.generator_model_path = os.path.join(generator_path, "pokemon_generator.pt")
        self.discriminator_model_path = os.path.join(discriminator_path, "pokemon_discriminator.pt")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = StyleGANGenerator(in_dims=z_dim, out_dims=image_dim).to(self.device)
        self.discriminator = StyleGANDiscriminator(in_dims=image_dim).to(self.device)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=l_rate)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=l_rate)
        self.loss_fun = BCELoss()
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.generator_loss = None
        self.discriminator_loss = None

    def train(self, epochs: int, dataloader: DataLoader, load_model: bool, save_after_epochs: int):
        writer = SummaryWriter(self.tensorboard_path)
        self.initialize_folder_structure()

        if load_model:
            self.generator.load(self.generator_model_path)
            self.discriminator.load(self.discriminator_model_path)

        for epoch in range(1, epochs + 1):
            for data in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Epoch {epoch}/{epochs}"):
                data_batch_size = data.size(dim=0)
                self.train_discriminator(data, data_batch_size)
                self.train_generator(data_batch_size)

            write_train_stats(
                writer=writer,
                epoch=epoch,
                generator_loss=self.generator_loss.cpu().item(),
                discriminator_loss=self.discriminator_loss.cpu().item(),
                generated_data=self.generate_fake_data(data_batch_size).cpu()
            )

            if epoch % save_after_epochs == 0:
                self.generator.save(self.generator_model_path)
                self.discriminator.save(self.discriminator_model_path)

    def train_generator(self, data_batch_size: int):
        pass

    def train_discriminator(self, data_batch: Tensor, data_batch_size: int):
        pass

    def generate_fake_data(self, data_batch_size: int) -> Tensor:
        random_noise = torch.randn(data_batch_size, self.z_dim, 1, 1, device=self.device)
        return self.generator(random_noise)

    def initialize_folder_structure(self):
        os.makedirs(self.generator_path, exist_ok=True)
        os.makedirs(self.discriminator_path, exist_ok=True)
        os.makedirs(self.tensorboard_path, exist_ok=True)

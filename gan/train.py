import torch
from torch import Tensor

from gan.discriminator import FNNDiscriminator
from gan.generator import FNNGenerator


class Trainer:

    def __init__(self, generator: FNNGenerator, discriminator: FNNDiscriminator, generator_optimizer,
                 discriminator_optimizer, loss_fun, z_dim: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fun = loss_fun
        self.z_dim = z_dim
        self.generator_loss = None
        self.discriminator_loss = None

    def train_generator(self, data_batch_size: int):
        self.generator.zero_grad()

        fake_data = self.generate_fake_data(data_batch_size)

        fake_output = self.discriminator(fake_data)

        self.generator_loss = self.loss_fun(fake_output, torch.ones(data_batch_size, 1).to(self.device))
        self.generator_loss.backward()
        self.generator_optimizer.step()

    def train_discriminator(self, data_batch: Tensor, data_batch_size: int):
        self.discriminator.zero_grad()

        fake_data = self.generate_fake_data(data_batch_size)

        real_data = data_batch.to(self.device)
        real_output = self.discriminator(real_data)
        real_loss = self.loss_fun(real_output, torch.ones(data_batch_size, 1).to(self.device))

        fake_output = self.discriminator(fake_data)
        fake_loss = self.loss_fun(fake_output, torch.zeros(data_batch_size, 1).to(self.device))

        self.discriminator_loss = real_loss + fake_loss
        self.discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def generate_fake_data(self, data_batch_size: int) -> Tensor:
        random_noise = torch.randn(data_batch_size, self.z_dim).to(self.device)
        return self.generator(random_noise)

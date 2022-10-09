import torch
from torch import Tensor

from dcgan.discriminator import DCGANDiscriminator
from dcgan.generator import DCGANGenerator


class Trainer:

    def __init__(self, generator: DCGANGenerator, discriminator: DCGANDiscriminator, generator_optimizer,
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
        fake_output = self.discriminator(fake_data).view(-1)

        real_targets = torch.rand(data_batch_size, device=self.device) * 0.1 + 0.9
        self.generator_loss = self.loss_fun(fake_output, real_targets)
        self.generator_loss.backward()
        self.generator_optimizer.step()

    def train_discriminator(self, data_batch: Tensor, data_batch_size: int):
        self.discriminator.zero_grad()

        fake_data = self.generate_fake_data(data_batch_size)

        real_data = data_batch.to(self.device)
        real_output = self.discriminator(real_data).view(-1)
        real_targets = torch.rand(data_batch_size, device=self.device) * 0.1 + 0.9
        real_loss = self.loss_fun(real_output, real_targets)
        real_loss.backward()

        fake_output = self.discriminator(fake_data).view(-1)
        fake_targets = torch.rand(data_batch_size, device=self.device) * 0.1
        fake_loss = self.loss_fun(fake_output, fake_targets)
        fake_loss.backward()

        self.discriminator_loss = real_loss + fake_loss
        self.discriminator_optimizer.step()

    def generate_fake_data(self, data_batch_size: int) -> Tensor:
        random_noise = torch.randn(data_batch_size, self.z_dim, 1, 1, device=self.device)
        return self.generator(random_noise)
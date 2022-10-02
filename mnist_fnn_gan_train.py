import json
from argparse import ArgumentParser

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

from dataloader import get_dataloader
from gan.discriminator import FNNDiscriminator
from gan.generator import FNNGenerator


def main(
        load_model: bool,
        batch_size: int,
        epochs: int,
        l_rate: float,
        generator_path: str = "models/generator.pt",
        discriminator_path: str = "models/discriminator.pt",
        tensorboard_path: str = "results/mnist_fnn_gan"
):
    random_noise_dim = 100
    mnist_image_dim = 28 * 28
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([
        ToTensor(),
        Normalize(mean=.5, std=.5)
    ])
    dataloader = get_dataloader(dataset="mnist", batch_size=batch_size, transform=transform, flatten=True)
    generator = FNNGenerator(in_dims=random_noise_dim, out_dims=mnist_image_dim).to(device)
    discriminator = FNNDiscriminator(in_dims=mnist_image_dim).to(device)
    generator_optimizer = Adam(generator.parameters(), lr=l_rate)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=l_rate)
    loss_fun = BCELoss()
    writer = SummaryWriter(tensorboard_path)

    if load_model:
        generator.load(generator_path)
        discriminator.load(discriminator_path)

    for epoch in range(epochs):
        for data in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Epoch {epoch}/{epochs}"):
            data_batch_size = data.size(dim=0)

            # train discriminator
            discriminator.zero_grad()

            random_noise = torch.randn(data_batch_size, random_noise_dim).to(device)
            fake_data = generator(random_noise)

            real_data = data.to(device)
            real_output = discriminator(real_data)
            real_loss = loss_fun(real_output, torch.ones(data_batch_size, 1).to(device))

            fake_output = discriminator(fake_data)
            fake_loss = loss_fun(fake_output, torch.zeros(data_batch_size, 1).to(device))

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # train generator
            generator.zero_grad()

            random_noise = torch.randn(data_batch_size, random_noise_dim).to(device)
            fake_data = generator(random_noise)

            fake_output = discriminator(fake_data)

            generator_loss = loss_fun(fake_output, torch.ones(data_batch_size, 1).to(device))
            generator_loss.backward()
            generator_optimizer.step()

        # write training stats to tensorboard
        writer.add_scalar(tag="generator loss", scalar_value=generator_loss.cpu().item(), global_step=epoch)
        writer.add_scalar(tag="discriminator loss", scalar_value=discriminator_loss.cpu().item(), global_step=epoch)
        writer.add_images(tag="generated images", dataformats="NCHW",
                          img_tensor=fake_data.cpu().view(data_batch_size, 1, 28, 28), global_step=epoch)

        if epoch % 20 == 0:
            generator.save(generator_path)
            discriminator.save(discriminator_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="train_config.json",
        type=str,
        required=False,
        help="path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(**config)

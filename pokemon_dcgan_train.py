import json
import os
from argparse import ArgumentParser

from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip
from tqdm import tqdm

from dataloader import get_dataloader
from dcgan import DCGANGenerator, DCGANDiscriminator, Trainer


def main(
        load_model: bool,
        batch_size: int,
        epochs: int,
        l_rate: float,
        generator_path: str = "models/pokemon_dcgan",
        discriminator_path: str = "models/pokemon_dcgan",
        tensorboard_path: str = "results/pokemon_dcgan",
        save_after_epochs: int = 20,
        random_noise_dim: int = 100,
        image_dim: int = 64
):
    os.makedirs(generator_path, exist_ok=True)
    os.makedirs(discriminator_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    generator_model_path = os.path.join(generator_path, "pokemon_generator.pt")
    discriminator_model_path = os.path.join(discriminator_path, "pokemon_discriminator.pt")

    transform = Compose([
        ToTensor(),
        Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
        Resize(image_dim),
        RandomHorizontalFlip()
    ])
    dataloader = get_dataloader(dataset="pokemon", batch_size=batch_size, transform=transform)
    generator = DCGANGenerator(in_dims=random_noise_dim, out_dims=image_dim)
    discriminator = DCGANDiscriminator(in_dims=image_dim)
    generator_optimizer = Adam(generator.parameters(), lr=l_rate)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=l_rate)
    loss_fun = BCELoss()
    dcgan_trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        loss_fun=loss_fun,
        z_dim=random_noise_dim
    )
    writer = SummaryWriter(tensorboard_path)

    if load_model:
        generator.load(generator_model_path)
        discriminator.load(discriminator_model_path)

    for epoch in range(1, epochs + 1):
        for data in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Epoch {epoch}/{epochs}"):
            data_batch_size = data.size(dim=0)
            dcgan_trainer.train_discriminator(data, data_batch_size)
            dcgan_trainer.train_generator(data_batch_size)

        writer.add_scalar(
            tag="generator loss",
            scalar_value=dcgan_trainer.generator_loss.cpu().item(),
            global_step=epoch
        )
        writer.add_scalar(
            tag="discriminator loss",
            scalar_value=dcgan_trainer.discriminator_loss.cpu().item(),
            global_step=epoch
        )
        writer.add_images(
            tag="generated images",
            dataformats="NCHW",
            img_tensor=dcgan_trainer.generate_fake_data(data_batch_size).cpu(),
            global_step=epoch
        )

        if epoch % save_after_epochs == 0:
            generator.save(generator_model_path)
            discriminator.save(discriminator_model_path)


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

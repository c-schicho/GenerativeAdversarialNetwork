import json
from argparse import ArgumentParser

from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

from dataloader import get_dataloader
from gan import FNNGenerator, FNNDiscriminator, Trainer


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

    transform = Compose([
        ToTensor(),
        Normalize(mean=.5, std=.5)
    ])
    dataloader = get_dataloader(dataset="mnist", batch_size=batch_size, transform=transform, flatten=True)
    generator = FNNGenerator(in_dims=random_noise_dim, out_dims=mnist_image_dim)
    discriminator = FNNDiscriminator(in_dims=mnist_image_dim)
    generator_optimizer = Adam(generator.parameters(), lr=l_rate)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=l_rate)
    loss_fun = BCELoss()
    gan_trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        loss_fun=loss_fun,
        z_dim=random_noise_dim
    )
    writer = SummaryWriter(tensorboard_path)

    if load_model:
        generator.load(generator_path)
        discriminator.load(discriminator_path)

    for epoch in range(epochs):
        for data in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Epoch {epoch}/{epochs}"):
            data_batch_size = data.size(dim=0)
            gan_trainer.train_discriminator(data, data_batch_size)
            gan_trainer.train_generator(data_batch_size)

        writer.add_scalar(
            tag="generator loss",
            scalar_value=gan_trainer.generator_loss.cpu().item(),
            global_step=epoch
        )
        writer.add_scalar(
            tag="discriminator loss",
            scalar_value=gan_trainer.discriminator_loss.cpu().item(),
            global_step=epoch
        )
        writer.add_images(
            tag="generated images",
            dataformats="NCHW",
            img_tensor=gan_trainer.generate_fake_data(data_batch_size).cpu().view(data_batch_size, 1, 28, 28),
            global_step=epoch
        )

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

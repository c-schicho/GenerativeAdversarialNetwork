import json
from argparse import ArgumentParser

from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip

from data import get_dataloader
from dcgan import Trainer as DCGAN_Trainer
from gan import Trainer as GAN_Trainer


def main(
        dataset: str,
        model: str,
        load_model: bool,
        batch_size: int,
        epochs: int,
        l_rate: float,
        generator_path: str,
        discriminator_path: str,
        tensorboard_path: str,
        save_after_epochs: int = 20,
        random_noise_dim: int = 100,
        image_dim: int = 64
):
    assert dataset in ["pokemon", "mnist"], "only pokemon and mnist datasets are supported"
    assert model in ["gan", "dcgan"], "only gan and dcgan models are supported"

    if dataset == "pokemon":
        assert model == "dcgan", "only dcgan is supported for the pokemon dataset"

        if model == "dcgan":
            transform = Compose([
                ToTensor(),
                Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
                Resize(image_dim),
                RandomHorizontalFlip()
            ])
            dataloader = get_dataloader(dataset=dataset, batch_size=batch_size, transform=transform)
            dcgan_trainer = DCGAN_Trainer(
                l_rate=l_rate,
                z_dim=random_noise_dim,
                image_dim=image_dim,
                generator_path=generator_path,
                discriminator_path=discriminator_path,
                tensorboard_path=tensorboard_path
            )
            dcgan_trainer.train(
                epochs=epochs,
                dataloader=dataloader,
                load_model=load_model,
                save_after_epochs=save_after_epochs
            )

    elif dataset == "mnist":
        assert model == "gan", "only gan is supported for the mnist dataset"

        if model == "gan":
            transform = Compose([
                ToTensor(),
                Normalize(mean=.5, std=.5)
            ])
            dataloader = get_dataloader(dataset=dataset, batch_size=batch_size, transform=transform, flatten=True)
            gan_trainer = GAN_Trainer(
                l_rate=l_rate,
                z_dim=random_noise_dim,
                image_dim=image_dim,
                generator_path=generator_path,
                discriminator_path=discriminator_path,
                tensorboard_path=tensorboard_path
            )
            gan_trainer.train(
                epochs=epochs,
                dataloader=dataloader,
                load_model=load_model,
                save_after_epochs=save_after_epochs
            )


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

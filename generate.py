import json
import os
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from dcgan import DCGANGenerator
from gan import FNNGenerator
from stylegan import StyleGANGenerator


def main(
        model: str,
        number_images: int,
        generator_path: str,
        output_path: str,
        random_noise_dim: int = 100,
        image_dim: int = 64
):
    assert model in ["gan", "dcgan", "stylegan"], "only gan, dcgan and stylegan models are supported"

    os.makedirs(output_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "stylegan":
        generator = StyleGANGenerator()
        random_noise = None
    elif model == "dcgan":
        generator = DCGANGenerator(in_dims=random_noise_dim, out_dims=image_dim)
        random_noise = torch.randn(number_images, random_noise_dim, 1, 1, device=device)
    else:
        generator = FNNGenerator(in_dims=random_noise_dim, out_dims=image_dim)
        random_noise = torch.randn(number_images, random_noise_dim, device=device)

    generator.load(generator_path)
    generator.to(device)
    generator.eval()

    images = generator(random_noise)

    for idx, image in enumerate(images):
        save_image(image, os.path.join(output_path, f"img_{idx}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="generate_config.json",
        type=str,
        required=False,
        help="path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(**config)

import json
import math
import os
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from gan import FNNGenerator


def main(
        number_images: int,
        generator_path: str = "models/mnist_gan",
        output_path: str = "images/mnist_gan",
        random_noise_dim: int = 100,
        image_dim: int = 28 * 28
):
    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = FNNGenerator(in_dims=random_noise_dim, out_dims=image_dim)
    generator.load(os.path.join(generator_path, "mnist_generator.pt"))
    generator.to(device)
    generator.eval()

    random_noise = torch.randn(number_images, random_noise_dim, device=device)

    image_width = int(math.sqrt(image_dim))
    images = generator(random_noise).cpu().view(number_images, 1, image_width, image_width)

    for idx, image in enumerate(images):
        save_image(image, os.path.join(output_path, f"img_{idx}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="generation_config.json",
        type=str,
        required=False,
        help="path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(**config)

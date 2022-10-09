import json
import math
import os
from argparse import ArgumentParser

import torch
from torchvision.utils import save_image

from dcgan import DCGANGenerator


def main():
    pass


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

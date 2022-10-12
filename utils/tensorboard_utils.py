from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def write_train_stats(writer: SummaryWriter, epoch: int, generator_loss: float, discriminator_loss: float,
                      generated_data: Tensor):
    writer.add_scalar(
        tag="generator loss",
        scalar_value=generator_loss,
        global_step=epoch
    )
    writer.add_scalar(
        tag="discriminator loss",
        scalar_value=discriminator_loss,
        global_step=epoch
    )
    writer.add_images(
        tag="generated images",
        dataformats="NCHW",
        img_tensor=generated_data,
        global_step=epoch
    )

from pathlib import Path

import matplotlib.pyplot as plt
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter


def class_weights_heatmap(
    tensorboard_writer: SummaryWriter, net: DataParallel, epoch: int
):
    tensorboard_writer.add_image(
        "classifier weights",
        net.module._classification.weight,
        epoch,
        dataformats="HW",
    )


def prototype_activations_violin_plot(
    tensorboard_writer: SummaryWriter, net: DataParallel, epoch: int, train_info: dict
):
    plt.clf()
    plt.figure().set_figheight(net.module._num_prototypes)
    plt.ylim((0, net.module._num_prototypes + 1))
    plt.violinplot(
        train_info["prototype_activations"],
        vert=False,
    )
    plt.yticks(range(1, net.module._num_prototypes + 1))
    plt.tight_layout()
    tensorboard_writer.add_figure(
        "prototype activations",
        plt.gcf(),
        epoch,
    )


def save_lr_curve(learning_rates: list, save_path: Path):
    plt.clf()
    plt.plot(learning_rates)
    plt.savefig(save_path)

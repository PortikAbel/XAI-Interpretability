from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from utils.log import Log


def save_model_w_condition(
    model: nn.Module,
    log: Log,
    model_dir: str | Path,
    model_name: str,
    accu: float,
    target_accu: float = 0.6,
) -> None:
    """
    Save the specified model if the condition is met regarding the accuracy.

    :param model: this is not the multi-gpu model
    :param log: the logger
    :param model_dir: the directory where the model should be saved
    :param model_name: the name of the model
    :param accu: the accuracy of the model
    :param target_accu: the target accuracy. Defaults to ``0.6``.
    """
    if type(model_dir) is str:
        model_dir = Path(model_dir)
    if accu > target_accu:
        log.info(f"\tabove {target_accu:.2%}")
        # torch.save(
        #     obj=model.state_dict(),
        #     f=model_dir / f"{model_name}_state_dict_{accu:.4f}.pth"
        # )
        torch.save(
            obj=model,
            f=model_dir / f"{model_name}{accu:.4f}.pth",
        )


def save_image(fname, arr):
    if arr.shape[-1] == 1:
        plt.imsave(
            fname=fname,
            arr=arr.squeeze(axis=2),
            cmap="gray",
        )
    else:
        plt.imsave(
            fname=fname,
            arr=arr,
        )

import torch


def preprocess(
    x: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    in_channels: int = 3,
) -> torch.Tensor:
    """
    Normalize the input tensor x with the mean and std.

    :param x: input data
    :param mean: mean of the dataset
    :param std: standard deviation of the dataset
    :param in_channels: number of color channels in the input data
    :return: the normalized input tensor
    """
    assert x.size(1) == in_channels
    y = torch.zeros_like(x)
    for i in range(in_channels):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_preprocess(
    x: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    in_channels: int = 3,
) -> torch.Tensor:
    """
    Undo the normalization used on the input data.

    :param x: normalized input data
    :param mean: mean of the dataset
    :param std: standard deviation of the dataset
    :param in_channels: number of color channels in the input data
    :return: the un-normalized input tensor
    """
    assert x.size(1) == in_channels
    y = torch.zeros_like(x)
    for i in range(in_channels):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

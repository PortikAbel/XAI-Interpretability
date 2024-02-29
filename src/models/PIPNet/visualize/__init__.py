import argparse  # noqa

import numpy as np
from pathlib import Path  # noqa

from PIL import Image
import torchvision.transforms as transforms


def get_patch_size(args):
    """
    Get patch size and skip size for the given image size and window shape.

    :param args: command line arguments
    :type args: argparse.Namespace
    :return: patch size and skip size (in each direction)
    :rtype: tuple[int, tuple[int, int]]
    """
    patch_size = 32
    skip = np.round((args.image_size - patch_size) / (args.wshape - 1)).astype(int)
    return patch_size, skip


def get_img_coordinates(img_size, softmaxes_shape, patch_size, skip, h_idx, w_idx):
    """
    Convert latent location to coordinates of image patch

    :param img_size: size of the image
    :type img_size: tuple[int, int]
    :param softmaxes_shape: shape of the softmaxes
    :type softmaxes_shape: tuple[int, int, int]
    :param patch_size: size of the patch
    :type patch_size: int
    :param skip: skip size
    :type skip: tuple[int, int]
    :param h_idx: index of the patch in the height dimension
    :type h_idx: int
    :param w_idx: index of the patch in the width dimension
    :type w_idx: int
    :return: coordinates of the image patch on the original image
    :rtype: tuple[int, int, int, int]
    """
    # in case latent output size is 26x26. For convnext with smaller strides.
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        # Since the outer latent patches have a smaller receptive field,
        # skip size is set to 4 for the first and last patch. 8 for rest.
        h_coord_min = max(0, (h_idx - 1) * skip[0] + 4)
        if h_idx < softmaxes_shape[-1] - 1:
            h_coord_max = h_coord_min + patch_size
        else:
            h_coord_min -= 4
            h_coord_max = h_coord_min + patch_size
        w_coord_min = max(0, (w_idx - 1) * skip[1] + 4)
        if w_idx < softmaxes_shape[-1] - 1:
            w_coord_max = w_coord_min + patch_size
        else:
            w_coord_min -= 4
            w_coord_max = w_coord_min + patch_size
    else:
        h_coord_min = h_idx * skip[0]
        h_coord_max = min(img_size[0], h_idx * skip[0] + patch_size)
        w_coord_min = w_idx * skip[1]
        w_coord_max = min(img_size[1], w_idx * skip[1] + patch_size)

    if h_idx == softmaxes_shape[1] - 1:
        h_coord_max = img_size[0]
    if w_idx == softmaxes_shape[2] - 1:
        w_coord_max = img_size[1]
    if h_coord_max == img_size[0]:
        h_coord_min = img_size[0] - patch_size
    if w_coord_max == img_size[1]:
        w_coord_min = img_size[1] - patch_size

    return h_coord_min, h_coord_max, w_coord_min, w_coord_max


def get_patch(img_path, args, h_idx, w_idx, softmaxes_size, patch_size=None, skip=None):
    """
    Get image patch from image at given location

    :param img_path: path to image
    :type img_path: Path
    :param args: command line arguments
    :type args: argparse.Namespace
    :param h_idx: index of the patch in the height dimension
    :type h_idx: int
    :param w_idx: index of the patch in the width dimension
    :type w_idx: int
    :param softmaxes_size: shape of the softmaxes
    :type softmaxes_size: tuple[int, int, int]
    :param patch_size: size of the patch. Defaults to None.
    :type patch_size: int
    :param skip: skip size. Defaults to None.
    :type skip: tuple[int, int]
    :return: image patch, coordinates of the patch on the original image
    """
    if patch_size is None or skip is None:
        patch_size, skip = get_patch_size(args)

    image = transforms.Resize(size=tuple(args.image_size))(
        Image.open(img_path).convert("RGB")
    )
    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)  # shape (1, 3, h, w)
    (
        h_coord_min,
        h_coord_max,
        w_coord_min,
        w_coord_max,
    ) = get_img_coordinates(
        args.image_size,
        softmaxes_size,
        patch_size,
        skip,
        h_idx,
        w_idx,
    )
    img_tensor_patch = img_tensor[
        0, :, h_coord_min:h_coord_max, w_coord_min:w_coord_max
    ]
    return (
        image,
        img_tensor_patch,
        (h_coord_max, h_coord_min),
        (w_coord_max, w_coord_min),
    )

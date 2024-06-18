import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data.config import DATASETS
from utils.environment import get_env


def set_device(gpu_ids: str, disable_gpu: bool = False) -> tuple[torch.device, list]:
    """
    Set the device to use for training.

    :param gpu_ids: GPU ids separated with comma
    :param disable_gpu: Whether to disable GPU. Defaults to ``False``.
    :return: The device to use for training
    """
    gpu_list = gpu_ids.split(",")
    device_ids = []
    if gpu_ids != "":
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    if not disable_gpu and torch.cuda.is_available():
        if len(device_ids) == 1:
            return torch.device(f"cuda:{gpu_ids}"), device_ids
        elif len(device_ids) == 0:
            device = torch.device("cuda")
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
            return device, device_ids
        else:
            print(
                "This code should work with multiple GPUs "
                "but we didn't test that, so we recommend to use only 1 GPU.",
                flush=True,
            )
            device_str = ""
            for d in device_ids:
                device_str += str(d)
                device_str += ","
            return torch.device("cuda:" + str(device_ids[0])), device_ids
    return torch.device("cpu"), []


class GeneralModelParametersParser(argparse.ArgumentParser):
    """
    Parse general arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(
            "General arguments",
            description="Specifies general arguments",
            **kwargs,
        )
        self.add_argument(
            "--seed",
            type=int,
            default=1,
            help="Random seed. Note that there will still be differences "
            "between runs due to nondeterminism. "
            "See https://pytorch.org/docs/stable/notes/randomness.html",
        )
        self.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Num workers in dataloaders.",
        )

        gpu_group = self.add_argument_group("GPU", "Specifies the GPU settings")
        gpu_group.add_argument(
            "--gpu_ids",
            type=str,
            default="",
            help="ID of gpu. Can be separated with comma",
        )
        gpu_group.add_argument(
            "--disable_gpu",
            action="store_true",
            help="Flag that disables GPU usage if set",
        )

        log_group = self.add_argument_group(
            "Logging",
            "Specifies the directory where the log files "
            "and other outputs should be saved",
        )
        log_group.add_argument(
            "--log_dir",
            type=Path,
            default=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
            help="The directory in which train progress should be logged",
        )
        log_group.add_argument(
            "--dir_for_saving_images",
            type=str,
            default="visualization_results",
            help="Directory for saving the prototypes and explanations",
        )

        dataset_group = self.add_argument_group(
            "Dataset", "Specifies the dataset to use and its hyperparameters"
        )
        dataset_group.add_argument(
            "--dataset",
            type=str,
            default="CUB-10",
            help="Data set on ProtoPNet should be trained",
        )
        dataset_group.add_argument(
            "--validation_size",
            type=float,
            default=0.0,
            help="Split between training and validation set. Can be zero when "
            "there is a separate test or validation directory. "
            "Should be between 0 and 1. Used for partimagenet (e.g. 0.2)",
        )
        dataset_group.add_argument(
            "--disable_normalize",
            action="store_true",
            help="Flag that disables normalization of the images",
        )

        image_size_group = dataset_group.add_argument_group(
            "Image size",
            "Specifies the size of the images. At least one of them is required",
        )
        image_size_group.add_argument(
            "--image_width",
            type=np.uint16,
            help="The width of the images in the dataset",
        )
        image_size_group.add_argument(
            "--image_height",
            type=np.uint16,
            help="The height of the images in the dataset",
        )

    @staticmethod
    def validate_data(
        args: argparse.Namespace, model_name: str, backbone_network: str = None
    ) -> argparse.Namespace:
        """
        Validate the data.

        :param args: Arguments to be validated
        :param model_name: Name of the model being trained
        :param backbone_network: Name of the backbone network. Defaults to ``None``.
        """
        if backbone_network is not None:
            model_name = f"{model_name}/{backbone_network}"
        results_location = get_env("RESULTS_LOCATION", must_exist=False) or get_env("PROJECT_ROOT")
        args.log_dir = Path(results_location, "runs", model_name, args.log_dir)
        args.log_dir = args.log_dir.resolve()
        args.log_dir.mkdir(parents=True, exist_ok=True)

        args.device, args.device_ids = set_device(args.gpu_ids, args.disable_gpu)

        if args.dataset not in DATASETS:
            raise ValueError(f"Dataset {args.dataset} is not supported")
        dataset_config = DATASETS[args.dataset]

        # assign properties in dataset config to args
        if args.image_width is None and args.image_height is None:
            args.image_width = dataset_config["img_shape"][1]
            args.image_height = dataset_config["img_shape"][0]
        elif args.image_width is None:
            args.image_width = args.image_height
        elif args.image_height is None:
            args.image_height = args.image_width
        args.img_shape = np.array((args.image_height, args.image_width))
        args.num_classes = dataset_config["num_classes"]
        args.color_channels = dataset_config["color_channels"]
        args.data_dir = dataset_config["data_dir"]
        args.train_dir = dataset_config["train_dir"]
        args.train_dir_projection = dataset_config.get(
            "train_dir_projection", args.train_dir
        )
        args.test_dir = dataset_config["test_dir"]
        args.test_dir_projection = dataset_config.get(
            "test_dir_projection", args.test_dir
        )
        args.image_folders = dataset_config["image_folders"]
        args.mean = dataset_config["mean"]
        args.std = dataset_config["std"]
        args.augm = dataset_config["augm"]

        return args

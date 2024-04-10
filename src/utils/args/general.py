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
    
    :param model_name: The name of the model being trained
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            "General arguments", description="Specifies general arguments", **kwargs,
        )
        self.add_argument(
            "--seed",
            type=int,
            default=1,
            help="Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html",
        )
        self.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Num workers in dataloaders.",
        )

        gpu_group = self.add_argument_group(
            "GPU", "Specifies the GPU settings"
        )
        gpu_group.add_argument(
            "--gpu_ids",
            type=str,
            default="",
            help="ID of gpu. Can be separated with comma",
        )
        gpu_group.add_argument(
            "--disable_cuda",
            action="store_true",
            help="Flag that disables GPU usage if set",
        )

        log_group = self.add_argument_group(
            "Logging",
            "Specifies the directory where the log files and other outputs should be saved",
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
            help="Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)",
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
        
        self.__model_name = model_name
    
    def _validate_data(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Validate the data.

        :param args: The arguments to validate
        """
        args.log_dir = Path(
            get_env("PROJECT_ROOT"), "runs", self.__model_name, args.log_dir
        )
        args.log_dir = args.log_dir.resolve()
        args.log_dir.mkdir(parents=True, exist_ok=True)

        args.device, args.device_ids = set_device(
            args.gpu_ids, args.disable_gpu
        )
        
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
        args.image_shape = np.array((args.image_height, args.image_width))
        args.num_classes = dataset_config["num_classes"]
        args.color_channels = dataset_config["color_channels"]
        args.train_dir = dataset_config["train_dir"]
        args.train_dir_projection = dataset_config["train_dir_projection"]
        args.test_dir = dataset_config["test_dir"]
        args.test_dir_projection = dataset_config["test_dir_projection"]
        args.image_folders = dataset_config["image_folders"]
        args.mean = dataset_config["mean"]
        args.std = dataset_config["std"]
        args.augm = dataset_config["augm"]
        
        return args
        
    def parse_known_args(self, args: list = None, namespace: argparse.Namespace = None):
        namespace, args = super().parse_known_args(args, namespace)
        namespace = self._validate_data(namespace)
        return namespace, args
        
    def parse_args(self, args: list = None, namespace: argparse.Namespace = None):
        namespace = super().parse_args(args, namespace)
        namespace = self._validate_data(namespace)
        return namespace

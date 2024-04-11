import argparse
import os
import pickle
from pathlib import Path
from datetime import datetime as dt
import numpy as np
from utils.environment import get_env


def get_args() -> argparse.Namespace:
    """
    Utility functions for handling parsed arguments
    """
    parser = argparse.ArgumentParser("Train a PIP-Net")

    dataset_group = parser.add_argument_group(
        "Dataset", "Specifies the dataset to use and its hyperparameters"
    )
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="CUB-10",
        help="Data set on PIP-Net should be trained",
    )
    dataset_group.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="""Split between training and validation set.
            Can be zero when there is a separate test or validation directory.
            Should be between 0 and 1.
            Used for partimagenet (e.g. 0.2)""",
    )

    image_size_group = parser.add_argument_group(
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

    net_group = parser.add_mutually_exclusive_group()
    net_group.add_argument(
        "--net",
        type=str,
        default="convnext_tiny_26",
        help="""Base network used as backbone of PIP-Net.
            Default is convnext_tiny_26 with adapted strides
            to output 26x26 latent representations.
            Other option is convnext_tiny_13 that outputs 13x13
            (smaller and faster to train, less fine-grained).
            Pretrained network on iNaturalist is only available for resnet50_inat.
            Options are: resnet18, resnet34, resnet50, resnet50_inat,
                resnet101, resnet152,
                convnext_tiny_26 and convnext_tiny_13.""",
    )
    net_group.add_argument(
        "--state_dict_dir_net",
        type=Path,
        help="""The directory containing a state dict with a pretrained PIP-Net.
            E.g., ./code/PIPNet/runs/run_pipnet/checkpoints/net_pretrained""",
    )

    net_parameter_group = parser.add_argument_group(
        "Network parameters", "Specifies the used network's hyperparameters"
    )
    net_parameter_group.add_argument(
        "--batch_size",
        type=np.uint16,
        default=64,
        help="""Batch size when training the model using minibatch gradient descent.
            Batch size is multiplied with number of available GPUs""",
    )
    net_parameter_group.add_argument(
        "--batch_size_pretrain",
        type=np.uint16,
        default=128,
        help="Batch size when pretraining the prototypes (first training stage)",
    )
    net_parameter_group.add_argument(
        "--train_backbone_during_pretrain",
        action="store_true",
        help="To train the whole backbone during pretrain "
        "(e.g. if dataset is very different from ImageNet)",
    )
    net_parameter_group.add_argument(
        "--epochs",
        type=np.uint16,
        default=60,
        help="The number of epochs PIP-Net should be trained (second training stage)",
    )
    net_parameter_group.add_argument(
        "--epochs_pretrain",
        type=np.uint16,
        default=10,
        help="""Number of epochs to pre-train the prototypes (first training stage).
            Recommended to train at least until the align loss < 1""",
    )
    net_parameter_group.add_argument(
        "--freeze_epochs",
        type=np.uint16,
        default=10,
        help="""Number of epochs where pretrained features_net will be frozen
            while training classification layer (and last layer(s) of backbone)""",
    )
    net_parameter_group.add_argument(
        "--epochs_to_finetune",
        type=np.uint16,
        default=3,
        help="""during finetuning, only train classification layer and freeze rest.
            usually done for a few epochs
            (at least 1, more depends on size of dataset)""",
    )
    net_parameter_group.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Flag that disables GPU usage if set",
    )
    net_parameter_group.add_argument(
        "--num_features",
        type=int,
        default=0,
        help="""Number of prototypes.
            When zero (default) the number of prototypes
            is the number of output channels of backbone.
            If this value is set, then a 1x1 conv layer will be added.
            Recommended to keep 0, but can be increased
            when number of classes > num output channels in backbone.""",
    )
    net_parameter_group.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="""When set, the backbone network is initialized with random weights
            instead of being pretrained on another dataset).""",
    )
    net_parameter_group.add_argument(
        "--bias",
        action="store_true",
        help="""Flag that indicates whether to include a trainable bias
            in the linear classification layer.""",
    )

    optimizer_group = parser.add_argument_group(
        "Optimizer", "Specifies the optimizer to use and its hyperparameters"
    )
    optimizer_group.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer that should be used when training PIP-Net",
    )
    optimizer_group.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="""The optimizer learning rate for training the weights
            from prototypes to classes""",
    )
    optimizer_group.add_argument(
        "--lr_block",
        type=float,
        default=0.0005,
        help="""The optimizer learning rate
            for training the last conv layers of the backbone""",
    )
    optimizer_group.add_argument(
        "--lr_net",
        type=float,
        default=0.0005,
        help="""The optimizer learning rate for the backbone.
            Usually similar as lr_block.""",
    )
    optimizer_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used in the optimizer",
    )

    loss_group = parser.add_argument_group(
        "Loss", "Specifies the loss function to use and its hyperparameters"
    )
    loss_group.add_argument(
        "--weighted_loss",
        action="store_true",
        help="""Flag that weights the loss based on the class balance of the dataset.
            Recommended to use when data is imbalanced.""",
    )
    loss_group.add_argument(
        "--tanh_loss",
        type=float,
        default=0.0,
        help="""tanh loss regulates that every prototype
            should be at least oncepresent in a mini-batch.""",
    )
    loss_group.add_argument(
        "--unif_loss",
        type=float,
        default=0.0,
        help="""Our tanh-loss optimizes for uniformity
            and was sufficient for our experiments.
            However, if pretraining of the prototypes
            is not working well for your dataset,
            you may try to add another uniformity loss
            from https://www.tongzhouwang.info/hypersphere/""",
    )
    loss_group.add_argument(
        "--variance_loss",
        type=float,
        default=0.0,
        help="""Regularizer term that enforces variance of features
            from https://arxiv.org/abs/2105.04906""",
    )

    log_group = parser.add_argument_group(
        "Logging",
        "Specifies the directory where the log files and other outputs should be saved",
    )
    log_group.add_argument(
        "--dir_for_saving_images",
        type=str,
        default="visualization_results",
        help="Directoy for saving the prototypes and explanations",
    )
    log_group.add_argument(
        "--log_prototype_activations_violin_plot",
        action="store_true",
        help="""Logs a violinplot in tensorboard
            for every prototype with their activations during train step""",
    )

    visualization_group = parser.add_argument_group(
        "Visualization", "Specifies which visualizations should be generated"
    )
    visualization_group.add_argument(
        "--visualize_topk",
        action="store_true",
        help="""Flag that indicates whether to visualize the top k
            activations of each prototype from test set.""",
    )
    visualization_group.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="""Flag that indicates whether to visualize the predictions
            on test data and the learned prototypes.""",
    )

    evaluation_group = parser.add_argument_group(
        "Evaluation", "Specifies which evaluation metrics should be calculated"
    )
    evaluation_group.add_argument(
        "--evaluate_purity",
        action="store_true",
        help="""Flag that indicates whether to evaluate purity of prototypes.
            prototype purity is a metric for measuring the overlap between
            the position of learned prototypes and labeled feature centers
            in the image space. Currently is measureable only on CUB-200-2011.""",
    )
    evaluation_group.add_argument(
        "--evaluate_ood",
        action="store_true",
        help="""Flag that indicates whether to evaluate OoD detection
            on other datasets than train set.""",
    )
    evaluation_group.add_argument(
        "--extra_test_image_folder",
        type=str,
        default="./experiments",
        help="""Folder with images that PIP-Net will predict and explain,
            that are not in the training or test set.
            E.g. images with 2 objects or OOD image.
            Images should be in subfolder.
            E.g. images in ./experiments/images/, and argument --./experiments""",
    )

    general_group = parser.add_argument_group("General", "Specifies general arguments")
    general_group.add_argument(
        "--seed",
        type=int,
        default=1,
        help="""Random seed. Note that there will still be differences between runs
            due to nondeterminism.
            See https://pytorch.org/docs/stable/notes/randomness.html""",
    )
    general_group.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="ID of gpu. Can be separated with comma",
    )
    general_group.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers in dataloaders.",
    )

    args, _ = parser.parse_known_args()

    if args.image_height is None and args.image_width is None:
        parser.error("Both image_height and image_width cannot be None")

    args.image_height = args.image_height or args.image_width
    args.image_width = args.image_width or args.image_height

    args.image_shape = np.array((args.image_height, args.image_width))

    if not args.tanh_loss and not args.unif_loss and not args.variance_loss:
        args.tanh_loss = True

    args.log_dir = Path(get_env("LOG_ROOT")) / "pipnet"
    args.log_dir = args.log_dir / args.net / dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    return args


def save_args(args: argparse.Namespace, directory_path: Path) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exist, create it
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
    # Save the args in a text file
    with (directory_path / "args.txt").open(mode="w") as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):
                # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write(f"{arg}: {val}\n")
    # Pickle the args for possible reuse
    with (directory_path / "args.pickle").open(mode="wb") as f:
        pickle.dump(args, f)

import argparse
import warnings
from pathlib import Path

import numpy as np

from models.args import ModelArgumentParser
from utils.args.general import GeneralModelParametersParser


def define_parser():
    parser = argparse.ArgumentParser(
        "Train ProtoPNet",
        description="Necessary parameters to train a ProtoPNet",
        parents=[GeneralModelParametersParser(add_help=False)],
    )

    net_group = parser.add_mutually_exclusive_group()
    net_group.add_argument(
        "--net",
        type=str,
        default="resnet18",
        help="Base network used as backbone of ProtoPNet. Default is resnet18. "
        "Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, "
        "resnet152, vgg13, vgg16 and vgg19.",
    )
    net_group.add_argument(
        "--state_dict_dir_net",
        type=Path,
        help="The directory containing a state dict with a pretrained ProtoPNet. "
        "E.g., ./runs/ProtoPNet/<run_name>/checkpoints/net_pretrained",
    )
    net_group.add_argument(
        "--backbone_only",
        action="store_true",
        help="Flag that indicates whether to train only the backbone network.",
    )

    net_parameter_group = parser.add_argument_group(
        "Network parameters", "Specifies the used network's hyperparameters"
    )
    net_parameter_group.add_argument(
        "--batch_size",
        type=np.uint16,
        default=16,
        help="Batch size when training the model using minibatch gradient descent. "
        "Batch size is multiplied with number of available GPUs",
    )
    net_parameter_group.add_argument(
        "--batch_size_push",
        type=np.uint16,
        default=32,
        help="Batch size when pushing the prototypes to the feature space",
    )
    net_parameter_group.add_argument(
        "--epochs",
        type=np.uint16,
        default=60,
        help="The number of epochs ProtoPNet should be trained (second training stage)",
    )
    net_parameter_group.add_argument(
        "--epochs_warm",
        type=np.uint16,
        default=10,
        help="Number of epochs to pre-train the prototypes (first training stage). "
        "Recommended to train at least until the align loss < 1",
    )
    net_parameter_group.add_argument(
        "--epochs_finetune",
        type=np.uint16,
        default=3,
        help="During fine-tuning, only train classification layer and freeze rest. "
        "Usually done for a few epochs (at least 1, more depends "
        "on size of dataset)",
    )
    push_parameter_group = net_parameter_group.add_argument_group(
        "Push parameters", "Parameters of the push phase."
    )
    push_parameter_group.add_argument(
        "--push_start",
        type=np.uint16,
        default=12,
        help="Epoch when the push phase starts. The push phase is the phase "
        "where the prototypes are pushed to the feature space.",
    )
    push_parameter_group.add_argument(
        "--push_interval",
        type=np.uint16,
        default=5,
        help="Interval in epochs between two push phases.",
    )
    prototype_parameter_group = net_parameter_group.add_argument_group(
        "Prototype parameters", "Parameters of the prototypes"
    )
    prototype_parameter_group.add_argument(
        "--n_prototypes_per_class",
        type=int,
        default=10,
        help="Number of prototypes per class.",
    )
    prototype_parameter_group.add_argument(
        "--prototype_depth",
        type=int,
        default=256,
        help="Depth of the prototypes."
    )
    prototype_parameter_group.add_argument(
        "--prototype_activation_function",
        type=str,
        default="log",
        help="Activation function for the prototypes.",
    )
    net_parameter_group.add_argument(
        "--add_on_layers_type",
        type=str,
        default="regular",
        choices=["regular", "bottleneck"],
        help="Type of add-on layer to use.",
    )
    net_parameter_group.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="When set, the backbone network is initialized with random weights "
        "instead of being pretrained on another dataset).",
    )
    net_parameter_group.add_argument(
        "--bias",
        action="store_true",
        help="Flag that indicates whether to include a trainable bias in "
        "the linear classification layer.",
    )

    optimizer_group = parser.add_argument_group(
        "Optimizer", "Specifies the optimizer to use and its hyperparameters"
    )
    optimizer_group.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer that should be used when training ProtoPNet",
    )
    warm_parameters_group = optimizer_group.add_argument_group(
        "Warm phase", "Optimizer parameters for the warm phase."
    )
    warm_parameters_group.add_argument(
        "--warm_lr_add_on_layers",
        type=float,
        default=3e-3,
        help="The optimizer learning rate for add-on layers in the warm phase.",
    )
    warm_parameters_group.add_argument(
        "--warm_lr_prototype_vectors",
        type=float,
        default=3e-3,
        help="The optimizer learning rate for prototype vectors in the warm phase.",
    )

    joint_parameters_group = optimizer_group.add_argument_group(
        "Joint phase", "Optimizer parameters for the joint phase."
    )
    joint_parameters_group.add_argument(
        "--joint_lr_features",
        type=float,
        default=1e-4,
        help="The optimizer learning rate for feature layers in the joint phase.",
    )
    joint_parameters_group.add_argument(
        "--joint_lr_add_on_layers",
        type=float,
        default=3e-3,
        help="The optimizer learning rate for add-on layers in the joint phase.",
    )
    joint_parameters_group.add_argument(
        "--joint_lr_prototype_vectors",
        type=float,
        default=3e-3,
        help="The optimizer learning rate for prototype vectors in the joint phase.",
    )
    joint_parameters_group.add_argument(
        "--joint_lr_step",
        type=np.uint16,
        default=5,
        help="The step size for the learning rate scheduler in the joint phase.",
    )
    finetune_parameter_group = optimizer_group.add_argument_group(
        "Fine-tune phase", "Optimizer parameters for the fine-tuning phase."
    )
    finetune_parameter_group.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-4,
        help="The optimizer learning rate for the fine-tuning phase.",
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
        help="Flag that weights the loss based on the class balance of the dataset. "
        "Recommended to use when data is imbalanced.",
    )
    loss_group.add_argument(
        "--separation_type",
        type=str,
        default="max",
        choices=["max", "avg", "margin"],
        help="Type of separation loss to use.",
    )
    loss_group.add_argument(
        "--binary_cross_entropy",
        action="store_true",
        help="Flag that indicates whether to use binary cross entropy loss.",
    )
    coefficient_group = loss_group.add_argument_group(
        "Loss coefficients", "Coefficients of the different loss terms."
    )
    coefficient_group.add_argument(
        "--coefficient_cross_entropy",
        type=int,
        default=1,
        help="Coefficient for the cross entropy loss.",
    )
    coefficient_group.add_argument(
        "--coefficient_clustering",
        type=float,
        default=8e-1,
        help="Coefficient for the clustering cost.",
    )
    coefficient_group.add_argument(
        "--coefficient_separation",
        type=float,
        default=6e-1,
        help="Coefficient for the separation cost.",
    )
    coefficient_group.add_argument(
        "--coefficient_separation_margin",
        type=float,
        default=1,
        help="Coefficient for the separation margin cost.",
    )
    coefficient_group.add_argument(
        "--coefficient_l1",
        type=float,
        default=1e-4,
        help="Coefficient for the l1 loss.",
    )
    coefficient_group.add_argument(
        "--coefficient_l2",
        type=float,
        default=1e-2,
        help="Coefficient for the l2 loss.",
    )

    log_group = parser.add_argument_group(
        "Logging",
        "Specifies the directory where the log files and other outputs should be saved",
    )
    log_group.add_argument(
        "--prototype_img_filename_prefix",
        type=str,
        default="prototype-img",
        help="Prefix for the prototype images.",
    )
    log_group.add_argument(
        "--prototype_self_act_filename_prefix",
        type=str,
        default="prototype-self-act",
        help="Prefix for the prototype self activations.",
    )
    log_group.add_argument(
        "--proto_bound_boxes_filename_prefix",
        type=str,
        default="bb",
        help="Prefix for the prototype images with bounding box.",
    )
    log_group.add_argument(
        "--weight_matrix_filename",
        type=str,
        default="outputL_weights",
        help="Filename for the weight matrix.",
    )

    visualization_group = parser.add_argument_group(
        "Visualization", "Specifies which visualizations should be generated"
    )
    visualization_group.add_argument(
        "--visualize_topk",
        action="store_true",
        help="Flag that indicates whether to visualize the top "
        "k activations of each prototype from test set.",
    )
    visualization_group.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="Flag that indicates whether to visualize the predictions "
        "on test data and the learned prototypes.",
    )
    return parser


class ProtoPNetArgumentParser(ModelArgumentParser):
    _parser = define_parser()

    @classmethod
    def get_args(cls, known_args_only: bool = True) -> argparse.Namespace:
        """
        Parse the arguments for the model.

        :param known_args_only: If ``True``, only known arguments are parsed.
            Defaults to ``True``.
        :return: specified arguments in the command line
        """
        super().get_args()
        GeneralModelParametersParser.validate_data(
            cls._args, "ProtoPNet", cls._args.net
        )

        cls._args.prototype_shape = (
            cls._args.n_prototypes_per_class * cls._args.num_classes,
            cls._args.prototype_depth,
            1,
            1,
        )

        if cls._args.backbone_only:
            if cls._args.epochs_warm > 0:
                warnings.warn(
                    "Training backbone model consists only in joint phase. "
                    "Setting number of warm epochs to 0."
                )
                cls._args.epochs_warm = 0

        cls._args.n_epochs = cls._args.epochs + cls._args.epochs_warm

        if not cls._args.backbone_only:
            if cls._args.push_start <= cls._args.epochs_warm:
                warnings.warn(
                    f"Push start epoch ({cls._args.push_start}) is before the end of "
                    f"warm phase ({cls._args.epochs_warm}). Push start modified to "
                    f"{cls._args.epochs_warm + cls._args.push_start}"
                )
                cls._args.push_start = cls._args.epochs_warm + cls._args.push_start
            # define the epochs where the prototypes are pushed to the feature space
            cls._args.push_epochs = np.arange(
                cls._args.push_start, cls._args.n_epochs, cls._args.push_interval
            )

            cls._args.push_epochs = set(cls._args.push_epochs)
            cls._args.n_epochs += len(cls._args.push_epochs) * cls._args.epochs_finetune
            cls._args.push_epochs.add(
                cls._args.n_epochs
            )  # add the last epoch to the push epochs

            cls._args.warm_optimizer_lrs = {
                "add_on_layers": cls._args.warm_lr_add_on_layers,
                "prototype_vectors": cls._args.warm_lr_prototype_vectors,
            }
        cls._args.joint_optimizer_lrs = {
            "features": cls._args.joint_lr_features,
            "add_on_layers": cls._args.joint_lr_add_on_layers,
            "prototype_vectors": cls._args.joint_lr_prototype_vectors,
        }

        cls._args.coefs = {
            "crs_ent": cls._args.coefficient_cross_entropy,
            "clst": cls._args.coefficient_clustering,
            # for margin mode use positive coefficient (e.g. 8e-1)
            "sep": cls._args.coefficient_separation,
            "sep_margin": cls._args.coefficient_separation_margin,
            "l1": cls._args.coefficient_l1,
            "l2": cls._args.coefficient_l2,
        }

        if cls._args.backbone_only:
            cls._args.coefs["sep"] = 0
            cls._args.coefs["sep_margin"] = 0
            cls._args.coefs["clst"] = 0
            cls._args.coefs["l2"] = 0

        return cls._args

import random
from pathlib import Path

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from captum.attr import IntegratedGradients, InputXGradient

import models.PIPNet.pipnet as model_pipnet
import models.ProtoPNet.model as model_ppnet
from data.funny_birds import FunnyBirds
from evaluation.explainer_wrapper.captum import CaptumAttributionExplainer
from evaluation.explainer_wrapper.PIPNet import PIPNetExplainer
from evaluation.model_wrapper.base import AbstractModel
from evaluation.model_wrapper.standard import StandardModel
from evaluation.explainer_wrapper.ProtoPNet import ProtoPNetExplainer
from evaluation.model_wrapper.PIPNet import PipNetModel
from evaluation.model_wrapper.ProtoPNet import ProtoPNetModel
from utils.file_operations import get_package
from evaluation.protocols import (
    accuracy_protocol,
    background_independence_protocol,
    controlled_synthetic_data_check_protocol,
    deletion_check_protocol,
    distractibility_protocol,
    preservation_check_protocol,
    single_deletion_protocol,
    target_sensitivity_protocol,
)
from models.resnet import resnet50
from models.vgg import vgg16
from utils.environment import get_env
from utils.log import BasicLog

parser = ArgumentParser(description="FunnyBirds - Explanation Evaluation")
parser.add_argument(
    "--data_subset",
    choices=["train", "test"],
    default="test",
    type=str,
    help="Subset of data to use."
)
parser.add_argument(
    "--model",
    required=True,
    choices=["resnet50", "vgg16", "vit_b_16", "ProtoPNet", "PIPNet"],
    help="model architecture",
)
parser.add_argument(
    "--explainer",
    required=True,
    choices=[
        "IntegratedGradients",
        "InputXGradient",
        "ProtoPNet",
        "PIPNet",
    ],
    help="explainer",
)
parser.add_argument(
    "--checkpoint_path",
    type=Path,
    required=False,
    default=None,
    help="path to trained model checkpoint",
)
parser.add_argument(
    "--epoch_number",
    type=int,
    required=False,
    help="Epoch number of model checkpoint to use.",
)
parser.add_argument(
    "--nr_itrs",
    default=2501,
    type=int,
    help="batch size for protocols that do not require custom BS such as accuracy",
)

parser.add_argument(
    "--accuracy", default=False, action="store_true", help="compute accuracy"
)
parser.add_argument(
    "--controlled_synthetic_data_check",
    default=False,
    action="store_true",
    help="compute controlled synthetic data check",
)
parser.add_argument(
    "--single_deletion",
    default=False,
    action="store_true",
    help="compute single deletion",
)
parser.add_argument(
    "--preservation_check",
    default=False,
    action="store_true",
    help="compute preservation check",
)
parser.add_argument(
    "--deletion_check",
    default=False,
    action="store_true",
    help="compute deletion check",
)
parser.add_argument(
    "--target_sensitivity",
    default=False,
    action="store_true",
    help="compute target sensitivity",
)
parser.add_argument(
    "--distractibility",
    default=False,
    action="store_true",
    help="compute distractibility",
)
parser.add_argument(
    "--background_independence",
    default=False,
    action="store_true",
    help="compute background dependence",
)
parser.add_argument(
    "--enable_console", action="store_true", help="Enable console output"
)


def create_model(args: Namespace, log: BasicLog):
    if args.model == "resnet50":
        model = resnet50(num_classes=50)
        model = StandardModel(model)
    elif args.model == "vgg16":
        model = vgg16(num_classes=50)
        model = StandardModel(model)
    elif args.model == "ProtoPNet":
        load_model_dir = args.checkpoint_path.parent

        log.warning("REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH")
        if args.epoch_number is None:
            raise parser.error("Epoch number is required for ProtoPNet")
        model = model_ppnet.construct_PPNet(
            base_architecture=args.net,
            backbone_only=args.backbone_only,
            pretrained=not args.disable_pretrained,
            img_shape=args.img_shape,
            prototype_shape=args.prototype_shape,
            num_classes=args.num_classes,
            prototype_activation_function=args.prototype_activation_function,
            add_on_layers_type=args.add_on_layers_type,
        )
        model = nn.DataParallel(model, device_ids=list(map(int, args.gpu_ids)))
        model = ProtoPNetModel(model, load_model_dir, args.epoch_number)
    elif args.model == "PIPNet":
        num_classes = args.num_classes
        (
            feature_net,
            add_on_layers,
            pool_layer,
            classification_layer,
            num_prototypes,
        ) = model_pipnet.get_network(num_classes, args)

        # Create a PIP-Net
        model = model_pipnet.PIPNet(
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            feature_net=feature_net,
            args=args,
            add_on_layers=add_on_layers,
            pool_layer=pool_layer,
            classification_layer=classification_layer,
        )
        model = nn.DataParallel(model, device_ids=list(map(int, args.gpu_ids)))
        model = PipNetModel(model)
    else:
        raise NotImplementedError(f"Model {args.model!r} not implemented")

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
        if type(state_dict) is not dict:
            state_dict.to(args.device)
            model.model.module = state_dict
        else:
            model.load_state_dict(state_dict["model_state_dict"])
    if type(model.model) is not nn.DataParallel:
        model.to(args.device)
    model.eval()

    return model


def create_explainer(args: Namespace, model: AbstractModel):
    match args.explainer:
        case "InputXGradient":
            explainer = InputXGradient(model)
            return CaptumAttributionExplainer(explainer)
        case "IntegratedGradients":
            explainer = IntegratedGradients(model)
            baseline = torch.zeros((1, 3, 256, 256)).to(args.device)
            return CaptumAttributionExplainer(explainer, baseline=baseline)
        case "ProtoPNet":
            return ProtoPNetExplainer(model)
        case "PIPNet":
            return PIPNetExplainer(model)
        case _:
            raise NotImplementedError("Explainer not implemented")


def main(args: Namespace, log: BasicLog):
    model = create_model(args, log)
    explainer = create_explainer(args, model)

    bathed_dataset = FunnyBirds(args.data_dir, args.data_subset)
    partmap_dataset = FunnyBirds(args.data_dir, args.data_subset, get_part_map=True)
    bathed_dataloader = DataLoader(bathed_dataset, batch_size=int(args.batch_size), shuffle=False)
    partmap_dataloader = DataLoader(partmap_dataset, batch_size=1, shuffle=False)

    accuracy, background_independence, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        log.info("Computing accuracy...")
        accuracy = accuracy_protocol(model, bathed_dataloader, args, log)
        accuracy = round(accuracy, 5)

    if args.controlled_synthetic_data_check:
        log.info("Computing controlled synthetic data check...")
        csdc = controlled_synthetic_data_check_protocol(
            model, partmap_dataloader, explainer, args, log
        )

    if args.target_sensitivity:
        log.info("Computing target sensitivity...")
        ts = target_sensitivity_protocol(
            model, partmap_dataloader, explainer, args, log
        )
        ts = round(ts, 5)

    if args.single_deletion:
        log.info("Computing single deletion...")
        sd = single_deletion_protocol(model, partmap_dataloader, explainer, args, log)
        sd = round(sd, 5)

    if args.preservation_check:
        log.info("Computing preservation check...")
        pc = preservation_check_protocol(
            model, partmap_dataloader, explainer, args, log
        )

    if args.deletion_check:
        log.info("Computing deletion check...")
        dc = deletion_check_protocol(model, partmap_dataloader, explainer, args, log)

    if args.distractibility:
        log.info("Computing distractibility...")
        distractibility = distractibility_protocol(
            model, partmap_dataloader, explainer, args, log
        )

    if args.background_independence:
        log.info("Computing background independence...")
        background_independence = background_independence_protocol(
            model, partmap_dataloader, args, log
        )
        background_independence = round(background_independence, 5)

    # select completeness and distractability thresholds
    # such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = (
            csdc[threshold] / 3.0
            + pc[threshold] / 3.0
            + dc[threshold] / 3.0
            + distractibility[threshold]
        )
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    log.info("FINAL RESULTS:")
    log.info("Accuracy, Background independence, CSDC, PC, DC, Distractability, SD, TS")
    log.info(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            accuracy,
            background_independence,
            round(csdc[best_threshold], 5),
            round(pc[best_threshold], 5),
            round(dc[best_threshold], 5),
            round(distractibility[best_threshold], 5),
            sd,
            ts,
        )
    )
    log.info(f"Best threshold: {best_threshold}")


if __name__ == "__main__":
    args, _ = parser.parse_known_args()

    match args.model:
        case "ProtoPNet":
            from models.ProtoPNet.util.args import (
                ProtoPNetArgumentParser as ModelArgumentParser,
            )
        case "PIPNet":
            from models.PIPNet.util.args import \
                PIPNetArgumentParser as ModelArgumentParser
        case _:
            raise ValueError(f"Unknown model: {args.model}")

    model_args = ModelArgumentParser.get_args()

    all_args = Namespace(**vars(args), **vars(model_args))

    results_location = get_env("RESULTS_LOCATION", must_exist=False) or get_env(
        "PROJECT_ROOT")
    dir_name = f"{all_args.model}-{all_args.net}"
    if all_args.checkpoint_path:
        to_output_dir = all_args.checkpoint_path
        if to_output_dir.is_file():
            to_output_dir = to_output_dir.parent
        dir_name += f"-{to_output_dir.name}"

    all_args.log_dir = Path(results_location, "runs", get_package(__file__), dir_name)
    all_args.log_dir = all_args.log_dir.resolve()

    # Create a logger
    log = BasicLog(all_args.log_dir, __name__, not args.enable_console)

    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)

    try:
        main(all_args, log)
    except Exception as e:
        log.exception(e)

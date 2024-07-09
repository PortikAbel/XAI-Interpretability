import random
import re
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.funny_birds import FunnyBirds
from evaluation.explainer_wrapper.ProtoPNet import ProtoPNetExplainer
from evaluation.model_wrapper.ProtoPNet import ProtoPNetModel
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
from models.ProtoPNet.model import construct_PPNet
from utils.environment import get_env
from utils.file_operations import get_package
from utils.log import BasicLog

parser = ArgumentParser(description="FunnyBirds - Explanation Evaluation")
parser.add_argument(
    "--data_subset",
    choices=["train", "test"],
    default="test",
    type=str,
    help="Subset of data to use.",
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


def evaluate(checkpoint_path, epoch, args, log):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create model
    if args.model == "ProtoPNet":
        load_model_dir = args.checkpoint_path

        log.warning("REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH")
        model = construct_PPNet(
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
        model = ProtoPNetModel(model, load_model_dir, epoch)
    else:
        raise NotImplementedError(f"Model {args.model!r} not implemented")

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if type(state_dict) is not dict:
            state_dict.to(args.device)
            model.model.module = state_dict
        else:
            model.load_state_dict(state_dict["model_state_dict"])
    if type(model.model) is not nn.DataParallel:
        model.to(args.device)
    model.eval()

    # create explainer
    if args.explainer == "ProtoPNet":
        explainer = ProtoPNetExplainer(model)
    else:
        raise NotImplementedError("Explainer not implemented")

    bathed_dataset = FunnyBirds(args.data_dir, args.data_subset)
    partmap_dataset = FunnyBirds(args.data_dir, args.data_subset, get_part_map=True)
    bathed_dataloader = DataLoader(
        bathed_dataset, batch_size=int(args.batch_size), shuffle=False
    )
    partmap_dataloader = DataLoader(partmap_dataset, batch_size=1, shuffle=False)

    accuracy, background_independence, csdc, pc, dc, distractibility, sd, ts = (
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    )

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

    # select completeness and distractibility thresholds
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

    del explainer
    del model

    return [
        accuracy,
        csdc[best_threshold],
        pc[best_threshold],
        dc[best_threshold],
        distractibility[best_threshold],
        background_independence,
        sd,
        ts,
        best_threshold,
    ]


def main():
    args, _ = parser.parse_known_args()

    match args.model:
        case "ProtoPNet":
            from models.ProtoPNet.util.args import (
                ProtoPNetArgumentParser as ModelArgumentParser,
            )
        case _:
            raise ValueError(f"Unknown model: {args.model}")

    model_args = ModelArgumentParser.get_args()

    all_args = Namespace(**vars(args), **vars(model_args))

    # Create a logger
    results_location = get_env("RESULTS_LOCATION", must_exist=False) or get_env(
        "PROJECT_ROOT"
    )
    log_dir = Path(
        results_location,
        "runs",
        get_package(__file__),
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    log = BasicLog(log_dir, __name__, not args.enable_console)

    try:
        checkpoints = [
            x
            for x in args.checkpoint_path.iterdir()
            if x.is_file() and x.suffix == ".pth" and x.name[0].isdigit()
        ]
        checkpoints.sort(
            key=lambda cp: int(re.match(r"^([0-9]+)_.*", cp.stem).group(1))
        )

        data_table = pd.DataFrame(
            columns=[
                "accuracy",
                "csdc",
                "pc",
                "dc",
                "distractibility",
                "background_independence",
                "sd",
                "ts",
                "threshold",
            ],
            index=pd.Index([], name="epoch"),
        )

        after_push = False
        epoch = None
        last_push_epoch = 0
        for cp in checkpoints:
            current_epoch = int(re.match(r"([0-9]+)_.*", cp.stem).group(1))
            if "_push_" in cp.stem:
                after_push = True
                last_push_epoch = current_epoch
                continue
            elif after_push:
                if current_epoch != epoch:
                    log.info(f"{cp} {last_push_epoch}")
                    data_table.loc[current_epoch] = evaluate(
                        cp, last_push_epoch, all_args, log
                    )
                    epoch = current_epoch
                    data_table.to_csv("metrics.csv")
    except Exception as e:
        log.exception(e)


if __name__ == "__main__":
    main()

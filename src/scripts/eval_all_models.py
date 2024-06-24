import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

import pandas as pd
import torch
from torch import nn

from evaluation.explainer_wrapper.ProtoPNet import ProtoPNetExplainer
from evaluation.model_wrapper.ProtoPNet import ProtoPNetModel
from evaluation.protocols import accuracy_protocol, \
    controlled_synthetic_data_check_protocol, target_sensitivity_protocol, \
    single_deletion_protocol, preservation_check_protocol, distractibility_protocol, \
    deletion_check_protocol, background_independence_protocol
from models.ProtoPNet.model import construct_PPNet

parser = ArgumentParser(description="FunnyBirds - Explanation Evaluation")
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
    "--accuracy", default=False, action="store_true",
    help="compute accuracy"
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


def evaluate(checkpoint_path, epoch, args):
    match args.model:
        case "ProtoPNet":
            from models.ProtoPNet.util.args import (
                ProtoPNetArgumentParser as ModelArgumentParser,
            )
        case _:
            raise ValueError(f"Unknown model: {args.model}")

    model_args = ModelArgumentParser.get_args()

    all_args = Namespace(**vars(args), **vars(model_args))

    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)

    # create model
    if all_args.model == "ProtoPNet":
        load_model_dir = all_args.checkpoint_path.parent

        print("REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH")
        model = construct_PPNet(
            base_architecture=all_args.net,
            backbone_only=all_args.backbone_only,
            pretrained=not all_args.disable_pretrained,
            img_shape=all_args.img_shape,
            prototype_shape=all_args.prototype_shape,
            num_classes=all_args.num_classes,
            prototype_activation_function=all_args.prototype_activation_function,
            add_on_layers_type=all_args.add_on_layers_type,
        )
        model = nn.DataParallel(model,
                                device_ids=list(map(int, all_args.gpu_ids)))
        model = ProtoPNetModel(model, load_model_dir, epoch)
    else:
        raise NotImplementedError(
            f"Model {all_args.model!r} not implemented")

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        if type(state_dict) is not dict:
            state_dict.to(all_args.device)
            model.model.module = state_dict
        else:
            model.load_state_dict(state_dict["model_state_dict"])
    if type(model.model) is not nn.DataParallel:
        model.to(all_args.device)
    model.eval()

    # create explainer
    if all_args.explainer == "ProtoPNet":
        explainer = ProtoPNetExplainer(model)
    else:
        raise NotImplementedError("Explainer not implemented")

    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if all_args.accuracy:
        print("Computing accuracy...")
        accuracy = accuracy_protocol(model, all_args)
        accuracy = round(accuracy, 5)

    if all_args.controlled_synthetic_data_check:
        print("Computing controlled synthetic data check...")
        csdc = controlled_synthetic_data_check_protocol(model, explainer,
                                                        all_args)

    if all_args.target_sensitivity:
        print("Computing target sensitivity...")
        ts = target_sensitivity_protocol(model, explainer, all_args)
        ts = round(ts, 5)

    if all_args.single_deletion:
        print("Computing single deletion...")
        sd = single_deletion_protocol(model, explainer, all_args)
        sd = round(sd, 5)

    if all_args.preservation_check:
        print("Computing preservation check...")
        pc = preservation_check_protocol(model, explainer, all_args)

    if all_args.deletion_check:
        print("Computing deletion check...")
        dc = deletion_check_protocol(model, explainer, all_args)

    if all_args.distractibility:
        print("Computing distractibility...")
        distractibility = distractibility_protocol(model, explainer,
                                                   all_args)

    if all_args.background_independence:
        print("Computing background independence...")
        background_independence = background_independence_protocol(model,
                                                                   all_args)
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

    return [accuracy, csdc[best_threshold], pc[best_threshold],
            dc[best_threshold], distractibility[best_threshold],
            background_independence, sd, ts, best_threshold]


def main():
    args, _ = parser.parse_known_args()

    p = Path(
        "/tankstorage/user_data/adel.bajcsi/results/Funny/runs/ProtoPNet/vgg16/2024-06-19-00-39-28/checkpoints/")

    checkpoints = [x for x in p.iterdir() if
                   x.is_file() and x.suffix == ".pth" and x.name[0].isdigit()]

    checkpoints.sort(key=lambda cp: int(re.match(r"([0-9]{2})_.*", cp.stem).group(1)))

    data_table = pd.read_csv("metrics.csv", header=0, index_col=0)

    after_push = True
    epoch = None
    last_push_epoch = 62
    for cp in checkpoints:
        current_epoch = int(re.match(r"([0-9]{2})_.*", cp.stem).group(1))
        if current_epoch <= 62:
            epoch = current_epoch
            continue
        if "_push_" in cp.stem:
            after_push = True
            last_push_epoch = current_epoch
            continue
        elif after_push:
            if current_epoch != epoch:
                print(cp, last_push_epoch)
                data_table.loc[current_epoch] = evaluate(cp, last_push_epoch, args)
                epoch = current_epoch
                data_table.to_csv("metrics.csv")


if __name__ == "__main__":
    main()

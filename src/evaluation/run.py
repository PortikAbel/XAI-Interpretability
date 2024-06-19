import random
from pathlib import Path

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from captum.attr import InputXGradient, IntegratedGradients

import models.PIPNet.pipnet as model_pipnet
import models.ProtoPNet.model as model_ppnet
from evaluation.explainer_wrapper.captum import CaptumAttributionExplainer
from evaluation.explainer_wrapper.PIPNet import PIPNetExplainer
from evaluation.explainer_wrapper.ProtoPNet import ProtoPNetExplainer
from evaluation.model_wrapper.PIPNet import PipNetModel
from evaluation.model_wrapper.ProtoPNet import ProtoPNetModel
from evaluation.model_wrapper.standard import StandardModel
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


def main():
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

    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)

    # create model
    if all_args.model == "resnet50":
        model = resnet50(num_classes=50)
        model = StandardModel(model)
    elif all_args.model == "vgg16":
        model = vgg16(num_classes=50)
        model = StandardModel(model)
    elif all_args.model == "ProtoPNet":
        load_model_dir = all_args.checkpoint_path.parent

        print("REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH")
        if all_args.epoch_number is None:
            raise parser.error("Epoch number is required for ProtoPNet")
        model = model_ppnet.construct_PPNet(
            base_architecture=all_args.net,
            backbone_only=all_args.backbone_only,
            pretrained=not all_args.disable_pretrained,
            img_shape=all_args.img_shape,
            prototype_shape=all_args.prototype_shape,
            num_classes=all_args.num_classes,
            prototype_activation_function=all_args.prototype_activation_function,
            add_on_layers_type=all_args.add_on_layers_type,
        )
        model = nn.DataParallel(model, device_ids=list(map(int, all_args.gpu_ids)))
        model = ProtoPNetModel(model, load_model_dir, all_args.epoch_number)
    elif all_args.model == "PIPNet":
        num_classes = all_args.num_classes
        (
            feature_net,
            add_on_layers,
            pool_layer,
            classification_layer,
            num_prototypes,
        ) = model_pipnet.get_network(num_classes, all_args)

        # Create a PIP-Net
        model = model_pipnet.PIPNet(
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            feature_net=feature_net,
            args=all_args,
            add_on_layers=add_on_layers,
            pool_layer=pool_layer,
            classification_layer=classification_layer,
        )
        model = nn.DataParallel(model, device_ids=list(map(int, all_args.gpu_ids)))
        model = PipNetModel(model)
    else:
        raise NotImplementedError(f"Model {all_args.model!r} not implemented")

    if all_args.checkpoint_path:
        state_dict = torch.load(all_args.checkpoint_path, map_location=torch.device("cpu"))
        if type(state_dict) is not dict:
            state_dict.to(all_args.device)
            model.model.module = state_dict
        else:
            model.load_state_dict(state_dict["model_state_dict"])
    if type(model.model) is not nn.DataParallel:
        model.to(all_args.device)
    model.eval()

    # create explainer
    if all_args.explainer == "InputXGradient":
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif all_args.explainer == "IntegratedGradients":
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1, 3, 256, 256)).to(all_args.device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif all_args.explainer == "ProtoPNet":
        explainer = ProtoPNetExplainer(model)
    elif all_args.explainer == "PIPNet":
        explainer = PIPNetExplainer(model)
    else:
        raise NotImplementedError("Explainer not implemented")

    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if all_args.accuracy:
        print("Computing accuracy...")
        accuracy = accuracy_protocol(model, all_args)
        accuracy = round(accuracy, 5)

    if all_args.controlled_synthetic_data_check:
        print("Computing controlled synthetic data check...")
        csdc = controlled_synthetic_data_check_protocol(model, explainer, all_args)

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
        distractibility = distractibility_protocol(model, explainer, all_args)

    if all_args.background_independence:
        print("Computing background independence...")
        background_independence = background_independence_protocol(model, all_args)
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

    print("FINAL RESULTS:")
    print("Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS")
    print(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            accuracy,
            round(csdc[best_threshold], 5),
            round(pc[best_threshold], 5),
            round(dc[best_threshold], 5),
            round(distractibility[best_threshold], 5),
            background_independence,
            sd,
            ts,
        )
    )
    print("Best threshold:", best_threshold)


if __name__ == "__main__":
    main()

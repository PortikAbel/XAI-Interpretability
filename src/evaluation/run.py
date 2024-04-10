import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from captum.attr import InputXGradient, IntegratedGradients

import models.PIPNet.pipnet as model_pipnet
import models.ProtoPNet.model as model_ppnet
from data.config import dataset_config
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
from models.PIPNet.util.args import get_args as get_pipnet_args
from models.resnet import resnet50
from models.vgg import vgg16

parser = argparse.ArgumentParser(description="FunnyBirds - Explanation Evaluation")
parser.add_argument(
    "--data", type=Path, required=True, help="path to dataset (default: imagenet)"
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
    "--epoch_number",
    type=int,
    required=False,
    help="Epoch number of model checkpoint to use.",
)
parser.add_argument(
    "--checkpoint_path",
    type=Path,
    required=False,
    default=None,
    help="path to trained model checkpoint",
),

parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="batch size for protocols that do not require custom BS such as accuracy",
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
    device = "cuda:" + str(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create model
    if args.model == "resnet50":
        model = resnet50(num_classes=50)
        model = StandardModel(model)
    elif args.model == "vgg16":
        model = vgg16(num_classes=50)
        model = StandardModel(model)
    elif args.model == "ProtoPNet":
        base_architecture = "resnet50"
        img_size = 256
        prototype_shape = (50 * 10, 128, 1, 1)
        num_classes = dataset_config["num_classes"]
        prototype_activation_function = "log"
        add_on_layers_type = "regular"
        load_model_dir = args.checkpoint_path.parent
        epoch_number = args.epoch_number

        print("REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH")
        model = model_ppnet.construct_PPNet(
            base_architecture=base_architecture,
            pretrained=True,
            img_shape=img_size,
            prototype_shape=prototype_shape,
            num_classes=num_classes,
            prototype_activation_function=prototype_activation_function,
            add_on_layers_type=add_on_layers_type,
        )
        model = ProtoPNetModel(model, load_model_dir, epoch_number)
    elif args.model == "PIPNet":
        num_classes = dataset_config["num_classes"]
        pipnet_args = get_pipnet_args()
        (
            feature_net,
            add_on_layers,
            pool_layer,
            classification_layer,
            num_prototypes,
        ) = model_pipnet.get_network(num_classes, pipnet_args)

        # Create a PIP-Net
        model = model_pipnet.PIPNet(
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            feature_net=feature_net,
            args=pipnet_args,
            add_on_layers=add_on_layers,
            pool_layer=pool_layer,
            classification_layer=classification_layer,
        )
        model = nn.DataParallel(model, device_ids=list(map(int, pipnet_args.gpu_ids)))
        model = PipNetModel(model)
    else:
        print("Model not implemented")

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict["model_state_dict"])
    model = model.to(device)
    model.eval()

    # create explainer
    if args.explainer == "InputXGradient":
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == "IntegratedGradients":
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1, 3, 256, 256)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif args.explainer == "ProtoPNet":
        explainer = ProtoPNetExplainer(model)
    elif args.explainer == "PIPNet":
        explainer = PIPNetExplainer(model)
    else:
        print("Explainer not implemented")

    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        print("Computing accuracy...")
        accuracy = accuracy_protocol(model, args)
        accuracy = round(accuracy, 5)

    if args.controlled_synthetic_data_check:
        print("Computing controlled synthetic data check...")
        csdc = controlled_synthetic_data_check_protocol(model, explainer, args)

    if args.target_sensitivity:
        print("Computing target sensitivity...")
        ts = target_sensitivity_protocol(model, explainer, args)
        ts = round(ts, 5)

    if args.single_deletion:
        print("Computing single deletion...")
        sd = single_deletion_protocol(model, explainer, args)
        sd = round(sd, 5)

    if args.preservation_check:
        print("Computing preservation check...")
        pc = preservation_check_protocol(model, explainer, args)

    if args.deletion_check:
        print("Computing deletion check...")
        dc = deletion_check_protocol(model, explainer, args)

    if args.distractibility:
        print("Computing distractibility...")
        distractibility = distractibility_protocol(model, explainer, args)

    if args.background_independence:
        print("Computing background independence...")
        background_independence = background_independence_protocol(model, args)
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

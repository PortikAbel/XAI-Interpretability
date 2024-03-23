import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.features import base_architecture_to_features


class PIPNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        feature_net: nn.Module,
        args: argparse.Namespace,
        add_on_layers: nn.Module,
        pool_layer: nn.Module,
        classification_layer: nn.Module,
    ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        self._args = args
        self._init_param_groups()

    def forward(self, xs, inference=False):
        features = self._net(xs)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        if inference:
            clamped_pooled = torch.where(
                pooled < 0.1, 0.0, pooled
            )  # during inference, ignore all prototypes
            # that have 0.1 similarity or lower
            out = self._classification(clamped_pooled)  # shape (bs*2, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled)  # shape (bs*2, num_classes)
            return proto_features, pooled, out
        
    def _init_param_groups(self):
        self.param_groups = dict()
        self.param_groups["backbone"] = []
        self.param_groups["to_train"] = []
        self.param_groups["to_freeze"] = []

        if "resnet" in self._args.net:
            # freeze resnet50 except last convolutional layer
            for name, param in self._net.named_parameters():
                if "layer4.2" in name:
                    self.param_groups["to_train"].append(param)
                elif "layer4" in name or "layer3" in name:
                    self.param_groups["to_freeze"].append(param)
                elif "layer2" in name:
                    self.param_groups["backbone"].append(param)
                else:  # such that model training fits on one gpu.
                    param.requires_grad = False
                    # self.param_groups["backbone"].append(param)

        elif "vgg" in self._args.net:
            import re
            for name, param in self._net.named_parameters():
                if re.match(r"^features.(18|[23]\d)", name):
                    self.param_groups["to_train"].append(param)
                elif re.match(r"^features.1\d", name):
                    self.param_groups["to_freeze"].append(param)
                elif re.match(r"^features.[5-9].", name):
                    self.param_groups["backbone"].append(param)
                else:
                    param.requires_grad = False

        elif "convnext" in self._args.net:
            for name, param in self._net.named_parameters():
                if "features.7.2" in name:
                    self.param_groups["to_train"].append(param)
                elif "features.7" in name or "features.6" in name:
                    self.param_groups["to_freeze"].append(param)
                # CUDA MEMORY ISSUES?
                # COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
                # elif 'features.5' in name or 'features.4' in name:
                #     self.param_groups["backbone"].append(param)
                # else:
                #     param.requires_grad = False
                else:
                    self.param_groups["backbone"].append(param)
        else:
            print("Network is not ResNet/VGG/ConvNext.", flush=True)
        self.param_groups["classification_weight"] = []
        self.param_groups["classification_bias"] = []
        for name, param in self._classification.named_parameters():
            if "weight" in name:
                self.param_groups["classification_weight"].append(param)
            elif "multiplier" in name:
                param.requires_grad = False
            elif self._args.bias:
                self.param_groups["classification_bias"].append(param)

    def get_optimizers(self):
        torch.manual_seed(self._args.seed)
        torch.cuda.manual_seed_all(self._args.seed)
        random.seed(self._args.seed)
        np.random.seed(self._args.seed)

        paramlist_net = [
            {
                "params": self.param_groups["backbone"],
                "lr": self._args.lr_net,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self.param_groups["to_freeze"],
                "lr": self._args.lr_block,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self.param_groups["to_train"],
                "lr": self._args.lr_block,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self._add_on.parameters(),
                "lr": self._args.lr_block * 10.0,
                "weight_decay_rate": self._args.weight_decay,
            },
        ]

        paramlist_classifier = [
            {
                "params": self.param_groups["classification_weight"],
                "lr": self._args.lr,
                "weight_decay_rate": self._args.weight_decay,
            },
            {"params": self.param_groups["classification_bias"], "lr": self._args.lr, "weight_decay_rate": 0},
        ]

        if self._args.optimizer == "Adam":
            optimizer_net = torch.optim.AdamW(
                paramlist_net, lr=self._args.lr, weight_decay=self._args.weight_decay
            )
            optimizer_classifier = torch.optim.AdamW(
                paramlist_classifier, lr=self._args.lr, weight_decay=self._args.weight_decay
            )
            return (
                optimizer_net,
                optimizer_classifier
            )
        else:
            raise ValueError("this optimizer type is not implemented")
        
    def pretrain(self):
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self._add_on.parameters():
            param.requires_grad = True
        for param in self._classification.parameters():
            param.requires_grad = False
        for param in self.param_groups["to_freeze"]:
            param.requires_grad = (
                True  # can be set to False when you want to freeze more layers
            )
        for param in self.param_groups["backbone"]:
            # can be set to True when you want to train whole backbone
            # (e.g. if dataset is very different from ImageNet)
            param.requires_grad = self._args.train_backbone_during_pretrain

    def finetune(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self._classification.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.param_groups["to_freeze"]:
            # Can be set to False if you want
            # to train fewer layers of backbone
            param.requires_grad = True
        for param in self._add_on.parameters():
            param.requires_grad = True
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.param_groups["backbone"]:
            param.requires_grad = False

    def unfreeze(self):
        for param in self._add_on.parameters():
            param.requires_grad = True
        for param in self.param_groups["to_freeze"]:
            param.requires_grad = True
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.param_groups["backbone"]:
            param.requires_grad = True



# adapted from
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad=True)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, torch.relu(self.weight), self.bias)


def get_network(num_classes: int, args: argparse.Namespace):
    in_channels = 3
    features = base_architecture_to_features[args.net](
        pretrained=not args.disable_pretrained, in_channels=in_channels
    )
    first_add_on_layer_in_channels = [
        i for i in features.modules() if isinstance(i, nn.Conv2d)
    ][-1].out_channels

    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1),  # softmax over every prototype for each patch,
            # such that for every location in image, sum over prototypes is 1
        )
    else:
        num_prototypes = args.num_features
        print(
            "Number of prototypes set from",
            first_add_on_layer_in_channels,
            "to",
            num_prototypes,
            ". Extra 1x1 conv layer added. Not recommended.",
            flush=True,
        )
        add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels,
                out_channels=num_prototypes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Softmax(dim=1),  # softmax over every prototype for each patch,
            # such that for every location in image, sum over prototypes is 1
        )
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),  # outputs (bs, ps,1,1)
        nn.Flatten(),  # outputs (bs, ps)
    )

    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)

    return (
        features,
        add_on_layers,
        pool_layer,
        classification_layer,
        num_prototypes,
    )

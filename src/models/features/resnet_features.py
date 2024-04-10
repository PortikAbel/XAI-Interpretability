import copy
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models import pretrained_models_dir
from models.resnet import BasicBlock, Bottleneck, conv1x1, model_urls


class ResNet_features(nn.Module):
    """
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    """

    def __init__(
        self, block, layers, num_classes=1000, zero_init_residual=False, in_channels=3
    ):
        super(ResNet_features, self).__init__()

        self.in_planes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.layer1 = self._make_layer(
            block=block, planes=64, num_blocks=self.layers[0]
        )
        self.layer2 = self._make_layer(
            block=block, planes=128, num_blocks=self.layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block=block, planes=256, num_blocks=self.layers[2], stride=1
        )
        self.layer4 = self._make_layer(
            block=block, planes=512, num_blocks=self.layers[3], stride=1
        )

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.in_planes, planes, stride, downsample))

        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            (
                block_kernel_sizes,
                block_strides,
                block_paddings,
            ) = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        """
        the number of conv layers in the network, not counting the number
        of bypass layers
        """

        return (
            self.block.num_layers * self.layers[0]
            + self.block.num_layers * self.layers[1]
            + self.block.num_layers * self.layers[2]
            + self.block.num_layers * self.layers[3]
            + 1
        )

    def __repr__(self):
        template = "resnet{}_features"
        return template.format(self.num_layers() + 1)


def _resnet_features(
    arch: str,
    layers: list,
    block: nn.Module = BasicBlock,
    pretrained: bool = False,
    **kwargs
):
    model = ResNet_features(block, layers, **kwargs)

    if pretrained:
        my_dict = model_zoo.load_url(model_urls[arch], model_dir=pretrained_models_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        model.load_state_dict(my_dict, strict=False)

    return model


def resnet18_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet_features("resnet18", [2, 2, 2, 2], pretrained=pretrained, **kwargs)


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet_features("resnet34", [3, 4, 6, 3], pretrained=pretrained, **kwargs)


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet_features(
        "resnet50", [3, 4, 6, 3], Bottleneck, pretrained=pretrained, **kwargs
    )


def resnet50_features_inat(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Inaturalist2017
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # use BBN pretrained weights of the conventional learning branch
        #   (from BBN.iNaturalist2017.res50.180epoch.best_model.pth)
        # https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf  # noqa
        if not os.path.exists(
            os.path.join(
                os.path.join("features", "state_dicts"),
                "BBN.iNaturalist2017.res50.180epoch.best_model.pth",
            )
        ):
            print(
                "To use Resnet50 pretrained on iNaturalist"
                "create a folder called state_dicts in the folder features, and"
                "download BBN.iNaturalist2017.res50.180epoch.best_model.pth"
                "to there from https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU.",  # noqa
                flush=True,
            )
        model_dict = torch.load(
            os.path.join(
                os.path.join("features", "state_dicts"),
                "BBN.iNaturalist2017.res50.180epoch.best_model.pth",
            )
        )
        # rename last residual block from cb_block to layer4.2
        new_model = copy.deepcopy(model_dict)
        for k in model_dict.keys():
            if k.startswith("module.backbone.cb_block"):
                splitted = k.split("cb_block")
                new_model["layer4.2" + splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith("module.backbone.rb_block"):
                del new_model[k]
            elif k.startswith("module.backbone."):
                splitted = k.split("backbone.")
                new_model[splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith("module.classifier"):
                del new_model[k]
        model.load_state_dict(new_model, strict=True)
    return model


def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet_features(
        "resnet101", [3, 4, 23, 3], Bottleneck, pretrained=pretrained, **kwargs
    )


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet_features(
        "resnet152", [3, 8, 36, 3], Bottleneck, pretrained=pretrained, **kwargs
    )

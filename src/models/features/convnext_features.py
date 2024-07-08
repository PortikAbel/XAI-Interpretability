import torch
import torch.nn as nn
from torchvision import models

from evaluation.model_wrapper.base import AbstractModel


class ConvNext_features(AbstractModel):
    """
    A wrapper for ProtoPNet models.
    Args:
        model: PyTorch ConvNext model
    """

    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

    def conv_info(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                print("------------------")
                print(name)
                print("Kernel size: ", module.kernel_size[0])
                self.kernel_sizes.append(module.kernel_size[0])
                print("Stride: ", module.stride[0])
                self.strides.append(module.stride[0])
                print("Padding: ", module.padding[0])
                self.paddings.append(module.padding[0])                
        return self.kernel_sizes, self.strides, self.paddings
    
    def forward(self, x):
        x = self.model(x)
        return x


def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if module.in_channels > threshold:
                    # replace bigger strides to reduce receptive field,
                    # skip some 2x2 layers.
                    # >100 gives output size (26, 26).
                    # >300 gives (13, 13)
                    module.stride = tuple(s // 2 for s in module.stride)

    return model


def convnext_tiny_26_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(
        pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT
    )
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
        model = replace_convlayers_convnext(model, 100)

    return ConvNext_features(model)


def convnext_tiny_13_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(
        pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT
    )
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
        model = replace_convlayers_convnext(model, 300)

    return ConvNext_features(model)

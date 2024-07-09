import torch
import torch.nn as nn
from torchvision import models

def convnext_tiny(pretrained: bool, num_classes: int, **kwargs):
    if pretrained:
        model = models.convnext_tiny(
            pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        model.classifier[2] = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

    return model
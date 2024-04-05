from models.features.convnext_features import (
    convnext_tiny_13_features,
    convnext_tiny_26_features,
)
from models.features.resnet_features import (
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet50_features_inat,
    resnet101_features,
    resnet152_features,
)
from models.features.vgg_features import (
    vgg11_bn_features,
    vgg11_features,
    vgg13_bn_features,
    vgg13_features,
    vgg16_bn_features,
    vgg16_features,
    vgg19_bn_features,
    vgg19_features,
)

base_architecture_to_features = {
    "convnext_tiny_13": convnext_tiny_13_features,
    "convnext_tiny_26": convnext_tiny_26_features,
    "resnet18": resnet18_features,
    "resnet34": resnet34_features,
    "resnet50": resnet50_features,
    "resnet101": resnet101_features,
    "resnet152": resnet152_features,
    "resnet50_inat": resnet50_features_inat,
    "vgg11": vgg11_features,
    "vgg11_bn": vgg11_bn_features,
    "vgg13": vgg13_features,
    "vgg13_bn": vgg13_bn_features,
    "vgg16": vgg16_features,
    "vgg16_bn": vgg16_bn_features,
    "vgg19": vgg19_features,
    "vgg19_bn": vgg19_bn_features,
}


base_architecture_to_layer_groups = {
    "convnext_tiny_13": {
        "to_train": ["features.5.3"],
        "to_freeze": ["features.5", "features.4"],
        "backbone": ['features.3', 'features.2'],
    },
    "convnext_tiny_26": {
        "to_train": ["features.7.2"],
        "to_freeze": ["features.7", "features.6"],
        "backbone": ['features.5', 'features.4'],
    },
    "resnet18": {
        "to_train": ["layer4.1"],
        "to_freeze": ["layer4", "layer3"],
        "backbone": ["layer2"],
    },
    "resnet34": {
        "to_train": ["layer4.1"],
        "to_freeze": ["layer4", "layer3"],
        "backbone": ["layer2"],
    },
    "resnet50": {
        "to_train": ["layer4.2"],
        "to_freeze": ["layer4", "layer3"],
        "backbone": ["layer2"],
    },
    "resnet101": {
        "to_train": ["layer3.2"],
        "to_freeze": ["layer3"],
        "backbone": ["layer2"],
    },
    "resnet152": {
        "to_train": ["layer2.7"],
        "to_freeze": ["layer2.6", "layer2.5", "layer2.4"],
        "backbone": ["layer2"],
    },
    "vgg11": {
        "to_train": ["features18"],
        "to_freeze": ["features16", "features13", "features11"],
        "backbone": ["features8", "features6"],
    },
    "vgg13": {
        "to_train": ["features22"],
        "to_freeze": ["features20", "features17", "features15"],
        "backbone": ["features12", "features10", "features7"],
    },
    "vgg16": {
        "to_train": ["features28"],
        "to_freeze": ["features26", "features24", "features21"],
        "backbone": ["features19", "features17", "features14"],
    },
    "vgg19": {
        "to_train": ["features34"],
        "to_freeze": ["features32", "features30", "features28"],
        "backbone": ["features25", "features23", "features21"],
    },
}

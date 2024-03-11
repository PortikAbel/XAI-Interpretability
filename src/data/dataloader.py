import albumentations as A
import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

from data.config import (
    dataset_config,
    test_batch_size,
    train_batch_size,
    train_push_batch_size,
)

# load the data
data_dir = dataset_config["data_dir"]
image_folder_structure = dataset_config["image_folders"]
img_shape = dataset_config["img_shape"]
train_push_dir = dataset_config["train_push_dir"]
test_dir = dataset_config["test_dir"]
train_dir = dataset_config["train_dir"]

mean = dataset_config["mean"]
std = dataset_config["std"]

# Pytorch transform. pipelines:
normalize = transforms.Normalize(mean=mean, std=std)
base_transform = transforms.Compose(
    [
        transforms.Resize(size=img_shape),
        transforms.ToTensor(),
    ]
)
normalizing_transform = transforms.Compose(
    [
        base_transform,
        normalize,
    ]
)


# Albumentations augm. pipelines:
train_augmentation = A.Compose(
    [
        A.augmentations.transforms.RandomBrightnessContrast(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
    ]
)


# helper class for augmentation with Albumentations library - when used with ImageFolder
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


# all datasets
# train set
if dataset_config["augm"]:  # data augmentation is added via albumentations
    print("AUGMENTING")
    normalize_augment = A.Compose(
        [
            A.ToFloat(max_value=256),  # 8-bits  # TODO: read max value from config
            train_augmentation,  # augmentation with Albumentations
            A.Normalize(
                mean=mean, std=std, max_pixel_value=1.0, p=1.0
            ),  # TODO: read max value from config
            ToTensorV2(),
        ]
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=Transforms(transforms=normalize_augment),
    )
else:
    train_dataset = datasets.ImageFolder(
        train_dir,
        normalizing_transform,
    )


train_sampler = torch.utils.data.RandomSampler(
    train_dataset,
    replacement=False,
    num_samples=100,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    # when running with a subset of the dataset, set Shuffle to False, otherwise to True
    shuffle=True,
    # sampler=train_sampler,  # OPTIONAL; for running with a subset of the whole dataset
)

# # push set
if dataset_config["augm"]:  # data augmentation is added via albumentations
    normalize = A.Compose(  # TODO: is this necessary for the push set?
        [
            A.ToFloat(max_value=256),  # 8-bits
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ToTensorV2(),
        ]
    )

    train_push_dataset = datasets.ImageFolder(
        train_dir,
        transform=Transforms(transforms=normalize),
    )

else:
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        base_transform,
    )

train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset,
    batch_size=train_push_batch_size,
    shuffle=False,
)

# # validation set
# TODO: change this to actually read validation set - if there in any
if dataset_config["augm"]:  # data augmentation is added via albumentations
    print("VALID DATASET")
    normalize_augment = A.Compose(
        [
            A.ToFloat(max_value=256),  # 8-bits
            A.Normalize(
                mean=mean, std=std, max_pixel_value=1.0, p=1.0
            ),  # TODO: read max value from config
            ToTensorV2(),
        ]
    )

    valid_dataset = datasets.ImageFolder(
        test_dir,
        transform=Transforms(transforms=normalize_augment),
    )
else:
    valid_dataset = datasets.ImageFolder(
        test_dir,
        normalizing_transform,
    )

valid_sampler = torch.utils.data.RandomSampler(
    valid_dataset,
    replacement=False,
    num_samples=200,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    # add to work with only a random subset of the original dataset
    sampler=valid_sampler,
)


if dataset_config["augm"]:  # data augmentation is added via albumentations
    print("TEST DATASET")
    normalize_augment = A.Compose(
        [
            A.ToFloat(max_value=256),  # 8-bits
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ToTensorV2(),
        ]
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=Transforms(transforms=normalize_augment),
    )
else:
    test_dataset = datasets.ImageFolder(
        test_dir,
        normalizing_transform,
    )


test_sampler = torch.utils.data.RandomSampler(
    test_dataset,
    replacement=False,
    num_samples=200,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    # add to work with only a random subset of the original dataset
    sampler=test_sampler,
)

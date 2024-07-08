import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torch import Tensor

from data.config import DATASETS


def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        train_set,
        train_set_pretraining,
        train_set_projection,
        test_set,
        test_set_projection,
        classes,
        train_indices,
        targets,
    ) = get_datasets(args)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    sampler = None
    to_shuffle_train_set = True

    if args.weighted_loss:
        if targets is None:
            raise ValueError(
                "Weighted loss not implemented for this dataset. "
                "Targets should be restructured"
            )
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907 # noqa
        class_sample_count = torch.tensor(
            [
                (targets[train_indices] == t).sum()
                for t in torch.unique(targets, sorted=True)
            ]
        )
        weight = 1.0 / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )
        to_shuffle_train_set = False

    def create_dataloader(dataset, batch_size, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dataset,
            # batch size is np.uint16, so we need to convert it to int
            batch_size=int(batch_size),
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=cuda,
            num_workers=args.num_workers,
            worker_init_fn=np.random.seed(args.seed),
            drop_last=drop_last,
        )

    train_loader = create_dataloader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=to_shuffle_train_set,
        drop_last=True,
    )
    train_loader_pretraining = create_dataloader(
        dataset=train_set_pretraining or train_set,
        batch_size=args.batch_size_pretrain,
        shuffle=to_shuffle_train_set,
        drop_last=True,
    )
    train_loader_projection = create_dataloader(
        dataset=train_set_projection,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    test_loader = create_dataloader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader_projection = create_dataloader(
        dataset=test_set_projection,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)

    return (
        train_loader,
        train_loader_pretraining,
        train_loader_projection,
        test_loader,
        test_loader_projection,
        classes,
    )


def get_datasets(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_config = DATASETS[args.dataset]

    train_dir = dataset_config["train_dir"]
    train_dir_pretrain = dataset_config.get("train_dir_pretrain", train_dir)
    train_dir_projection = dataset_config.get("train_dir_projection", train_dir)
    test_dir = dataset_config.get("test_dir", None)
    test_dir_projection = dataset_config.get("test_dir_projection", test_dir)

    train_val_set = torchvision.datasets.ImageFolder(train_dir)
    classes = train_val_set.classes
    targets = train_val_set.targets
    indices = list(range(len(train_val_set)))

    train_indices = indices

    (
        transform_no_augment,
        transform1,
        transform1p,
        transform2,
    ) = get_transforms(args)

    if test_dir is None:
        if args.validation_size <= 0.0:
            raise ValueError(
                "There is no test set directory, so validation size "
                "should be > 0 such that training set can be split."
            )
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(
            train_indices,
            test_size=args.validation_size,
            stratify=subset_targets,
            random_state=args.seed,
        )
        test_set = torch.utils.data.Subset(
            torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment),
            indices=test_indices,
        )
        print(
            "Samples in train_set:",
            len(indices),
            "of which",
            len(train_indices),
            "for training and ",
            len(test_indices),
            "for testing.",
            flush=True,
        )
    else:
        test_set = torchvision.datasets.ImageFolder(
            test_dir, transform=transform_no_augment
        )
    if test_dir_projection is not None:
        test_set_projection = torchvision.datasets.ImageFolder(
            test_dir_projection,
            transform=transform_no_augment,
        )
    else:
        test_set_projection = test_set

    train_set = torch.utils.data.Subset(
        TwoAugSupervisedDataset(
            train_val_set, transform1=transform1, transform2=transform2
        ),
        indices=train_indices,
    )
    train_set_projection = torchvision.datasets.ImageFolder(
        train_dir_projection,
        transform=transform_no_augment,
    )
    if train_dir_pretrain is not None:
        train_val_set_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = train_val_set_pr.targets
        indices_pr = list(range(len(train_val_set_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(
                indices_pr,
                test_size=args.validation_size,
                stratify=subset_targets_pr,
                random_state=args.seed,
            )

        train_set_pretraining = torch.utils.data.Subset(
            TwoAugSupervisedDataset(
                train_val_set_pr, transform1=transform1p, transform2=transform2
            ),
            indices=train_indices_pr,
        )
    else:
        train_set_pretraining = None

    return (
        train_set,
        train_set_pretraining,
        train_set_projection,
        test_set,
        test_set_projection,
        classes,
        train_indices,
        torch.LongTensor(targets),
    )


def get_transforms(args: argparse.Namespace):
    dataset_config = DATASETS[args.dataset]

    mean = dataset_config["mean"]
    std = dataset_config["std"]
    img_shape = tuple(args.image_shape)

    normalize = transforms.Normalize(mean=mean, std=std)

    transform_no_augment = transforms.Compose(
        [
            transforms.Resize(size=img_shape),
            transforms.ToImage(),
            transforms.ConvertImageDtype(),
            normalize,
        ]
    )

    if dataset_config["augm"]:
        # transform1: first step of augmentation
        match args.dataset:
            case "CUB-200-2011" | "CUB-10" | "Funny" | "Funny-10" | "Funny-3":
                transform1 = transforms.Compose(
                    [
                        transforms.Resize(size=(img_shape[0] + 8, img_shape[1] + 8)),
                        TrivialAugmentWideNoColor(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(
                            size=(img_shape[0] + 4, img_shape[1] + 4),
                            scale=(0.95, 1.0),
                        ),
                    ]
                )
            case _:
                transform1 = transforms.Compose(
                    [
                        transforms.Resize(
                            size=(img_shape[0] + 48, img_shape[1] + 48),
                        ),
                        TrivialAugmentWideNoColor(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(
                            size=(img_shape[0] + 8, img_shape[1] + 8),
                            scale=(0.95, 1.0),
                        ),
                    ]
                )

        # transform1p: alternative for transform1 during pretrain
        match args.dataset:
            case "CUB-200-2011" | "CUB-10" | "Funny" | "Funny-10" | "Funny-3":
                transform1p = transforms.Compose(
                    [
                        transforms.Resize(
                            size=(img_shape[0] + 32, img_shape[1] + 32)
                        ),  # for pretraining, crop can be bigger since
                        # it doesn't matter when bird is not fully visible
                        TrivialAugmentWideNoColor(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(
                            size=(img_shape[0] + 4, img_shape[1] + 4),
                            scale=(0.95, 1.0),
                        ),
                    ]
                )
            case _:
                transform1p = transform_no_augment

        # transform2: second step of augmentation
        # applied twice on the result of transform1(p) to obtain two similar imgs
        transform2 = transforms.Compose(
            [
                TrivialAugmentWideNoShape(),
                transforms.RandomCrop(size=img_shape),  # includes crop
                transforms.ToImage(),
                transforms.ConvertImageDtype(),
                normalize,
            ]
        )
    else:
        transform1 = transform_no_augment
        transform1p = transform_no_augment
        transform2 = transform_no_augment

    return (
        transform_no_augment,
        transform1,
        transform1p,
        transform2,
    )


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    """Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if isinstance(dataset, torchvision.datasets.folder.ImageFolder):
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


# function copied fromhttps://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted  # noqa
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

import argparse

import albumentations as A
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from data.config import DATASETS
from utils.log import Log


def get_dataloaders(log: Log, args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        train_set,
        push_set,
        test_set,
        classes,
        train_indices,
        targets,
    ) = get_datasets(log, args)

    # Determine if GPU should be used
    cuda = not args.disable_gpu and torch.cuda.is_available()
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
        log.info(f"Weights for weighted sampler: {weight}")
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
    test_loader = create_dataloader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    push_loader = create_dataloader(
        dataset=push_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    log.info(f"Num classes (k) = {len(classes)}, {classes[:5]}, etc.")

    return (
        train_loader,
        push_loader,
        test_loader,
        classes,
    )


def get_datasets(log: Log, args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    """
    dataset_config = DATASETS[args.dataset]

    train_dir = dataset_config["train_dir"]
    train_dir_projection = dataset_config.get("train_dir_projection", train_dir)
    test_dir = dataset_config.get("test_dir", None)

    train_val_set = torchvision.datasets.ImageFolder(train_dir)
    classes = train_val_set.classes
    targets = train_val_set.targets
    indices = list(range(len(train_val_set)))

    train_indices = indices

    (
        transform_no_augment,
        transform_train,
        transform_push,
        transform_validation,
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
            torchvision.datasets.ImageFolder(
                train_dir, transform=Transforms(transform_validation)
            ),
            indices=test_indices,
        )
        log.info(
            f"Samples in train_set: {len(indices)} of which {len(train_indices)} for "
            f"training and {len(test_indices)} for testing.",
        )
    else:
        test_set = torchvision.datasets.ImageFolder(
            test_dir, transform=Transforms(transform_validation)
        )

    train_set = datasets.ImageFolder(
        train_dir,
        transform=Transforms(transform_train),
    )

    push_set = torchvision.datasets.ImageFolder(
        train_dir_projection,
        transform=Transforms(transform_push),
    )

    return (
        train_set,
        push_set,
        test_set,
        classes,
        train_indices,
        torch.LongTensor(targets),
    )


def get_transforms(args: argparse.Namespace):
    normalize = A.Normalize(mean=args.mean, std=args.std, max_pixel_value=1.0, p=1.0)
    base_transform = A.Compose(
        [
            A.ToFloat(max_value=256),
            A.Resize(height=args.img_shape[0], width=args.img_shape[1]),
            ToTensorV2(),
        ]
    )
    transform_no_augment = A.Compose(
        [
            A.ToFloat(max_value=256),
            A.Resize(height=args.img_shape[0], width=args.img_shape[1]),
            normalize,
            ToTensorV2(),
        ]
    )

    if args.augm:
        transform_train = A.Compose(
            [
                A.ToFloat(max_value=256),
                A.Resize(height=args.img_shape[0], width=args.img_shape[1]),
                A.RandomBrightnessContrast(),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                normalize,
                ToTensorV2(),
            ]
        )

        transform_test = A.Compose(
            [
                A.ToFloat(max_value=256),
                A.Resize(height=args.img_shape[0], width=args.img_shape[1]),
                normalize,
                ToTensorV2(),
            ]
        )
    else:
        transform_train = transform_no_augment
        transform_test = transform_no_augment

    transform_push = base_transform

    return (
        transform_no_augment,
        transform_train,
        transform_push,
        transform_test,
    )


# helper class for augmentation with Albumentations library - when used with ImageFolder
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]

import os
import cv2
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

from data.config import DATASETS

# labels_dict = {"Expressway": 0, "Motorway": 1, "Rural": 2, "Urban": 3}
labels_dict = {"Motorway": 0, "Rural": 1, "Urban": 2}


class RoadTypeDetectionDataset(Dataset):
    def __init__(
        self,
        subset="training",
        augmentation=A.NoOp(),
        target_transform=lambda data: torch.tensor(data, dtype=torch.long),
    ):
        self.data_dir = DATASETS["road_type_detection"]["data_dir"]
        self.augment = A.Compose(
            [
                A.ToFloat(max_value=4096),  # 12-bits
                A.crops.transforms.Crop(
                    x_min=0,
                    y_min=0,
                    x_max=DATASETS["road_type_detection"]["img_shape"][1],
                    y_max=DATASETS["road_type_detection"]["img_shape"][0],
                    p=1.0,
                ),  # cropping the lower region of images (car's board)
                # A.augmentations.geometric.resize.Resize(
                #   height=img_shape[0],
                #   width=img_shape[1]), # not all images are the same size
                augmentation,
                ToTensorV2(),
            ]
        )
        self.target_transform = target_transform

        annotations_file_name = DATASETS["road_type_detection"][
            "annotations_file_name"
        ]
        annotations_file = os.path.join(self.data_dir, annotations_file_name)
        columns = ["road_type", "stage_loso", "color_path"]

        self.img_labels = pd.read_csv(
            annotations_file, sep=",", usecols=columns
        )

        self.subset = subset

        self.img_labels = self.img_labels[
            self.img_labels["stage_loso"] == subset
        ]

        # self.img_labels = self.img_labels[
        #     self.img_labels["stage_loso"] == ("test" if test else "training")
        # ]

        self.targets = self.img_labels["road_type"]

        self.img_labels.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_dir, self.img_labels.at[idx, "color_path"]
        )
        image = cv2.imread(
            img_path, cv2.IMREAD_UNCHANGED
        )  # reading 12 bits image

        # saving a few samples to check results of the augmentation:
        # filename = self.img_labels.at[idx, "color_path"].replace("/", "__")
        # cv2.imwrite(
        # "/bigdata/userhome/annamari/self-explain/code/bagnet_pytorch/image_samples/"
        # + self.subset + filename, (image/4095)*256)

        if self.augment:
            image = self.augment(image=image)["image"]

        label = self.img_labels.at[idx, "road_type"]
        label = labels_dict[label]
        if self.target_transform:
            label = self.target_transform(label)

        # cv2.imwrite(
        # "/bigdata/userhome/annamari/self-explain/code/bagnet_pytorch/image_samples/aumg_"
        # +  self.subset + filename, image.numpy().transpose([1, 2, 0])*256)

        # return image, label

        # TODO: add bool parameter that can be used to indicate that image labels
        # should be returned, too
        # use this to return image paths, too
        return image, label, self.img_labels.at[idx, "color_path"]


class RoadTypeDetectionDatasetWithPath(RoadTypeDetectionDataset):
    def __init__(
        self,
        subset="training",
        augmentation=A.NoOp(),
        target_transform=lambda data: torch.tensor(data, dtype=torch.long),
    ):
        super().__init__(subset, augmentation, target_transform)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_dir, self.img_labels.at[idx, "color_path"]
        )
        image = cv2.imread(
            img_path, cv2.IMREAD_UNCHANGED
        )  # reading 12 bits image

        if self.augment:
            image = self.augment(image=image)["image"]

        label = self.img_labels.at[idx, "road_type"]
        label = labels_dict[label]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path

import os
import pandas as pd

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data.config import DATASETS


class RoadConditionMonitoringDataset(Dataset):
    def __init__(
        self,
        test=False,
        transform=None,
        target_transform=None,
    ):
        self.data_dir = DATASETS["road_condition_monitoring"]["data_dir"]
        self.transform = transforms
        self.target_transform = target_transform

        annotations_file_name = DATASETS["road_condition_monitoring"][
            "annotations_file_name"
        ]
        annotations_file = os.path.join(self.data_dir, annotations_file_name)
        columns = ["class", "stage", "color_path"]
        self.img_labels = pd.read_csv(annotations_file, usecols=columns)
        self.img_labels = self.img_labels[
            self.img_labels["stage"] == ("test" if test else "training")
        ]
        self.img_labels.reset_index(inplace=True, drop=True)
        self.img_labels["color_path"] = self.img_labels[
            "color_path"
        ].transform(lambda s: s.removeprefix("color_data/"))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_dir, self.img_labels.at[idx, "color_path"]
        )
        # image = read_image(img_path)
        label = self.img_labels.at[idx, "class"]
        # if self.transform:
        #     image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return img_path, label

import os

from utils.environment import get_env


DATASETS = {}

data_dir = os.path.join(get_env("DATA_ROOT"), "CUB_200", "dataset")
DATASETS["CUB-200-2011"] = {
    "img_shape": (224, 224),
    "num_classes": 200,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train_crop"),
    "train_dir_projection": os.path.join(data_dir, "train_full"),
    "test_dir": os.path.join(data_dir, "test_crop"),
    "test_dir_projection": os.path.join(data_dir, "test_full"),
    "image_folders": True,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "augm": True,
}


data_dir = os.path.join(get_env("DATA_ROOT"), "CUB_200", "small")
DATASETS["CUB-10"] = {
    "img_shape": (224, 224),
    "num_classes": 10,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train_crop"),
    "train_dir_projection": os.path.join(data_dir, "train_full"),
    "test_dir": os.path.join(data_dir, "test_crop"),
    "test_dir_projection": os.path.join(data_dir, "test_full"),
    "image_folders": True,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "augm": True,
}


data_dir = os.path.join(get_env("DATA_ROOT"), "MNIST", "ImageFolders")
DATASETS["MNIST"] = {
    "img_shape": (32, 32),
    "num_classes": 10,
    "num_prototypes_per_class": 10,
    "color_channels": 1,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train"),
    "test_dir": os.path.join(data_dir, "test"),
    "image_folders": True,
    "augm": False,
}


data_dir = os.path.join(get_env("DATA_ROOT"), "pascal_voc")
DATASETS["pascal_voc"] = {
    "img_shape": (224, 224),
    "num_classes": 20,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_push_dir": os.path.join(data_dir, "train_4"),
    "train_dir": os.path.join(data_dir, "train_4"),
    "test_dir": os.path.join(data_dir, "test_4"),
    "mean": (0.4592, 0.4360, 0.4035),
    "std": (0.2704, 0.2669, 0.2792),
    "image_folders": True,
    "augm": False,
}


train_batch_size = 32
test_batch_size = 32
train_push_batch_size = 32

# change here to read the config for the data set you want to work with:
dataset_config = DATASETS["CUB-10"]

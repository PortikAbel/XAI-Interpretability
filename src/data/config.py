import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

assert (
    os.getenv("PROJECT_ROOT") is not None
), "Please set the environment variable PROJECT_ROOT in .env file"
assert (
    os.getenv("DATASET_LOCATION") is not None
), "Please set the environment variable DATASET_LOCATION in .env file"
assert (
    os.getenv("BOSCH_DATA_LOCATION") is not None
), "Please set the environment variable BOSCH_DATA_LOCATION in .env file"

# STUDGPU:
bosch_dir = os.path.join(
    os.getenv("BOSCH_DATA_LOCATION"),
    "L4-12-bits",
    # "view",
)

DATASETS = {}

data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "CUB_200", "dataset")
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


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "CUB_200", "small")
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


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "BDD100k", "100k-weather")
DATASETS["BDD"] = {
    "img_shape": (224, 224),
    "num_classes": 7,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train"),
    "test_dir": os.path.join(data_dir, "test"),
    "image_folders": True,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "augm": False,
}


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "MNIST", "ImageFolders")
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


# data_dir = os.path.join(bosch_dir, "RoadTypeDetection", "small")
data_dir = os.path.join(bosch_dir, "RoadTypeDetection")
DATASETS["road_type_detection"] = {
    # "img_shape": (512, 1664),  # original
    # L1/4; original shape: (128, 416) - reduced width by 20px to crop out car's board
    # "img_shape": (64, 208),
    "img_shape": (128, 416),
    "num_classes": 3,
    "num_prototypes_per_class": 10,
    "classes": ["Motorway", "Rural", "Urban"],
    "color_channels": 1,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train"),
    "test_dir": os.path.join(data_dir, "test"),
    "image_folders": False,
    "mean": (0.4294,),
    "std": (0.2697,),
    "annotations_file_name": "dataset-meta.csv",
    "augm": True,
}


data_dir = os.path.join(bosch_dir, "RoadConditionMonitoring")
DATASETS["road_condition_monitoring"] = {
    "img_shape": (512, 1664),
    "num_classes": 10,
    "num_prototypes_per_class": 10,
    "color_channels": 1,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "images_LUV", "rcm_seq_to_frames_train"),
    "test_dir": os.path.join(data_dir, "images_LUV", "rcm_seq_to_frames_test"),
    "image_folders": False,
    "annotations_file_name": "dataset-meta.csv",
    "augm": True,
}


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "pascal_voc")
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


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "PETS", "dataset")
DATASETS["pets"] = {
    "img_shape": (224, 224),
    "num_classes": 0,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train"),
    "test_dir": os.path.join(data_dir, "test"),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "image_folders": True,
    "augm": False,
}


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "partimagenet", "dataset", "all")
DATASETS["partimagenet"] = {
    "img_shape": (224, 224),
    "num_classes": 0,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": data_dir,
    # use --validation_size of 0.2
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "image_folders": True,
    "augm": False,
}


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "cars", "dataset")
DATASETS["CARS"] = {
    "img_shape": (224, 224),
    "num_classes": 0,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train"),
    "test_dir": os.path.join(data_dir, "test"),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "image_folders": True,
    "augm": False,
}


data_dir = os.path.join(os.getenv("DATASET_LOCATION"), "data")
DATASETS["grayscale_example"] = {
    "img_shape": (224, 224),
    "num_classes": 0,
    "num_prototypes_per_class": 10,
    "color_channels": 3,
    "data_dir": data_dir,
    "train_dir": os.path.join(data_dir, "train"),
    "test_dir": os.path.join(data_dir, "test"),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "image_folders": True,
    "augm": False,
}

train_batch_size = 32
test_batch_size = 32
train_push_batch_size = 32

# change here to read the config for the data set you want to work with:
dataset_config = DATASETS["road_type_detection"]

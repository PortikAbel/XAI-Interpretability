"""
Based on some (give or random) classes generates a smaller dataset from FunnyBirds.

Raises:
    ValueError: Either --classes or --random should be provided

Examples:
    Generate a dataset containing the given classes.

    ``python generate_funny_10.py --classes 0 3 6 7 10 32 37 44 45 47``

    Generate a dataset containing 10 random classes.

    ``python generate_funny_10.py --random 10``
"""

import json
import warnings
from argparse import ArgumentParser

import numpy as np

from data.config import DATASETS

parser = ArgumentParser(
    "Generate Funny-10 dataset",
    description="Based on some (give or random) classes generates a "
    "smaller dataset from FunnyBirds",
)
parser.add_argument(
    "--classes",
    type=int,
    nargs="*",
    help="Classes that will be selected for the new dataset",
)
parser.add_argument(
    "--random", type=int, default=0, help="Number of random classes to select"
)


args = parser.parse_args()
if args.random == 0 and len(args.classes) == 0:
    raise ValueError("Either --classes or --random should be provided")

if args.random > len(args.classes):
    warnings.warn(
        f"Random number of classes is greater than the number "
        f"of classes selected. {args.random} - {len(args.classes)} "
        f"classes will be selected randomly"
    )

funny_dir = DATASETS["Funny"]["data_dir"]
small_funny_dir = DATASETS["Funny-10"]["data_dir"]
print(funny_dir)

args.random = max(0, args.random - len(args.classes))
selected_classes = np.array(args.classes)
if args.random > 0:
    classes = np.random.choice(range(0, 50), size=args.random, replace=False)
    selected_classes = np.concatenate([selected_classes, classes])
print(selected_classes)

small_funny_dir.mkdir()
dirs = [d for d in funny_dir.iterdir() if d.is_dir()]
for i in selected_classes:
    for d in dirs:
        new_dir = small_funny_dir / d.name / f"{i}"
        new_dir.parent.mkdir(exist_ok=True)
        new_dir.symlink_to(d / f"{i}")

(small_funny_dir / "parts.json").symlink_to(funny_dir / "parts.json")

class_file = funny_dir / "classes.json"
with class_file.open("r") as cls_file:
    classes = json.load(cls_file)
    new_data = [i for i in classes if i["class_idx"] in selected_classes]
    with (small_funny_dir / "classes.json").open("w") as new_cls_file:
        json.dump(new_data, new_cls_file)

import argparse
import re

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(__file__, description="Export prototypes")
parser.add_argument(
    "-p", "--path",
    type=Path,
    help="Path to the visualization results",
    required=True,
)
parser.add_argument(
    "-n", "--n_prototypes",
    type=int,
    help="Number of prototypes per class",
    default=10,
)
parser.add_argument(
    "--n_classes",
    type=int,
    help="Number of classes to show on the resulting image",
    default=10,
)
parser.add_argument(
    "-r", "--regex_pattern",
    type=str,
    help="Regex pattern to filter the files",
    default="*original_with_self_act*",
)
parser.add_argument(
    "-o", "--output",
    type=Path,
    help="Output path",
    default=Path("."),
)
parser.add_argument(
    "--filename",
    type=str,
    help="Filename of the resulting image",
    default="prototypes.png",
)

args = parser.parse_args()

result_path = args.path
selected_files = [_ for _ in result_path.glob(args.regex_pattern)]
if len(selected_files) == 0:
    raise ValueError(
        f"No files were selected from {result_path} based on "
        f"pattern {args.regex_pattern!r}"
    )

selected_files.sort(
    key=lambda file: int(re.search(r"(\d+)", file.name).group(0))
)

selected_classes = np.random.choice(
    np.arange(len(selected_files) // args.n_prototypes), args.n_classes, replace=False
)

fig, axs = plt.subplots(args.n_classes, args.n_prototypes, figsize=(120, 120))
for i, class_idx in enumerate(selected_classes):
    for j in range(args.n_prototypes):
        img = plt.imread(selected_files[class_idx * args.n_prototypes + j])
        axs[i, j].imshow(img)
        # axs[i, j].axis("off")
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

for j, ax_j in enumerate(range(len(axs[0])), start=1):
    axs[0, ax_j].set_title(f"Prototype {j}")

for i, ax_i in enumerate(range(len(axs[:, 0])), start=1):
    axs[ax_i, 0].set_ylabel(f"Class {i}", rotation=0, size="large")

plt.savefig(args.output / args.filename, dpi=300, bbox_inches="tight")

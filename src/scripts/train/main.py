import argparse
import random
import sys

import numpy as np
import torch

from utils.log import Log

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument(
    "--model", type=str, default="PIPNet", help="name of the explainable model to train"
)


args, _ = parser.parse_known_args()
match args.model:
    case "ProtoPNet":
        from models.ProtoPNet.trainer import train_model
        from models.ProtoPNet.util.args import (
            ProtoPNetArgumentParser as model_args_parser,
        )
    case "PIPNet":
        from models.PIPNet.trainer import train_model
        from models.PIPNet.util.args import PIPNetArgumentParser as model_args_parser
    case _:
        raise ValueError(f"Unknown model: {args.model}")

model_args = model_args_parser.get_args()

# Create a logger
log = Log(args.log_dir)
print("Log dir: ", args.log_dir, flush=True)
# Log the run arguments
model_args_parser.save_args(log.metadata_dir)

torch.manual_seed(model_args.seed)
torch.cuda.manual_seed_all(model_args.seed)
random.seed(model_args.seed)
np.random.seed(model_args.seed)

standard_output_file = args.log_dir / "out.txt"
error_output_file = args.log_dir / "error.txt"

sys.stdout.close()
sys.stderr.close()
sys.stdout = standard_output_file.open(mode="w")
sys.stderr = error_output_file.open(mode="w")

train_model(log, model_args)

# close output files
sys.stdout.close()
sys.stderr.close()

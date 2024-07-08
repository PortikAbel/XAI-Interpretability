import argparse
import random
import warnings

import numpy as np
import torch

from utils.log import Log

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument(
    "--model", type=str, default="PIPNet", help="name of the explainable model to train"
)
parser.add_argument(
    "--enable_console", action="store_true", help="Enable console output"
)


args, _ = parser.parse_known_args()
match args.model:
    case "ProtoPNet":
        from models.ProtoPNet.trainer import train_model
        from models.ProtoPNet.util.args import (
            ProtoPNetArgumentParser as ModelArgumentParser,
        )
    case "PIPNet":
        from models.PIPNet.trainer import train_model
        from models.PIPNet.util.args import PIPNetArgumentParser as ModelArgumentParser
    case _:
        raise ValueError(f"Unknown model: {args.model}")

model_args = ModelArgumentParser.get_args()

# Create a logger
log = Log(model_args.log_dir, __name__, not args.enable_console)

# Log the run arguments
ModelArgumentParser.save_args(log.metadata_dir)

torch.manual_seed(model_args.seed)
torch.cuda.manual_seed_all(model_args.seed)
random.seed(model_args.seed)
np.random.seed(model_args.seed)

try:
    train_model(log, model_args)
except Exception as e:
    log.exception(e)
log.close()

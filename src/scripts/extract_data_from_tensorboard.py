import argparse

from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.environment import get_env
from utils.file_operations import get_package
from utils.log import BasicLog

parser = argparse.ArgumentParser(
    __file__,
    description="Extract values from the tensorboard"
)
parser.add_argument(
    "-d", "--dir",
    type=Path,
    required=True,
    help="Tensorboard log directory",
)
parser.add_argument(
    "-n", "--names",
    type=str,
    nargs="+",
    action="append",
    help="List of value names to be extracted into one file"
)
parser.add_argument(
    "-o", "--output",
    type=str,
    nargs="*",
    default=[],
    help="Names of the output files"
)
parser.add_argument(
    "-e", "--epochs",
    type=int,
    help="If steps are logged, this number is used the compute the values per epoch"
)
parser.add_argument(
    "--log_dir",
    type=Path,
    default=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    help="The directory in which train progress should be logged",
)
parser.add_argument(
    "--enable_console",
    action="store_true",
    help="Enable console output"
)


def tflog2pandas(
        path: Path, name_group: list, epochs: int | None, logger: BasicLog
) -> pd.DataFrame:
    """
    Extract scalars from tensorboard.

    :param path: tensorboard log directory
    :param name_group: name of scalars to export
    :param epochs: number of epochs
    :param logger:
    :return: exported scalar values
    """
    runlog_data = pd.DataFrame()
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        logger.info(event_acc.Tags())
        for name in name_group:
            if name not in tags:
                raise ValueError(
                    f"Scalar {name} was not found on tensorboard. "
                    f"Select one of {', '.join(tags)}"
                )
            event_list = event_acc.Scalars(name)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            if epochs is not None:
                step_per_epoch = int(np.ceil(step[-1] / epochs))
                values = [
                    np.average(values[i:i + step_per_epoch])
                    for i in range(0, len(values), step_per_epoch)
                ]
                if len(values) < epochs:
                    missing = int(np.ceil(step[0] / step_per_epoch))
                    values = [None] * missing + values
                step = range(epochs)
                print(values)
            current = pd.DataFrame(values, columns=[name], index=step)
            runlog_data = pd.concat([runlog_data, current], axis="columns")
    except Exception as e:
        logger.error(f"Event file possibly corrupt: {path}")
        logger.exception(e)
    return runlog_data


def get_longest_match(names: list) -> str:
    """
    Get the longest common part in the specified list of strings.

    :param names: list of metric names
    :return: longest common part of the given strings
    """
    previous_match = names[0]
    i = 1
    while i < len(names) and len(previous_match) > 0:
        match = SequenceMatcher(a=previous_match, b=names[i]).find_longest_match()
        previous_match = previous_match[match.a:match.a + match.size]
        i += 1

    return previous_match


def check_args() -> argparse.Namespace:
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError(
            f"Number of epochs must be a positive integer, but got {args.epochs}"
        )
    if not args.log_dir.is_absolute():
        results_location = (
                get_env("RESULTS_LOCATION", must_exist=False) or
                get_env("PROJECT_ROOT")
        )
        args.log_dir = Path(
            results_location,
            "runs",
            get_package(__file__),
            args.log_dir
        )
        args.log_dir = args.log_dir.resolve()

    if len(args.output) > 0 and len(args.output) != len(args.names):
        raise ValueError(f"All output file names must be specified or neither of them.")

    return args


def main(args):
    logger = BasicLog(args.log_dir, __name__, not args.enable_console)
    try:
        total = len(args.names)
        for i, name_group in enumerate(args.names, start=1):
            prefix = f"{i} / {total}"
            logger.info(f"{prefix} {name_group}")
            data = tflog2pandas(args.dir, name_group, args.epochs, logger)
            #df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
            if i - 1 < len(args.output):
                filename = args.output[i - 1]
            else:
                filename = get_longest_match(name_group)
                if len(filename) == 0:
                    filename = f"group-{i}"
            filename = logger.log_dir / filename
            filename = filename.with_suffix(".csv")
            data.to_csv(filename)
            logger.info(f"{prefix} File saved to {filename}")
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main(check_args())

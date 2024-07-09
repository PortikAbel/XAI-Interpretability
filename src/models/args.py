import argparse
import pickle
from abc import ABC, abstractmethod
from pathlib import Path


class ModelArgumentParser(ABC):
    _parser: argparse.ArgumentParser
    _args: argparse.Namespace

    @classmethod
    @abstractmethod
    def get_args(cls, known_args_only: bool = True) -> argparse.Namespace:
        """
        Parse the arguments for the model.

        :param known_args_only: If ``True``, only known arguments are parsed.
            Defaults to ``True``.
        :return: specified arguments in the command line
        """
        if known_args_only:
            cls._args = cls._parser.parse_known_args()[0]
        else:
            cls._args = cls._parser.parse_args()
        return cls._args

    @classmethod
    def save_args(cls, directory_path: Path) -> None:
        """
        Save the arguments in the specified directory as
            - a text file called 'args.txt'
            - a pickle file called 'args.pickle'
        :param directory_path: The path to the directory where the
            arguments should be saved
        """
        # If the specified directory does not exist, create it
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
        # Save the args in a text file
        with (directory_path / "args.txt").open(mode="w") as f:
            for arg in vars(cls._args):
                val = getattr(cls._args, arg)
                if isinstance(val, str):
                    # Add quotation marks to indicate that
                    # the argument is of string type
                    val = f"'{val}'"
                f.write(f"{arg}: {val}\n")
        # Pickle the args for possible reuse
        with (directory_path / "args.pickle").open(mode="wb") as f:
            pickle.dump(cls._args, f)

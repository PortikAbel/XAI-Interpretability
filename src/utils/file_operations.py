from pathlib import Path

from utils.environment import get_env


def get_package(file_path: str | Path, root: str | Path = None) -> str:
    """
    Get name of the package separated by '.'-s relative to a given path.

    :param file_path: path to the file
    :param root: path to the root. If `None`, then PROJECT_ROOT will be used.
    :return: name of the package
    """
    if type(file_path) is str:
        file_path = Path(file_path)
    if root is None:
        root = get_env("PROJECT_ROOT")
    if type(root) is str:
        root = Path(root)

    if file_path.is_relative_to(root):
        relative = file_path.relative_to(root).with_suffix("")
        return str(relative).replace("/", ".")

    return file_path.stem


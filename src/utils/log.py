import argparse
from pathlib import Path


class Log:

    """
    Object for managing the log directory
    """

    def __init__(self, log_dir: Path):  # Store log in log_dir
        self._log_dir = log_dir
        self._logs = dict()

        self._log_file = self._log_dir / "log.txt"
        self._tqdm_file = (self._log_dir / "tqdm.txt").open(mode="w")

        # Ensure the directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self._log_file.is_file():
            # make log file empty if it already exists
            self._log_file.write_text("")

    @property
    def tqdm_file(self):
        return self._tqdm_file

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir / "checkpoints"

    @property
    def metadata_dir(self):
        return self._log_dir / "metadata"

    def log_message(self, msg: str):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        """
        with self._log_file.open(mode="a") as f:
            f.write(msg + "\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key
            (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise KeyError("Log already exists!")
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with (self.log_dir / f"{log_name}.csv").open(mode="w") as f:
            f.write(",".join((key_name,) + value_names) + "\n")

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception("Log not existent!")
        if len(values) != len(self._logs[log_name][1]):
            raise Exception("Not all required values are logged!")
        # Write a new line with the given values
        with (self.log_dir / f"{log_name}.csv").open(mode="a") as f:
            f.write(",".join(str(v) for v in (key,) + values) + "\n")

    def close(self):
        self._tqdm_file.close()

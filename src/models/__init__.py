from pathlib import Path
from utils.environment import get_env


pretrained_models_dir = Path(get_env("PROJECT_ROOT")) / "pretrained"

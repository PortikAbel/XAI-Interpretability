import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())
pretrained_models_dir = Path(os.getenv("PROJECT_ROOT")) / "models" / "pretrained"

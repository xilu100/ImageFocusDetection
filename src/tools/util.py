from pathlib import Path


def get_root_dir():
    root_path = Path(__file__).resolve().parent.parent.parent
    return root_path

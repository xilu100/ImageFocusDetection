import shutil
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.evaluate import evaluate_all
from src.preprocessing import normalize_raw, segment_nor_img, label_patches, visualize_labels
from src.training import train_all


class StreamToLogger:
    def __init__(self, logger, level, stream):
        self.logger = logger
        self.level = level
        self.stream = stream
        self._buffer = ""

    def write(self, message):
        self.stream.write(message)
        self.stream.flush()

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
        self._buffer = ""
        self.stream.flush()


def setup_logging():
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.__stdout__),
        ],
    )

    stdout_logger = logging.getLogger("stdout")
    stderr_logger = logging.getLogger("stderr")
    stdout_logger.propagate = True
    stderr_logger.propagate = True

    sys.stdout = StreamToLogger(stdout_logger, logging.INFO, sys.__stdout__)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR, sys.__stderr__)

    logging.info("Log file: %s", log_file)


def main():
    setup_logging()

    patch_size = 32
    process = 0
    train = 1
    evaluate = 1

    if process:
        print("=== Step 1: Preprocessing ===")
        delete_folder()

        normalize_raw.normalize_images(patch_size)
        segment_nor_img.segment_images(patch_size)
        label_patches.label(top_percent=75)
        visualize_labels.visualize()

    if train:
        print("=== Step 2: Training ===")
        train_all.train_models(patch_size)

    if evaluate:
        print("=== Step 3: Evaluation ===")
        evaluate_all.evaluate_valid_set()


def delete_folder():
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent

    samples_dir = parent_dir / "data" / "samples"
    samples_label_dir = parent_dir / "data" / "samples_labels"

    valid_samples_dir = parent_dir / "data" / "valid_samples"
    valid_samples_label_dir = parent_dir / "data" / "valid_samples_labels"

    model_dir = parent_dir / "src" / "training" / "model_save"

    folders_to_remove = [
        samples_label_dir,
        samples_dir,
        valid_samples_label_dir,
        valid_samples_dir,
        model_dir,
    ]

    for folder in folders_to_remove:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        else:
            print(f"The folder does not exist: {folder}")


if __name__ == "__main__":
    main()

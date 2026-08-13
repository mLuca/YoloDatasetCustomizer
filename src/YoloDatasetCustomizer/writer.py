import os

import yaml

from .paths import _normalize_path


class YoloDataFileWriter:
    """Create and write a YOLO dataset YAML file for a generated dataset."""

    def __init__(
        self,
        data_set_dir: str = "./",
        class_names: list[str] | None = None,
        train_split: str = "./train",
        val_split: str = "./valid",
        test_split: str = "./test",
    ) -> None:
        """Initialize writer with dataset directory, class names, and split locations."""
        self.data_set_dir = _normalize_path(data_set_dir)
        self.class_names = list(class_names) if class_names is not None else []
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.FILE_NAME = "data.yaml"

    def write(self):
        """Write the YOLO data.yaml file with train/val/test paths and class names."""
        classes = {}
        for i, class_name in enumerate(self.class_names):
            classes[i] = class_name
        data = {
            "train": f"{self.train_split}/images",
            "val": f"{self.val_split}/images",
            "test": f"{self.test_split}/images",
            "names": classes,
            "nc": len(self.class_names),
        }

        with open(os.path.join(self.data_set_dir, self.FILE_NAME), "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

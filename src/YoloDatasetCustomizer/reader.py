import os
from typing import Any

import yaml

from .paths import _SPLIT_NAMES, _normalize_path


class YoloDataFileReader:
    """Read a YOLO dataset YAML file and expose its split paths and class names."""

    def __init__(self, file_path: str):
        """
        Args:
            file_path: Valid path to the data set YAML file
        Raises:
            yaml.YAMLError: When there is an issue with reading the data-set YAML file
            FileNotFoundError: When the data-set YAML file does not exist
            ValueError: When a value in the YAML file is missing or is faulty
            TypeError: When a value comes in an unexpected type
        """
        self.__file_path = _normalize_path(file_path)
        self.__file_dir = os.path.dirname(self.__file_path)
        self.__splits_paths: dict[str, list[str]] = {}
        self.__class_names: list[str] = []

        self.sync_content()

    def get_split_paths_for_split_name(self, split_name: str) -> list[str]:
        """Return a list of dataset split paths for the requested split name."""
        return self.__splits_paths.get(split_name, [])

    def get_class_names(self) -> list[str]:
        """Return the class names defined in the dataset YAML."""
        return self.__class_names

    def get_file_path(self) -> str:
        """Return the absolute path to the loaded YAML file."""
        return self.__file_path

    def get_file_dir(self) -> str:
        """Return the directory containing the loaded dataset YAML file."""
        return self.__file_dir

    def get_indices_for_names(self, class_names: set[str]) -> set[str]:
        """Map a set of class names to their corresponding numeric label indices."""
        ret: set[str] = set()
        for class_name in class_names:
            try:
                idx = self.__class_names.index(class_name)
                ret.add(str(idx))
            except ValueError:
                pass
        return ret

    def sync_content(self):
        """Reload YAML content and refresh split and name mappings."""
        with open(self.__file_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        for split_name in _SPLIT_NAMES:
            self.__splits_paths[split_name] = self.__read_yaml_entry(content, split_name)
        self.__class_names = self.__read_yaml_entry(content, "names")

    def __read_yaml_entry(self, yaml_content: Any, entry_name: str, missing_ok: bool = True) -> list[str]:
        """Normalize YAML entry values into a list of paths or names."""
        entry = yaml_content.get(entry_name)
        ret: list[str] = []
        if entry:
            if isinstance(entry, str):
                ret.append(entry)
            elif isinstance(entry, dict):
                ret.extend(entry.values())
            elif isinstance(entry, list):
                ret.extend(entry)
            else:
                raise TypeError(f"Unexpected type for {entry_name} in: {self.__file_path}")
        else:
            if not missing_ok:
                raise ValueError(f"{entry_name} not defined in: {self.__file_path}")

        return ret

    def __repr__(self) -> str:
        return (
            f"\nYoloDatasetFile(\n"
            f"    path: {self.__file_path}\n"
            f"    split_paths: {self.__splits_paths}\n"
            f"    class_names: {self.__class_names}\n"
            f")\n"
        )

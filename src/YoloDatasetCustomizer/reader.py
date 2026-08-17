import os
from typing import Any

import yaml

from .paths import _SPLIT_NAMES, _normalize_path


class _UniqueKeyLoader(yaml.SafeLoader):
    """A YAML SafeLoader that rejects mappings with duplicate keys.

    PyYAML's default loaders silently keep only the last value for a duplicate
    mapping key (e.g. ``0: car`` followed by ``0: person`` collapses to
    ``{0: 'person'}``)"""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen_keys: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen_keys:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"found duplicate key: {key!r}", key_node.start_mark
                )
            seen_keys.add(key)
        return super().construct_mapping(node, deep)


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
        self.__name_to_index: dict[str, int] = {}

        self.sync_content()

    def get_split_paths_for_split_name(self, split_name: str) -> list[str]:
        """Return a list of dataset split paths for the requested split name."""
        return self.__splits_paths.get(split_name, [])

    def get_class_names(self) -> list[str]:
        """Return the class names defined in the dataset YAML, ordered by their class index."""
        return list(self.__name_to_index.keys())

    def get_file_path(self) -> str:
        """Return the absolute path to the loaded YAML file."""
        return self.__file_path

    def get_file_dir(self) -> str:
        """Return the directory containing the loaded dataset YAML file."""
        return self.__file_dir

    def get_indices_for_names(self, class_names: set[str]) -> set[str]:
        """Map a set of class names to their corresponding numeric label indices."""
        return {str(self.__name_to_index[class_name]) for class_name in class_names if class_name in self.__name_to_index}

    def sync_content(self):
        """Reload YAML content and refresh split and name mappings."""
        with open(self.__file_path, encoding="utf-8") as f:
            content = yaml.load(f, Loader=_UniqueKeyLoader)
        for split_name in _SPLIT_NAMES:
            self.__splits_paths[split_name] = self.__read_yaml_entry(content, split_name)
        self.__name_to_index = self.__read_class_names_entry(content)

    def __read_class_names_entry(self, yaml_content: Any) -> dict[str, int]:
        """Parse the 'names' entry into a name-to-index map, ordered by class index."""
        entry = yaml_content.get("names")
        if not entry:
            return {}

        if isinstance(entry, str):
            return {entry: 0}
        elif isinstance(entry, dict):
            try:
                index_to_name = {int(index): name for index, name in entry.items()}
            except (TypeError, ValueError) as e:
                raise TypeError(f"Class indices in 'names' must be integers in: {self.__file_path}") from e
            if len(index_to_name) != len(entry):
                raise ValueError(f"Duplicate class indices found in 'names' in: {self.__file_path}")
            if len(set(index_to_name.values())) != len(index_to_name):
                raise ValueError(f"Duplicate class names found in 'names' in: {self.__file_path}")
            ordered_indices = sorted(index_to_name)
            return {index_to_name[index]: index for index in ordered_indices}
        elif isinstance(entry, list):
            if len(set(entry)) != len(entry):
                raise ValueError(f"Duplicate class names found in 'names' in: {self.__file_path}")
            return {name: index for index, name in enumerate(entry)}
        else:
            raise TypeError(f"Unexpected type for names in: {self.__file_path}")

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
            f"    class_names: {self.get_class_names()}\n"
            f")\n"
        )

import os
import re

from .paths import _get_unique_path


class LabelFile:
    """Read YOLO label files and copy label lines with remapped class indices."""

    def __init__(self, file_path: str):
        """
        Args:
            file_path: Valid path to the label file
        Raises:
            FileNotFoundError: When the label file does not exist
        """
        self.__class_indices: set[str] = set()
        self.file_path = _get_unique_path(file_path) if os.path.isdir(file_path) else os.path.abspath(file_path)
        if not os.path.exists(self.file_path) or not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"No label file found at: {self.file_path}")

        self.read_label_file()

    def read_label_file(self):
        """Parse the label file and collect all class indices referenced in it."""
        with open(self.file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                match = re.match(r"^(\d+) ", line)
                if not match:
                    print(f"WARNING: No class found on line {i} in label file {self.file_path}")
                else:
                    self.__class_indices.add(match.group(1))

    def get_class_indices(self) -> set[str]:
        """Return the set of class indices in the loaded label file."""
        return self.__class_indices

    def copy_by_class_indices(self, old_to_new_index_match: dict[str, str], dst_path: str, file_name: str = "") -> str | None:
        """Copy the label file to the destination with indices remapped for the new dataset."""
        common_class_indices = self.__class_indices & set(old_to_new_index_match.keys())
        if not common_class_indices:
            return None

        dst_path = os.path.abspath(dst_path)
        if os.path.basename(dst_path) != "labels":
            dst_path = os.path.join(dst_path, "labels")

        if os.path.exists(dst_path) and os.path.isdir(dst_path):
            new_content: list[str] = []
            with open(self.file_path, encoding="utf-8") as f:
                for line in f:
                    match = re.match(r"^(\d+) ", line)
                    if not match:
                        continue
                    elif match.group(1) in common_class_indices:
                        old_index = match.group(1)
                        new_index = old_to_new_index_match[old_index]
                        if new_index != old_index:
                            line = line.replace(old_index, new_index, 1)
                        new_content.append(line)

            if file_name == "":
                file_name = os.path.basename(self.file_path)
            dst_file_path = _get_unique_path(os.path.join(dst_path, file_name))

            with open(dst_file_path, "w", encoding="utf-8") as f:
                f.writelines(new_content)
        else:
            raise FileNotFoundError(f"Directory not found: {dst_path}")

        return dst_file_path

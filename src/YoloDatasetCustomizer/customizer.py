import glob
import os
import shutil

import yaml

from .labels import LabelFile
from .paths import _SPLIT_NAMES, _get_unique_path, _normalize_path
from .reader import YoloDataFileReader
from .writer import YoloDataFileWriter


class YoloDatasetCustomizer:
    """Find and customize YOLO datasets across one or more directory paths."""

    def __init__(self, data_set_paths: list[str]):
        """Initialize with YAML dataset paths or directories to scan for YOLO datasets."""
        self.__found_data_file_paths: set[str] = set()
        self.__data_sets: list[YoloDataFileReader] = []

        self.add_data_sets(data_set_paths)
        self.SUPPORTED_IMAGE_FORMATS = [
            ".avif",
            ".bmp",
            ".dng",
            ".heic",
            ".jp2",
            ".jpeg",
            ".jpg",
            ".mpo",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        ]

    def add_data_sets(self, data_set_paths: list[str]):
        """Discover YOLO data.yaml files from explicit paths or directory trees."""
        data_set_paths = [_normalize_path(path) for path in data_set_paths]

        self.__found_data_file_paths.update(
            [path for path in data_set_paths if os.path.basename(path).lower() == "data.yaml"]
        )

        paths_without_file_ending = [
            path for path in data_set_paths if os.path.basename(path).lower() != "data.yaml"
        ]
        for path in paths_without_file_ending:
            self.__found_data_file_paths.update(glob.glob(os.path.join(path, "**", "data.yaml"), recursive=True))

        for path in self.__found_data_file_paths:
            try:
                ydfr = YoloDataFileReader(path)
                self.__data_sets.append(ydfr)
            except (yaml.YAMLError, TypeError, ValueError) as e:
                print(f"ERROR: Issue loading YAML file at {path}: {e}")
            except FileNotFoundError:
                print(f"ERROR: Yaml file does not exist at {path}")

    def get_found_data_file_paths(self) -> set[str]:
        """Return all discovered YOLO dataset YAML file paths."""
        return self.__found_data_file_paths

    def get_found_class_names(self) -> set[str]:
        """Return the union of all class names from discovered datasets."""
        found_classes: set[str] = set()
        for ydf in self.__data_sets:
            found_classes.update(ydf.get_class_names())
        return found_classes

    def create_new_dataset_for_class_names(
        self,
        class_names: set[str],
        dst_path: str = ".",
        data_set_name: str = "new_dataset",
        ignore_img_formats: bool = False,
    ) -> bool:
        """Create a filtered dataset containing only the requested classes."""
        if len(class_names) == 0:
            print("ERROR: No classes selected for new dataset.")
            return False

        new_dataset_path = os.path.join(_normalize_path(dst_path), data_set_name)
        new_dataset_path = _get_unique_path(new_dataset_path)

        new_class_indeces = self.__generate_new_class_indeces(class_names)
        print(f"INFO: New class indeces: {new_class_indeces}")

        for data_set in self.__data_sets:
            common_class_names = set(data_set.get_class_names()) & class_names
            if not common_class_names:
                continue

            old_to_new_index_match: dict[str, str] = {}
            for class_name in common_class_names:
                old_index = str(data_set.get_class_names().index(class_name))
                new_index = new_class_indeces[class_name]
                old_to_new_index_match[old_index] = new_index

            for split_name in _SPLIT_NAMES:
                for split_path in data_set.get_split_paths_for_split_name(split_name):
                    old_images_path = os.path.join(_normalize_path(data_set.get_file_dir()), split_path)
                    old_labels_path = self.__generate_label_path_from_img_path(old_images_path)

                    missing_path = next(
                        (path for path in (old_labels_path, old_images_path) if not os.path.exists(path)), None
                    )
                    if missing_path is not None:
                        print(f"WARNING: Path '{missing_path}' does not exist! Skipping it.")
                        continue

                    new_labels_path = os.path.join(new_dataset_path, split_name, "labels")
                    new_images_path = os.path.join(new_dataset_path, split_name, "images")
                    os.makedirs(new_labels_path, exist_ok=True)
                    os.makedirs(new_images_path, exist_ok=True)

                    for original_label_file_path in glob.glob(os.path.join(old_labels_path, "*.txt")):
                        label_file = LabelFile(original_label_file_path)
                        new_label_file_path = label_file.copy_by_class_indeces(old_to_new_index_match, new_labels_path)
                        if new_label_file_path:
                            file_name = os.path.basename(original_label_file_path)
                            file_name_no_ext = file_name[: file_name.rfind(".")]
                            corresponding_image_files = glob.glob(os.path.join(old_images_path, f"{file_name_no_ext}.*"))
                            for img in corresponding_image_files:
                                try:
                                    _, ext = os.path.splitext(img)
                                except Exception:
                                    print(f"ERROR: Not a valid image, extension missing for: {img}")
                                    return False

                                if not ignore_img_formats and ext.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                                    print(
                                        f"ERROR: Image format not supported by yolo: {img}\n"
                                        f"Supported formats are: {self.SUPPORTED_IMAGE_FORMATS}"
                                    )
                                    return False

                                new_image_name, _ = os.path.splitext(os.path.basename(new_label_file_path))
                                new_image_name = new_image_name + ext
                                destination = os.path.join(new_images_path, new_image_name)
                                try:
                                    shutil.copyfile(img, destination)
                                except OSError as e:
                                    print(
                                        f"ERROR: Image could not be copied:\n"
                                        f"Source: {img}\nDestination: {destination}\n"
                                        f"Exception was: {e}"
                                    )
                                    return False

        data_file_writer = YoloDataFileWriter(new_dataset_path, list(new_class_indeces.keys()))
        try:
            data_file_writer.write()
        except Exception as e:
            print(f"ERROR: Could not write new data.yaml to {new_dataset_path}. Generation of {data_set_name} is incomplete!")
            print(f"Exception was: {e}")
            return False

        return True

    def __generate_new_class_indeces(self, class_names: set[str]) -> dict[str, str]:
        """Generate a fresh label index mapping for the selected class names."""
        ret: dict[str, str] = {}
        for i, name in enumerate(class_names):
            ret[name] = str(i)
        return ret

    def __generate_label_path_from_img_path(self, img_path: str) -> str:
        """Convert an images directory path into its corresponding labels directory path."""
        img_path = _normalize_path(img_path)
        return os.path.normpath(os.path.join(os.path.dirname(img_path), "labels"))

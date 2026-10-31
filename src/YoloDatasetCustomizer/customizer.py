import glob
import os
import shutil

import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn

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

        new_class_indices = self.__generate_new_class_indices(class_names)
        print(f"INFO: New class indices: {new_class_indices}")

        # Always create every split folder so the folder structure matches what data.yaml references,
        # even when no source dataset contributed images to a split.
        for split_name in _SPLIT_NAMES:
            os.makedirs(os.path.join(new_dataset_path, split_name, "labels"), exist_ok=True)
            os.makedirs(os.path.join(new_dataset_path, split_name, "images"), exist_ok=True)

        # Plan first, so every stage knows how many files it is going to copy
        planned_files: dict[str, list[tuple[LabelFile, dict[str, list[str]], dict[str, str]]]] = {
            split_name: [] for split_name in _SPLIT_NAMES
        }
        for data_set in self.__data_sets:
            common_class_names = set(data_set.get_class_names()) & class_names
            if not common_class_names:
                continue

            old_to_new_index_match: dict[str, str] = {}
            for class_name in common_class_names:
                old_indices = data_set.get_indices_for_names({class_name})
                if not old_indices:
                    continue
                old_index = next(iter(old_indices))
                new_index = new_class_indices[class_name]
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

                    # Index images by exact stem so 'car.*' can't pick up e.g. 'car.v2.jpg'
                    images_by_stem: dict[str, list[str]] = {}
                    for image_name in os.listdir(old_images_path):
                        stem = os.path.splitext(image_name)[0]
                        images_by_stem.setdefault(stem, []).append(os.path.join(old_images_path, image_name))

                    for original_label_file_path in glob.glob(os.path.join(old_labels_path, "*.txt")):
                        label_file = LabelFile(original_label_file_path)
                        if label_file.get_class_indices() & set(old_to_new_index_match):
                            planned_files[split_name].append((label_file, images_by_stem, old_to_new_index_match))

        # One bar per stage (white, filled green), all visible from the start
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(style="white", complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
        ) as progress:
            stage_tasks = {
                split_name: progress.add_task(f"[{stage}/{len(_SPLIT_NAMES)}] {split_name:<5}", total=len(planned_files[split_name]))
                for stage, split_name in enumerate(_SPLIT_NAMES, start=1)
            }
            for split_name in _SPLIT_NAMES:
                new_labels_path = os.path.join(new_dataset_path, split_name, "labels")
                new_images_path = os.path.join(new_dataset_path, split_name, "images")
                for label_file, images_by_stem, old_to_new_index_match in planned_files[split_name]:
                    if not self.__copy_label_and_images(
                        label_file, images_by_stem, old_to_new_index_match, new_labels_path, new_images_path, ignore_img_formats
                    ):
                        return False
                    progress.advance(stage_tasks[split_name])

        # Pass the split folder names explicitly so data.yaml points at the folders created above
        data_file_writer = YoloDataFileWriter(
            new_dataset_path,
            list(new_class_indices.keys()),
            train_split="train",
            val_split="val",
            test_split="test",
        )
        try:
            data_file_writer.write()
        except Exception as e:
            print(f"ERROR: Could not write new data.yaml to {new_dataset_path}. Generation of {data_set_name} is incomplete!")
            print(f"Exception was: {e}")
            return False

        print(f"SUCCESS: Dataset '{os.path.basename(new_dataset_path)}' created at: {new_dataset_path}")
        return True

    def __copy_label_and_images(
        self,
        label_file: LabelFile,
        images_by_stem: dict[str, list[str]],
        old_to_new_index_match: dict[str, str],
        new_labels_path: str,
        new_images_path: str,
        ignore_img_formats: bool,
    ) -> bool:
        """Copy one label file with remapped indices plus its images. Returns False on error."""
        new_label_file_path = label_file.copy_by_class_indices(old_to_new_index_match, new_labels_path)
        if not new_label_file_path:
            return True

        file_name_no_ext = os.path.splitext(os.path.basename(label_file.file_path))[0]
        for img in images_by_stem.get(file_name_no_ext, []):
            _, ext = os.path.splitext(img)

            if not ignore_img_formats and ext.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                print(
                    f"\nERROR: Image format not supported by yolo: {img}\n"
                    f"Supported formats are: {self.SUPPORTED_IMAGE_FORMATS}"
                )
                return False

            new_image_name, _ = os.path.splitext(os.path.basename(new_label_file_path))
            destination = os.path.join(new_images_path, new_image_name + ext)
            try:
                shutil.copyfile(img, destination)
            except OSError as e:
                print(
                    f"\nERROR: Image could not be copied:\n"
                    f"Source: {img}\nDestination: {destination}\n"
                    f"Exception was: {e}"
                )
                return False
        return True

    def __generate_new_class_indices(self, class_names: set[str]) -> dict[str, str]:
        """Generate a fresh label index mapping for the selected class names."""
        ret: dict[str, str] = {}
        for i, name in enumerate(class_names):
            ret[name] = str(i)
        return ret

    def __generate_label_path_from_img_path(self, img_path: str) -> str:
        """Convert an images directory path into its corresponding labels directory path."""
        img_path = _normalize_path(img_path)
        return os.path.normpath(os.path.join(os.path.dirname(img_path), "labels"))

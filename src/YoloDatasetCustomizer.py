import yaml
import glob
import os
import re

class YoloDataFile():
    def __init__(self, file_path: str):
        """
        Args:
            file_path: Valid path to the data set YAML file
        Raises:
            yaml.YAMLError: When there is an  issue with reading the data-set  YAML file
            FileNotFoundError: When the data-set YAML file does not exist
            ValueError: When a value in the YAML file is missing or is faulty
            TypeError: When a value comes in an unexpected type
        """
        self.__file_path = file_path
        self.__training_split_path = ""
        self.__validation_split_path = ""
        self.__testing_split_path = ""
        self.__class_names = []

        self.sync_content()
        
        
    
    def get_training_split_path(self)-> str:
        return self.__training_split_path
    def get_validation_split_path(self)-> str:
        return self.__validation_split_path
    def get_testing_split_path(self)-> str:
        return self.__testing_split_path
    def get_class_names(self) -> [str]:
        return self.__class_names
    def get_file_path(self) -> str:
        return self.__file_path

    def sync_content(self):
        with open(self.__file_path) as f:
            content = yaml.safe_load(f)
            self.__training_split_path = content.get("train", "")
            self.__validation_split_path = content.get("val", "")
            self.__testing_split_path = content.get("test", "")
            names = content.get("names")
            if names:
                if isinstance(names, dict):
                    self.__class_names.extend(names.values())
                elif isinstance(names, list):
                    self.__class_names.extend(names)
                else:
                    raise TypeError(f"Unexpected type for class names in: {self.__file_path}")
            else:
                raise ValueError(f"No classes are defined in: {self.__file_path}")
    
    def __repr__(self) -> str:
        return f"\nYoloDatasetFile(\n    path: {self.__file_path}\n    training_split_path: {self.__training_split_path}\n    validation_split_path: {self.__validation_split_path}\n    testing_split_path: {self.__testing_split_path}\n    class_names: {self.__class_names}\n)\n"

class LabelFile():
    def __init__(self, file_path: str):
        """
        Args:
            file_path: Valid path to the label file
        Raises:
            FileNotFoundError: When the lablelfile does not exist
        """
        self.class_indeces = set()
        self.file_path = os.path.abspath(file_path)
        if not os.path.exists(self.file_path) or not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"No label file found at: {self.file_path}")
        
        self.read_label_file()
    
    def read_label_file(self):
        with open(self.file_path) as f:
            for i, line in enumerate(f):
                match = re.match(r"^(\d+) ", line)
                if not match.group(1):
                    print(f"WARNING: No class found on line {i} in label file {self.file_path}")
                else:
                    self.class_indeces.add(match.group(1))

    def get_class_indices(self) -> set:
        return self.class_indeces
    
    def copy_with_classes(self, class_indeces: set(), dst_path: str) -> bool:
        common_class_indeces = self.class_indeces & class_indeces
        if not common_class_indeces:
            return False

        dst_path = os.path.abspath(dst_path)
        if os.path.exists(dst_path) and os.path.isdir(dst_path):
            new_content = []
            with open(self.file_path) as f:
                for line in f:
                    match = re.match(r"^(\d+) ", line)
                    if not match.group(1):
                        continue
                    elif match.group(1) in common_class_indeces:
                        new_content.append(line)
            
            dst_file_path = "/".join(dast_path, os.path.basename(self.file_path))
            with open(dst_file_path, "w") as f:
                f.write(new_content)
        else:
            raise FileNotFoundError(f"Directory not found: {dst_path}")
        
        return True




class YoloDatasetCustomizer():
    def __init__(self, data_set_paths: [str] ):
        self.__found_data_file_paths = set()
        self.__data_sets: list(YoloDataFile) = []

        self.add_data_sets(data_set_paths)
        self.DATA_SUB_DIRS = ["train", "valid", "test"]
        self.SUPPORTED_IMAGE_FORMATS = ["avif", "bmp", "dng", "heic", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"]

    def add_data_sets(self, data_set_paths: [str]):
        data_set_paths = list(map(lambda p: os.path.abspath(p), data_set_paths))

        self.__found_data_file_paths.update([path for path in data_set_paths if path.endswith("data.yaml")])

        paths_without_file_ending = [path for path in data_set_paths if not path.endswith("data.yaml")]
        for path in paths_without_file_ending:
            self.__found_data_file_paths.update(glob.glob(path + "/**/data.yaml", recursive=True))

        for path in self.__found_data_file_paths:
            try:
                ydf = YoloDataFile(path)
            except (yaml.YAMLError, TypeError, ValueError) as e:
                print(f"ERROR: Issue loading YAML file at {path}: {e}")
            except FileNotFoundError:
                print(f"ERROR: Yaml file does not exist at {path}")
            print(f"Adding Dataset: {ydf}")
            self.__data_sets.append(ydf)


    def get_found_data_file_paths(self) -> [str]:
        return self.__found_data_file_paths
    
    def get_found_class_names(self) -> set(str):
        found_classes = set()
        for ydf in self.__data_sets:
            found_classes.update(ydf.get_class_names())
        return found_classes

    def create_new_dataset_for_class_names(self, class_names: set(str), dst_path: str = ".", data_set_name: str = "new_dataset"):
        new_dataset_path = "/".join(os.abspath(dst_path), data_set_name)
        new_dataset_path = __get_unique_path(new_dataset_path)

        

        
        for data_set in self.__data_sets:
            common_classes = set(data_set.get_class_names) & class_names
            if not common_classes:
                continue
            for sub_dir in self.DATA_SUB_DIRS:
                root_path = "/".join(os.path.abspath(data_set.get_file_path), sub_dir)
                labels_path = "/".join(root_path, "labels")
                images_path = "/".join(root_path, "images")

                if not os.path.exists(labels_dir) or not os.path.exists(images_path):
                    print(f"WARNING: labels/images do not exist in {root_path}!")
                    continue

                new_labels_path = "/".join(new_dataset_path, sub_dir, "labels")
                new_iamges_path = "/".join(new_dataset_path, sub_dir, "images")
                os.makedirs(new_labels_path)
                os.makedirs(new_images_path)

                for label_file_path in glob.glob(f"{labels_path}/*.txt"):
                    label_file = LabelFile(label_file_path)
                    if label_file.copy_with_classes(common_classes, new_dataset_path):

    
    def __get_unique_path(self, path:str) -> str:
        path = os.abspath(path)
        unique_path = path
        index = 1

        while True:
            if os.path.exists(unique_path):
                unique_path = "".join(path, f"_{index}")
                index += 1
            else:
                return unique_path






        
    

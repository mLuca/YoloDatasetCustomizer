import yaml
import glob
import os
import shutil
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
        self.__file_path = os.path.abspath(file_path)
        self.__file_dir = os.path.dirname(self.__file_path)
        self.__training_split_path = ""
        self.__validation_split_path = ""
        self.__testing_split_path = ""
        self.__class_names: list[str] = []

        self.sync_content()
        
        
    
    def get_training_split_path(self)-> str:
        return self.__training_split_path
    def get_validation_split_path(self)-> str:
        return self.__validation_split_path
    def get_testing_split_path(self)-> str:
        return self.__testing_split_path
    def get_class_names(self) -> list[str]:
        return self.__class_names
    def get_file_path(self) -> str:
        return self.__file_path
    def get_file_dir(self) -> str:
            return self.__file_dir
    def get_indices_for_names(self, class_names: set[str]) -> set[str]:
        ret: set[str] = set()
        for class_name in class_names:
            try:
                idx = self.__class_names.index(class_name)
                ret.add(str(idx))
            except:
                pass
        return ret

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
        self.__class_indeces: set[str] = set()
        self.file_path = os.path.abspath(file_path)
        if not os.path.exists(self.file_path) or not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"No label file found at: {self.file_path}")
        
        self.read_label_file()
    
    def read_label_file(self):
        with open(self.file_path) as f:
            for i, line in enumerate(f):
                match = re.match(r"^(\d+) ", line)
                if not match:
                    print(f"WARNING: No class found on line {i} in label file {self.file_path}")
                else:
                    self.__class_indeces.add(match.group(1))

    def get_class_indices(self) -> set[str]:
        return self.__class_indeces
    
    def copy_by_class_indeces(self, class_indeces: set[str], dst_path: str) -> bool:
        common_class_indeces = self.__class_indeces & class_indeces
        print(f"INFO: Requested class indices: {class_indeces}\nAvailabhle indices: {self.__class_indeces}")
        if not common_class_indeces:
            return False
        
        dst_path = os.path.abspath(dst_path)
        if not os.path.basename(dst_path) == "labels":
            dst_path = os.path.join(dst_path,"labels")

        if os.path.exists(dst_path) and os.path.isdir(dst_path):
            new_content: list[str] = []
            with open(self.file_path) as f:
                for line in f:
                    match = re.match(r"^(\d+) ", line)
                    if not match:
                        continue
                    elif match.group(1) in common_class_indeces:
                        new_content.append(line)
            
            dst_file_path = "/".join([dst_path, os.path.basename(self.file_path)])
            with open(dst_file_path, "w") as f:
                f.writelines(new_content)
        else:
            raise FileNotFoundError(f"Directory not found: {dst_path}")
        
        return True




class YoloDatasetCustomizer():
    def __init__(self, data_set_paths: list[str]):
        self.__found_data_file_paths = set()
        self.__data_sets: list[YoloDataFile] = []

        self.add_data_sets(data_set_paths)
        self.DATA_SUB_DIRS = ["train", "valid", "test"]
        self.SUPPORTED_IMAGE_FORMATS = ["avif", "bmp", "dng", "heic", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"]

    def add_data_sets(self, data_set_paths: list[str]):
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
            self.__data_sets.append(ydf)


    def get_found_data_file_paths(self) -> list[str]:
        return self.__found_data_file_paths
    
    def get_found_class_names(self) -> set[str]:
        found_classes = set()
        for ydf in self.__data_sets:
            found_classes.update(ydf.get_class_names())
        return found_classes

    def create_new_dataset_for_class_names(self, class_names: set[str], dst_path: str = ".", data_set_name: str = "new_dataset") -> bool:
        new_dataset_path = "/".join([os.path.abspath(dst_path), data_set_name])
        new_dataset_path = self.__get_unique_path(new_dataset_path)

        for data_set in self.__data_sets:
            print(f"INFO: Processeing {data_set.get_file_dir()}")
            common_class_names = set(data_set.get_class_names()) & class_names
            if not common_class_names:
                continue

            for sub_dir in self.DATA_SUB_DIRS:
                old_dataset_path = "/".join([os.path.abspath(data_set.get_file_dir()), sub_dir])
                old_labels_path = "/".join([old_dataset_path, "labels"])
                old_images_path = "/".join([old_dataset_path, "images"])

                if not os.path.exists(old_labels_path) or not os.path.exists(old_images_path):
                    print(f"WARNING: labels/images do not exist in {old_dataset_path}!")
                    continue

                new_labels_path = "/".join([new_dataset_path, sub_dir, "labels"])
                new_images_path = "/".join([new_dataset_path, sub_dir, "images"])
                os.makedirs(new_labels_path, exist_ok=True)
                os.makedirs(new_images_path, exist_ok=True)

                for label_file_path in glob.glob(f"{old_labels_path}/*.txt"):
                    label_file = LabelFile(label_file_path)
                    if label_file.copy_by_class_indeces(data_set.get_indices_for_names(common_class_names), new_labels_path):
                        file_name = os.path.basename(label_file_path)
                        file_name_no_ext = file_name[:file_name.rfind('.')]
                        corresponding_image_files = glob.glob(f"{old_images_path}/{file_name_no_ext}.*")
                        for img in corresponding_image_files:
                            try:
                                ext = img[img.rindex(".") + 1 :]
                            except:
                                print(f"ERROR: Not a valid image, extension missing for: {img}")
                                return False

                            if ext.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                                print(f"ERROR: Image format not supported by yolo: {img}\nSupported formats are: {self.SUPPORTED_IMAGE_FORMATS}")
                                return False
                            
                            destination = "/".join([new_images_path, os.path.basename(img)])
                            if not shutil.copyfile(img, destination):
                                print(f"ERROR: Image could not be copied:\nSource: {img}\nDestination: {destination}")
                                return False
        return True
                        


    
    def __get_unique_path(self, path:str) -> str:
        path = os.path.abspath(path)
        unique_path = path
        index = 1

        while True:
            if os.path.exists(unique_path):
                unique_path = "".join([path, f"_{index}"])
                index += 1
            else:
                return unique_path






        
    

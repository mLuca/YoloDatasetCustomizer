import yaml
import glob
import os
import shutil
import re
from typing import Any

def get_unique_path(path:str) -> str:
    path = os.path.abspath(path)
    base, ext = os.path.splitext(path)
    unique_path = path
    index = 1
    while True:
        if os.path.exists(unique_path):
            unique_path = "".join([base, f"_{index}", ext])
            index += 1
        else:
            return unique_path


class YoloDataFileWriter():
    def __init__(self, data_set_dir: str = "./", class_names: list[str] = [], train_split: str = "./train", val_split: str = "./valid", test_split: str = "./test") -> None:
        self.data_set_dir = os.path.abspath(data_set_dir)
        self.class_names = class_names
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.FILE_NAME = "data.yaml"

    def write(self):
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

        with open(os.path.join(self.data_set_dir, self.FILE_NAME), "w") as f:
            yaml.dump(data,f,  default_flow_style=False)



        

class YoloDataFileReader():
    def __init__(self, file_path: str):
        """
        Args:
            file_path: Valid path to the data set YAML file
        Raises:
            yaml.YAMLError: When there is an  issue with reading the data-set YAML file
            FileNotFoundError: When the data-set YAML file does not exist
            ValueError: When a value in the YAML file is missing or is faulty
            TypeError: When a value comes in an unexpected type
        """
        self.__file_path = os.path.abspath(file_path)
        self.__file_dir = os.path.dirname(self.__file_path)
        self.__training_split_paths: list[str] = []
        self.__validation_split_paths: list[str] = []
        self.__testing_split_paths: list[str] = []
        self.__class_names: list[str] = []

        self.sync_content()
        
        
    
    def get_training_split_paths(self)-> list[str]:
        return self.__training_split_paths
    def get_validation_split_paths(self)-> list[str]:
        return self.__validation_split_paths
    def get_testing_split_paths(self)-> list[str]:
        return self.__testing_split_paths
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
            self.__training_split_paths = self.__read_yaml_entry(content, "train")
            self.__validation_split_paths = self.__read_yaml_entry(content, "val")
            self.__testing_split_paths = self.__read_yaml_entry(content, "test")
            self.__class_names = self.__read_yaml_entry(content, "names")
            

    def __read_yaml_entry(self, yaml_content: Any, entry_name: str, missing_ok:bool = True) -> list[str]:
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
        return f"\nYoloDatasetFile(\n    path: {self.__file_path}\n    training_split_path: {self.__training_split_paths}\n    validation_split_path: {self.__validation_split_paths}\n    testing_split_path: {self.__testing_split_paths}\n    class_names: {self.__class_names}\n)\n"

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
    
    def copy_by_class_indeces(self, old_to_new_index_match: dict[str,str], dst_path: str, file_name: str = "") -> str|None:
        
        common_class_indeces = self.__class_indeces & set(old_to_new_index_match.keys())
        if not common_class_indeces:
            return None
        
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
                        old_index = match.group(1)
                        new_index = old_to_new_index_match[old_index]
                        if new_index != old_index:
                            line = line.replace(old_index, new_index, 1)
                            print(f"INFO: Replaceing index {old_index} with {new_index}:\n {line}")
                        new_content.append(line)

            if file_name == "":
                file_name = os.path.basename(self.file_path)
            dst_file_path = get_unique_path("/".join([dst_path, file_name]))

            with open(dst_file_path, "w") as f:
                f.writelines(new_content)
        else:
            raise FileNotFoundError(f"Directory not found: {dst_path}")
        
        return dst_file_path




class YoloDatasetCustomizer():
    def __init__(self, data_set_paths: list[str]):
        self.__found_data_file_paths: set[str] = set()
        self.__data_sets: list[YoloDataFileReader] = []

        self.add_data_sets(data_set_paths)
        self.SPLIT_DIRS = ["train", "valid", "test"]
        self.SUPPORTED_IMAGE_FORMATS = [".avif", ".bmp", ".dng", ".heic", ".jp2", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"]

    def add_data_sets(self, data_set_paths: list[str]):
        data_set_paths = list(map(lambda p: os.path.abspath(p), data_set_paths))

        self.__found_data_file_paths.update([path for path in data_set_paths if path.endswith("data.yaml")])

        paths_without_file_ending = [path for path in data_set_paths if not path.endswith("data.yaml")]
        for path in paths_without_file_ending:
            self.__found_data_file_paths.update(glob.glob(path + "/**/data.yaml", recursive=True))

        for path in self.__found_data_file_paths:
            try:
                ydfr = YoloDataFileReader(path)
                self.__data_sets.append(ydfr)
            except (yaml.YAMLError, TypeError, ValueError) as e:
                print(f"ERROR: Issue loading YAML file at {path}: {e}")
            except FileNotFoundError:
                print(f"ERROR: Yaml file does not exist at {path}")
            


    def get_found_data_file_paths(self) -> set[str]:
        return self.__found_data_file_paths
    
    def get_found_class_names(self) -> set[str]:
        found_classes: set[str] = set()
        for ydf in self.__data_sets:
            found_classes.update(ydf.get_class_names())
        return found_classes

    def create_new_dataset_for_class_names(self, class_names: set[str], dst_path: str = ".", data_set_name: str = "new_dataset", ignore_img_formats: bool = False) -> bool:
        if len(class_names) == 0:
            print("Error: No classes selected for new dataset.")
            return False
        
        new_dataset_path = "/".join([os.path.abspath(dst_path), data_set_name])
        new_dataset_path = get_unique_path(new_dataset_path)

        new_class_indeces = self.__generate_new_class_indeces(class_names)
        print(f"INFO: New class indeces: {new_class_indeces}")

        for data_set in self.__data_sets:
            print(f"INFO: Processeing {data_set.get_file_dir()}")
            common_class_names = set(data_set.get_class_names()) & class_names
            if not common_class_names:
                continue
            else:
                old_to_new_index_match: dict[str, str] = {}
                for class_name in common_class_names:
                    old_index = str(data_set.get_class_names().index(class_name))
                    new_index = new_class_indeces[class_name]
                    old_to_new_index_match[old_index] = new_index
            

            for split_dir in self.SPLIT_DIRS:
                old_dataset_path = "/".join([os.path.abspath(data_set.get_file_dir()), split_dir])
                old_labels_path = "/".join([old_dataset_path, "labels"])
                old_images_path = "/".join([old_dataset_path, "images"])

                if not os.path.exists(old_labels_path) or not os.path.exists(old_images_path):
                    print(f"WARNING: labels/images do not exist in {old_dataset_path}! Skipping it.")
                    continue

                new_labels_path = "/".join([new_dataset_path, split_dir, "labels"])
                new_images_path = "/".join([new_dataset_path, split_dir, "images"])
                os.makedirs(new_labels_path, exist_ok=True)
                os.makedirs(new_images_path, exist_ok=True)

                for original_label_file_path in glob.glob(f"{old_labels_path}/*.txt"):
                    label_file = LabelFile(original_label_file_path)

                    new_label_file_path = label_file.copy_by_class_indeces(old_to_new_index_match, new_labels_path)
                    if new_label_file_path:
                        file_name = os.path.basename(original_label_file_path)
                        file_name_no_ext = file_name[:file_name.rfind('.')]
                        corresponding_image_files = glob.glob(f"{old_images_path}/{file_name_no_ext}.*")
                        for img in corresponding_image_files:
                            try:
                                _ , ext = os.path.splitext(img)
                            except:
                                print(f"ERROR: Not a valid image, extension missing for: {img}")
                                return False

                            if not ignore_img_formats and ext.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                                print(f"ERROR: Image format not supported by yolo: {img}\nSupported formats are: {self.SUPPORTED_IMAGE_FORMATS}")
                                return False

                            new_image_name,_ = os.path.splitext(os.path.basename(new_label_file_path))
                            new_image_name = new_image_name + ext
                            destination = "/".join([new_images_path, new_image_name])
                            if not shutil.copyfile(img, destination):
                                print(f"ERROR: Image could not be copied:\nSource: {img}\nDestination: {destination}")
                                return False

        data_file_writer = YoloDataFileWriter(new_dataset_path, list(new_class_indeces.keys()))
        try:
            data_file_writer.write()
        except Exception as e:
            print(f"ERROR: Could not write new data.yaml to {new_dataset_path}. Generation of {data_set_name} is incomplete!")
            print(f"Exception was: {e}")
            return False


        return True

    def __generate_new_class_indeces(self, class_names:set[str])-> dict[str, str]:
        ret: dict[str, str] = {}
        for i, name in enumerate(class_names):
            ret[name] = str(i)
        return ret
    







        
    

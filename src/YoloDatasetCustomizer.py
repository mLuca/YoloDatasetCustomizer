import yaml
import glob
import os

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


class YoloDatasetCustomizer():
    def __init__(self, data_set_paths: [str] ):
        self.__found_data_file_paths = set()
        self.__data_sets: list(YoloDataFile) = []

        self.add_data_sets(data_set_paths)        

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

    def create_dataset_for_class_names(self, class_names: set(str)):
        
    

import yaml
import glob
import os
import shutil
import re
from typing import Any

_SPLIT_NAMES =["train", "val", "test"]

def _get_unique_path(path:str) -> str:
    """Return an unused file path by appending a numeric suffix when needed."""
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
    """Create and write a YOLO dataset YAML file for a generated dataset."""
    def __init__(self, data_set_dir: str = "./", class_names: list[str] = [], train_split: str = "./train", val_split: str = "./valid", test_split: str = "./test") -> None:
        """Initialize writer with dataset directory, class names, and split locations."""
        self.data_set_dir = os.path.abspath(data_set_dir)
        self.class_names = class_names
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

        with open(os.path.join(self.data_set_dir, self.FILE_NAME), "w") as f:
            yaml.dump(data,f,  default_flow_style=False)



        

class YoloDataFileReader():
    """Read a YOLO dataset YAML file and expose its split paths and class names."""
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
        self.__splits_paths: dict[str,list[str]] = {}
        self.__class_names: list[str] = []

        self.sync_content()

    def get_split_paths_for_split_name(self, split_name:str) -> list[str]:
        """Return a list of dataset split paths for the requested split name."""
        return self.__splits_paths.get(split_name,[]) 
    
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
            except:
                pass
        return ret

    def sync_content(self):
        """Reload YAML content and refresh split and name mappings."""
        with open(self.__file_path) as f:
            content = yaml.safe_load(f)
            for split_name in _SPLIT_NAMES:
                self.__splits_paths[split_name] = self.__read_yaml_entry(content, split_name)
            self.__class_names = self.__read_yaml_entry(content, "names")
            

    def __read_yaml_entry(self, yaml_content: Any, entry_name: str, missing_ok:bool = True) -> list[str]:
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
        return f"\nYoloDatasetFile(\n    path: {self.__file_path}\n    training_split_path: {self.__training_split_paths}\n    validation_split_path: {self.__validation_split_paths}\n    testing_split_path: {self.__testing_split_paths}\n    class_names: {self.__class_names}\n)\n"

class LabelFile():
    """Read YOLO label files and copy label lines with remapped class indices."""
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
        """Parse the label file and collect all class indices referenced in it."""
        with open(self.file_path) as f:
            for i, line in enumerate(f):
                match = re.match(r"^(\d+) ", line)
                if not match:
                    print(f"WARNING: No class found on line {i} in label file {self.file_path}")
                else:
                    self.__class_indeces.add(match.group(1))

    def get_class_indices(self) -> set[str]:
        """Return the set of class indices in the loaded label file."""
        return self.__class_indeces
    
    def copy_by_class_indeces(self, old_to_new_index_match: dict[str,str], dst_path: str, file_name: str = "") -> str|None:
        """Copy the label file to the destination with indices remapped for the new dataset."""
        
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
                        new_content.append(line)

            if file_name == "":
                file_name = os.path.basename(self.file_path)
            dst_file_path = _get_unique_path("/".join([dst_path, file_name]))

            with open(dst_file_path, "w") as f:
                f.writelines(new_content)
        else:
            raise FileNotFoundError(f"Directory not found: {dst_path}")
        
        return dst_file_path




class YoloDatasetCustomizer():
    """Find and customize YOLO datasets across one or more directory paths."""
    def __init__(self, data_set_paths: list[str]):
        """Initialize with YAML dataset paths or directories to scan for YOLO datasets."""
        self.__found_data_file_paths: set[str] = set()
        self.__data_sets: list[YoloDataFileReader] = []

        self.add_data_sets(data_set_paths)
        self.SUPPORTED_IMAGE_FORMATS = [".avif", ".bmp", ".dng", ".heic", ".jp2", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"]

    def add_data_sets(self, data_set_paths: list[str]):
        """Discover YOLO data.yaml files from explicit paths or directory trees."""
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
        """Return all discovered YOLO dataset YAML file paths."""
        return self.__found_data_file_paths
    
    def get_found_class_names(self) -> set[str]:
        """Return the union of all class names from discovered datasets."""
        found_classes: set[str] = set()
        for ydf in self.__data_sets:
            found_classes.update(ydf.get_class_names())
        return found_classes

    def create_new_dataset_for_class_names(self, class_names: set[str], dst_path: str = ".", data_set_name: str = "new_dataset", ignore_img_formats: bool = False) -> bool:
        """Create a filtered dataset containing only the requested classes."""
        if len(class_names) == 0:
            print("ERROR: No classes selected for new dataset.")
            return False
        
        # Build a unique target folder for the new dataset
        new_dataset_path = "/".join([os.path.abspath(dst_path), data_set_name])
        new_dataset_path = _get_unique_path(new_dataset_path)

        # Create a new label index mapping for the selected classes
        new_class_indeces = self.__generate_new_class_indeces(class_names)
        print(f"INFO: New class indeces: {new_class_indeces}")

        # Iterate over all loaded datasets and keep only objects for selected classes
        for data_set in self.__data_sets:
            common_class_names = set(data_set.get_class_names()) & class_names
            if not common_class_names:
                continue
            else:
                # Map old indices from the source dataset to new indices for the target dataset
                old_to_new_index_match: dict[str, str] = {}
                for class_name in common_class_names:
                    old_index = str(data_set.get_class_names().index(class_name))
                    new_index = new_class_indeces[class_name]
                    old_to_new_index_match[old_index] = new_index
            

            # Copy files from each split (train/val/test) if they contain selected classes
            for split_name in _SPLIT_NAMES:
                for split_path in data_set.get_split_paths_for_split_name(split_name):
                    old_images_path = os.path.join(os.path.abspath(data_set.get_file_dir()), split_path)
                    old_labels_path = self.__generate_label_path_from_img_path(old_images_path)             

                    # Skip missing split directories and continue with the next path
                    for path in [old_labels_path, old_images_path]:
                        if not os.path.exists(path):
                            print(f"WARNING: Path '{path}' does not exist! Skipping it.")
                            continue

                    new_labels_path = "/".join([new_dataset_path, split_name, "labels"])
                    new_images_path = "/".join([new_dataset_path, split_name, "images"])
                    os.makedirs(new_labels_path, exist_ok=True)
                    os.makedirs(new_images_path, exist_ok=True)

                    # Process every label file in the source labels directory
                    for original_label_file_path in glob.glob(f"{old_labels_path}/*.txt"):
                        label_file = LabelFile(original_label_file_path)

                        # Copy only labels that belong to selected classes, remapping indices, renaming file if name collison occurs
                        new_label_file_path = label_file.copy_by_class_indeces(old_to_new_index_match, new_labels_path)
                        if new_label_file_path:
                            # Find images belonging to copied label file and copy them with renaming if neccessary
                            file_name = os.path.basename(original_label_file_path)
                            file_name_no_ext = file_name[:file_name.rfind('.')]
                            corresponding_image_files = glob.glob(f"{old_images_path}/{file_name_no_ext}.*")
                            for img in corresponding_image_files:
                                try:
                                    _ , ext = os.path.splitext(img)
                                except:
                                    print(f"ERROR: Not a valid image, extension missing for: {img}")
                                    return False

                                # Validate image format before copying
                                if not ignore_img_formats and ext.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                                    print(f"ERROR: Image format not supported by yolo: {img}\nSupported formats are: {self.SUPPORTED_IMAGE_FORMATS}")
                                    return False

                                # Preserve the label filename and copy the image alongside the new label
                                new_image_name,_ = os.path.splitext(os.path.basename(new_label_file_path))
                                new_image_name = new_image_name + ext
                                destination = "/".join([new_images_path, new_image_name])
                                if not shutil.copyfile(img, destination):
                                    print(f"ERROR: Image could not be copied:\nSource: {img}\nDestination: {destination}")
                                    return False

        # Write the new dataset YAML once file copying is complete
        data_file_writer = YoloDataFileWriter(new_dataset_path, list(new_class_indeces.keys()))
        try:
            data_file_writer.write()
        except Exception as e:
            print(f"ERROR: Could not write new data.yaml to {new_dataset_path}. Generation of {data_set_name} is incomplete!")
            print(f"Exception was: {e}")
            return False


        return True

    def __generate_new_class_indeces(self, class_names:set[str])-> dict[str, str]:
        """Generate a fresh label index mapping for the selected class names."""
        ret: dict[str, str] = {}
        for i, name in enumerate(class_names):
            ret[name] = str(i)
        return ret

    def __generate_label_path_from_img_path(self, img_path: str) -> str:
        """Convert an images directory path into its corresponding labels directory path."""
        img_path = "".join([img_path, "/"])
        return img_path.replace("/images/", "/labels/")
    







        
    

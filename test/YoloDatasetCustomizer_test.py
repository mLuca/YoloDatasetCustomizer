import os
import sys
import tempfile
import shutil
import unittest
import yaml
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from YoloDatasetCustomizer import (
    YoloDataFileWriter,
    YoloDataFileReader,
    LabelFile,
    YoloDatasetCustomizer,
)


class TestYoloDatasetCustomizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace_root = tempfile.mkdtemp(prefix="yolodatasetcustomizer_test_")

        # Good dataset used for positive tests
        cls.good_dataset_dir = os.path.join(cls.workspace_root, "good_dataset")
        os.makedirs(cls.good_dataset_dir, exist_ok=True)
        writer = YoloDataFileWriter(
            data_set_dir=cls.good_dataset_dir,
            class_names=["person", "car"],
            train_split="train",
            val_split="valid",
            test_split="test",
        )
        writer.write()
        for split_name in ("train", "valid", "test"):
            images_dir = os.path.join(cls.good_dataset_dir, split_name, "images")
            labels_dir = os.path.join(cls.good_dataset_dir, split_name, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

        cls._create_dummy_image_file(os.path.join(cls.good_dataset_dir, "train", "images", "person_car.jpg"))
        cls._write_text_file(
            os.path.join(cls.good_dataset_dir, "train", "labels", "person_car.txt"),
            "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n",
        )
        cls._create_dummy_image_file(os.path.join(cls.good_dataset_dir, "valid", "images", "car.png"))
        cls._write_text_file(
            os.path.join(cls.good_dataset_dir, "valid", "labels", "car.txt"),
            "1 0.2 0.3 0.4 0.5\n",
        )
        cls._create_dummy_image_file(os.path.join(cls.good_dataset_dir, "test", "images", "person.jpg"))
        cls._write_text_file(
            os.path.join(cls.good_dataset_dir, "test", "labels", "person.txt"),
            "0 0.4 0.5 0.6 0.7\n",
        )

        # Dataset with invalid YAML entry type for names
        cls.invalid_yaml_dataset_dir = os.path.join(cls.workspace_root, "invalid_yaml_dataset")
        os.makedirs(cls.invalid_yaml_dataset_dir, exist_ok=True)
        invalid_yaml_content = """
train: train/images
val: valid/images
test: test/images
names: !!set {person: null}
nc: 1
"""
        cls._write_text_file(
            os.path.join(cls.invalid_yaml_dataset_dir, "data.yaml"),
            invalid_yaml_content,
        )
        cls._create_dummy_image_file(os.path.join(cls.invalid_yaml_dataset_dir, "train", "images", "person_car.jpg"))
        os.makedirs(os.path.join(cls.invalid_yaml_dataset_dir, "train", "labels"), exist_ok=True)
        cls._write_text_file(
            os.path.join(cls.invalid_yaml_dataset_dir, "train", "labels", "person_car.txt"),
            "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n",
        )

        # Dataset with unsupported image format for the customizer
        cls.unsupported_image_dataset_dir = os.path.join(cls.workspace_root, "unsupported_image_dataset")
        os.makedirs(cls.unsupported_image_dataset_dir, exist_ok=True)
        writer = YoloDataFileWriter(
            data_set_dir=cls.unsupported_image_dataset_dir,
            class_names=["person", "car"],
            train_split="train",
            val_split="valid",
            test_split="test",
        )
        writer.write()
        os.makedirs(os.path.join(cls.unsupported_image_dataset_dir, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(cls.unsupported_image_dataset_dir, "train", "labels"), exist_ok=True)
        cls._create_dummy_image_file(os.path.join(cls.unsupported_image_dataset_dir, "train", "images", "person_car.gif"))
        cls._write_text_file(
            os.path.join(cls.unsupported_image_dataset_dir, "train", "labels", "person_car.txt"),
            "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8\n",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace_root, ignore_errors=True)

    @staticmethod
    def _write_text_file(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _create_dummy_image_file(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        extension = os.path.splitext(path)[1].lower()
        if extension == ".png":
            header = b"\x89PNG\r\n\x1a\n"
        elif extension in (".jpg", ".jpeg"):
            header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
        elif extension == ".gif":
            header = b"GIF89a"
        else:
            header = b"\x00"

        with open(path, "wb") as f:
            f.write(header)

    def test_yolo_data_file_writer_writes_yaml(self):
        yaml_path = os.path.join(self.good_dataset_dir, "data.yaml")
        self.assertTrue(os.path.exists(yaml_path))
        with open(yaml_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded["train"], "train/images")
        self.assertEqual(loaded["val"], "valid/images")
        self.assertEqual(loaded["test"], "test/images")
        self.assertEqual(loaded["names"], {0: "person", 1: "car"})
        self.assertEqual(loaded["nc"], 2)

    def test_yolo_data_file_writer_uses_independent_default_class_names(self):
        writer_a = YoloDataFileWriter()
        writer_b = YoloDataFileWriter()

        writer_a.class_names.append("person")

        self.assertEqual(writer_a.class_names, ["person"])
        self.assertEqual(writer_b.class_names, [])

    def test_yolo_data_file_reader_reads_splits_names_and_indices(self):
        reader = YoloDataFileReader(os.path.join(self.good_dataset_dir, "data.yaml"))
        self.assertEqual(reader.get_file_path(), os.path.abspath(os.path.join(self.good_dataset_dir, "data.yaml")))
        self.assertEqual(reader.get_split_paths_for_split_name("train"), ["train/images"])
        self.assertEqual(reader.get_split_paths_for_split_name("val"), ["valid/images"])
        self.assertEqual(reader.get_class_names(), ["person", "car"])
        self.assertEqual(reader.get_indices_for_names({"person"}), {"0"})
        self.assertEqual(reader.get_indices_for_names({"unknown"}), set())

    def test_yolo_data_file_reader_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            YoloDataFileReader(os.path.join(self.workspace_root, "does_not_exist.yaml"))

    def test_yolo_data_file_reader_raises_type_error_for_invalid_names_entry(self):
        with self.assertRaises(TypeError):
            YoloDataFileReader(os.path.join(self.invalid_yaml_dataset_dir, "data.yaml"))

    def test_label_file_parses_class_indices_and_copy_by_class_indices(self):
        label_path = os.path.join(self.good_dataset_dir, "train", "labels", "person_car.txt")
        label_file = LabelFile(label_path)
        self.assertEqual(label_file.get_class_indices(), {"0", "1"})

        output_labels_dir = os.path.join(self.workspace_root, "copied_labels")
        os.makedirs(os.path.join(output_labels_dir, "labels"), exist_ok=True)
        copied_path = label_file.copy_by_class_indeces({"1": "0"}, output_labels_dir)
        self.assertIsNotNone(copied_path)
        self.assertTrue(os.path.exists(copied_path))
        with open(copied_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("0 0.5 0.6 0.7 0.8", content)
        self.assertNotIn("1 0.5 0.6 0.7 0.8", content)
        self.assertNotIn("0.1 0.2 0.3 0.4", content)

    def test_label_file_copy_by_class_indeces_returns_none_when_missing_indices(self):
        label_path = os.path.join(self.good_dataset_dir, "train", "labels", "person_car.txt")
        label_file = LabelFile(label_path)
        self.assertIsNone(label_file.copy_by_class_indeces({"2": "0"}, os.path.join(self.workspace_root, "labels")))

    def test_label_file_copy_with_missing_destination_directory_raises(self):
        label_path = os.path.join(self.good_dataset_dir, "train", "labels", "person_car.txt")
        label_file = LabelFile(label_path)
        missing_destination = os.path.join(self.workspace_root, "missing", "labels")
        with self.assertRaises(FileNotFoundError):
            label_file.copy_by_class_indeces({"0": "0"}, missing_destination)

    def test_yolo_dataset_customizer_discovers_all_data_yaml_files(self):
        customizer = YoloDatasetCustomizer([self.workspace_root])
        found_paths = customizer.get_found_data_file_paths()
        self.assertGreaterEqual(len(found_paths), 3)
        self.assertIn(os.path.abspath(os.path.join(self.good_dataset_dir, "data.yaml")), found_paths)
        self.assertIn(os.path.abspath(os.path.join(self.invalid_yaml_dataset_dir, "data.yaml")), found_paths)
        self.assertIn(os.path.abspath(os.path.join(self.unsupported_image_dataset_dir, "data.yaml")), found_paths)
        self.assertEqual(customizer.get_found_class_names(), {"person", "car"})

    def test_yolo_dataset_customizer_handles_windows_style_paths(self):
        windows_style_root = self.workspace_root.replace(os.sep, "\\")
        customizer = YoloDatasetCustomizer([windows_style_root])
        self.assertIn(os.path.abspath(os.path.join(self.good_dataset_dir, "data.yaml")), customizer.get_found_data_file_paths())
        self.assertEqual(customizer.get_found_class_names(), {"person", "car"})

    def test_create_new_dataset_returns_false_when_image_copy_fails(self):
        customizer = YoloDatasetCustomizer([self.good_dataset_dir])
        with patch("YoloDatasetCustomizer.customizer.shutil.copyfile", side_effect=OSError("disk full")):
            result = customizer.create_new_dataset_for_class_names(
                {"person", "car"}, dst_path=self.workspace_root, data_set_name="copy_failure_dataset"
            )
        self.assertFalse(result)

    def test_yolo_dataset_customizer_creates_filtered_dataset(self):
        customizer = YoloDatasetCustomizer([self.good_dataset_dir])
        success = customizer.create_new_dataset_for_class_names({"car"}, dst_path=self.workspace_root, data_set_name="filtered_car_dataset")
        self.assertTrue(success)

        new_dataset_path = os.path.join(self.workspace_root, "filtered_car_dataset")
        self.assertTrue(os.path.isdir(new_dataset_path))
        new_yaml_path = os.path.join(new_dataset_path, "data.yaml")
        self.assertTrue(os.path.exists(new_yaml_path))
        with open(new_yaml_path, "r", encoding="utf-8") as f:
            new_data = yaml.safe_load(f)
        self.assertEqual(new_data["names"], {0: "car"})

        new_label_path = os.path.join(new_dataset_path, "train", "labels", "person_car.txt")
        self.assertTrue(os.path.exists(new_label_path))
        with open(new_label_path, "r", encoding="utf-8") as f:
            copied_label = f.read()
        self.assertIn("0 0.5 0.6 0.7 0.8", copied_label)
        self.assertNotIn("1 0.5 0.6 0.7 0.8", copied_label)
        self.assertNotIn("0.1 0.2 0.3 0.4", copied_label)

        new_image_path = os.path.join(new_dataset_path, "train", "images", "person_car.jpg")
        self.assertTrue(os.path.exists(new_image_path))

    def test_yolo_dataset_customizer_returns_false_for_empty_class_names(self):
        customizer = YoloDatasetCustomizer([self.good_dataset_dir])
        self.assertFalse(customizer.create_new_dataset_for_class_names(set(), dst_path=self.workspace_root, data_set_name="empty_selection"))

    def test_yolo_dataset_customizer_returns_false_for_unsupported_image_format(self):
        customizer = YoloDatasetCustomizer([self.unsupported_image_dataset_dir])
        destination_name = "unsupported_image_dataset_filtered"
        success = customizer.create_new_dataset_for_class_names({"person"}, dst_path=self.workspace_root, data_set_name=destination_name)
        self.assertFalse(success)
        self.assertFalse(os.path.exists(os.path.join(self.workspace_root, destination_name, "data.yaml")))


if __name__ == "__main__":
    unittest.main(verbosity=2)

from src.YoloDatasetCustomizer import YoloDatasetCustomizer

def main():
    dfr_1 = YoloDatasetCustomizer(["./test/data/GermanTrashbin_yolov11/data.yaml", "./test/data/"])
    #print(f"{dfr_1.get_found_data_file_paths()}\n")
    print(f"Found classes: {dfr_1.get_found_class_names()}")


if __name__ == "__main__":
    main()

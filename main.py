from src.YoloDatasetCustomizer import YoloDatasetCustomizer

def main():
    dfr_1 = YoloDatasetCustomizer(["./test/data/Dog+Person/data.yaml", "./test/data/Person+Car"])
    #print(f"{dfr_1.get_found_data_file_paths()}\n")
    print(f"Found classes: {dfr_1.get_found_class_names()}")
    dfr_1.create_new_dataset_for_class_names({"Person"})


if __name__ == "__main__":
    main()

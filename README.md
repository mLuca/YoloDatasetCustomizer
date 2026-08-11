# Intro

This tool is meant to help create a custom dataset by picking and choosing class names from pre-existing datasets.

# Limitations

- The dataset YAML file is expected to be called "data.yaml"
- The 'path' variable in data.yaml is not processed.
  It is assumed that the dataset root path is the directory where data.yaml resides and all data is present already.
- The 'script' variable in data.yaml is not processed. Make sure all data is present.

# Usage

    form YoloDatasetCustomizer import YoloDatasetCustomizer

    ydc = YoloDatasetCustomizer(list_to_existing_datasets)
    ydc.add([one_more_existing_dataset])

    if ydc.create_new_dataset_for_class_names(set_of_wanted_class_names, optional_custom_path, optional_custom_name):
        print("All good")
    else:
        print("Oh no.")

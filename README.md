# Intro

This tool is meant to help create a custom dataset by picking and choosing class names from pre-existing datasets.

# Limitations

- The dataset YAML file is expected to be called _data.yaml_
- The 'path' variable in data.yaml is not processed.
  It is assumed that the dataset root path is the directory where _data.yaml_ resides and all data is present already.
- The _script_ variable in _data.yaml_ is not processed. Make sure all data is present.

# Usage

If a path ends on _data.yaml_ it will take that file into account.
If a path doesn't end on _/data.yaml_ it will recursevly look for all _data.yaml_ files from that directory downards.
e.g.

    from YoloDatasetCustomizer import YoloDatasetCustomizer

    ydc = YoloDatasetCustomizer(list_to_existing_datasets)
    ydc.add([one_more_existing_dataset])

    if ydc.create_new_dataset_for_class_names(set_of_wanted_class_names, optional_custom_path, optional_custom_name):
        print("All good")
    else:
        print("Oh no.")

# Run tests

The project uses uv. Install dependencies and run tests with:

    uv run pytest

# Lint & type-check

    uv run ruff check src test
    uv run mypy

# Build project

Build the package

    uv build

Publish to PyPI (still need twine)

    uv pip install twine
    uv run twine upload dist/\*

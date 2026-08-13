import os

_SPLIT_NAMES = ["train", "val", "test"]


def _normalize_path(path: str) -> str:
    """Normalize path separators and return an absolute path for cross-platform use."""
    return os.path.abspath(str(path).replace("\\", os.sep))


def _get_unique_path(path: str) -> str:
    """Return an unused file path by appending a numeric suffix when needed."""
    path = _normalize_path(path)
    base, ext = os.path.splitext(path)
    unique_path = path
    index = 1
    while os.path.exists(unique_path):
        unique_path = f"{base}_{index}{ext}"
        index += 1
    return unique_path

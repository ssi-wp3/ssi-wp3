import os


def get_test_path(filename: str) -> str:
    return os.path.join(os.getcwd(), filename)
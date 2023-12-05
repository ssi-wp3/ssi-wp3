import os


def get_test_path(filename: str = "") -> str:
    test_directory = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(test_directory, filename)

def remove_test_files():
    test_directory = get_test_path()
    for filename in os.listdir(test_directory):
        os.remove(os.path.join(test_directory, filename))
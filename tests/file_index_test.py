from ssi.file_index import FileIndex
from test_utils import get_test_path
from pathlib import Path
import unittest
import shutil
import os


class FileIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_path = get_test_path("recursive_directory")
        self.file_index = FileIndex(self.test_path, ".txt")

        self.files = [os.path.join(self.test_path, "test1.txt"),
                      os.path.join(self.test_path, "test2.csv"),
                      os.path.join(self.test_path, "dir1/test3.txt"),
                      os.path.join(self.test_path, "dir1/test4.txt"),
                      os.path.join(self.test_path, "dir1/subdir1/test5.txt"),
                      os.path.join(self.test_path, "dir1/subdir1/test6.txt"),
                      os.path.join(self.test_path, "dir1/subdir2/test7.txt"),
                      os.path.join(self.test_path, "dir2/test8.txt"),
                      ]

        for file in self.files:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "w") as f:
                f.write("Test")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_path)

    def test_root_directory(self):
        print(self.file_index.root_directory.name)
        print(self.test_path)
        self.assertEqual(self.file_index.root_directory, Path(self.test_path))

    def test_file_extension(self):
        self.assertEqual(self.file_index.file_extension, ".txt")

    def test_files(self):
        self.assertEqual({os.path.splitext(os.path.basename(filename))[0]
                         for filename in self.files if filename.endswith(".txt")},
                         set(self.file_index.files.keys()))
        self.assertEqual({Path(filename) for filename in self.files if filename.endswith(".txt")},
                         set(self.file_index.files.values()))

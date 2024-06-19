from ssi.file_index import FileIndex
from test_utils import get_test_path
import unittest
import shutil


class FileIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_path = get_test_path("recursive_directory")
        self.file_index = FileIndex(self.test_path, ".txt")

        self.files = ["recursive_directory/test1.txt",
                      "recursive_directory1/test2.csv",
                      "recursive_directory2/dir1/test3.txt",
                      "recursive_directory2/dir1/test4.txt",
                      "recursive_directory2/dir1/subdir1/test5.txt",
                      "recursive_directory2/dir1/subdir1/test6.txt",
                      "recursive_directory2/dir1/subdir2/test7.txt",
                      "recursive_directory2/dir2/test8.txt",
                      ]

    def tearDown(self) -> None:
        shutil.rmtree(self.test_path)

    def test_root_directory(self):
        self.assertEqual(self.file_index.root_directory, self.test_path)

    def test_file_extension(self):
        self.assertEqual(self.file_index.file_extension, ".txt")

    def test_files(self):
        self.assertEqual(self.file_index.files, [
                         filename for filename in self.files if filename.endswith(".txt")])

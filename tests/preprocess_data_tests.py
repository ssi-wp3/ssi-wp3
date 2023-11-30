from ssi.preprocess_data import *
import unittest


class TestPreprocessData(unittest.TestCase):
    def test_split_coicop_returns_full_coicop_number(self):
        self.assertEqual("011201", split_coicop("011201"))
        self.assertEqual("022312", split_coicop("022312"))
        self.assertEqual("123423", split_coicop("123423"))
        self.assertEqual("054534", split_coicop("054534"))
        self.assertEqual("065645", split_coicop("065645"))

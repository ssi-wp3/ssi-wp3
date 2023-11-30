from ssi.preprocess_data import *
import unittest
import pandas as pd


class TestPreprocessData(unittest.TestCase):
    def test_split_coicop_returns_full_coicop_number(self):
        coicop_series = pd.Series([
            "011201", "022312", "123423", "054534", "065645"
        ])

        self.assertTrue(coicop_series.equals(split_coicop(
            coicop_series).coicop_number))

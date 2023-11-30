from ssi.preprocess_data import *
import unittest
import pandas as pd


class TestPreprocessData(unittest.TestCase):
    def setUp(self) -> None:
        self.coicop_series = pd.Series([
            "011201", "022312", "123423", "054534", "065645"
        ])

    def test_split_coicop_returns_full_coicop_number(self):
        self.assertTrue(self.coicop_series.equals(split_coicop(
            self.coicop_series).coicop_number))

    def test_split_coicop_returns_coicop_divisions(self):
        coicop_division = pd.Series([
            "01", "02", "12", "05", "06"
        ])

        self.assertTrue(coicop_division.equals(split_coicop(
            self.coicop_series).coicop_division))

    def test_split_coicop_returns_coicop_groups(self):
        coicop_group = pd.Series([
            "011", "022", "123", "054", "065"
        ])

        self.assertTrue(coicop_group.equals(split_coicop(
            self.coicop_series).coicop_group))

    def test_split_coicop_returns_coicop_classes(self):
        coicop_class = pd.Series([
            "0112", "0223", "1234", "0545", "0656"
        ])

        self.assertTrue(coicop_class.equals(split_coicop(
            self.coicop_series).coicop_class))

    def test_split_coicop_returns_coicop_subclasses(self):
        coicop_subclass = pd.Series([
            "01120", "02231", "12342", "05453", "06564"
        ])
        self.assertTrue(coicop_subclass.equals(split_coicop(
            self.coicop_series).coicop_subclass))

    def test_split_coicop_returns_coicop_subsubclasses(self):
        self.assertTrue(self.coicop_series.equals(split_coicop(
            self.coicop_series).coicop_subsubclass))

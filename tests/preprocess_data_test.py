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

    def test_add_leading_zero(self):
        dataframe = pd.DataFrame(
            {"coicop_number": ["1234", "12345", "123456"]})
        dataframe = add_leading_zero(dataframe)
        self.assertTrue(dataframe["coicop_number"].equals(
            pd.Series(["1234", "012345", "123456"])))

    def test_add_unique_product_id(self):
        dataframe = pd.DataFrame(
            {"ean_name": ["product1", "product2", "product3"]})
        dataframe = add_unique_product_id(dataframe)
        self.assertTrue(dataframe["product_id"].equals(
            pd.Series([hash("product1"), hash("product2"), hash("product3")])))

    def test_add_coicop_levels(self):
        dataframe = pd.DataFrame(
            {"coicop_number": ["011201", "022312", "123423", "054534", "065645"]})
        dataframe = add_coicop_levels(dataframe)
        self.assertTrue(dataframe["coicop_division"].equals(
            pd.Series(["01", "02", "12", "05", "06"])))
        self.assertTrue(dataframe["coicop_group"].equals(
            pd.Series(["011", "022", "123", "054", "065"])))
        self.assertTrue(dataframe["coicop_class"].equals(
            pd.Series(["0112", "0223", "1234", "0545", "0656"])))
        self.assertTrue(dataframe["coicop_subclass"].equals(
            pd.Series(["01120", "02231", "12342", "05453", "06564"])))

    def test_get_category_counts(self):
        dataframe = pd.DataFrame({
            "coicop_number": ["011201", "011201", "022312", "022312", "022312",
                              "123423", "054534", "054534", "054534", "054534",
                              "065645", "065645", "065645", "065645", "065645"],
            "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

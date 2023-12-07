from ssi.synthetic_data import generate_fake_data_with_coicop_levels
from ssi.data_exploration import filter_coicop_level, get_product_counts_per_year, get_product_counts_per_category_and_time
import pandas as pd
import unittest


class DataExplorationTest(unittest.TestCase):
    def test_filter_on_coicop_level(self):
        dataframe = generate_fake_data_with_coicop_levels(100, 2018, 2021)
        for coicop_level in ["coicop_division", "coicop_group", "coicop_class", "coicop_subclass"]:
            for coicop_value in dataframe[coicop_level].unique():
                filtered_dataframe = filter_coicop_level(
                    dataframe, coicop_level, coicop_value)
                self.assertEqual(
                    len(filtered_dataframe), len(dataframe[dataframe[coicop_level] == coicop_value]))
                self.assertEqual(
                    len(filtered_dataframe), len(filtered_dataframe[filtered_dataframe[coicop_level] == coicop_value]))
                self.assertTrue(dataframe[dataframe[coicop_level] == coicop_value].equals(
                    filtered_dataframe[filtered_dataframe[coicop_level] == coicop_value]))

    def test_get_product_counts_per_year(self):
        dataframe = pd.DataFrame({
            "year_month": ["201801", "201801", "201901", "201902", "201903", "202001", "202002", "202003", "202004", "202004"],
            "product_id": [1, 2, 1, 2, 2, 3, 1, 2, 3, 1]
        })

        expected_dataframe = pd.DataFrame({
            "count": [2, 2, 3]
        }, index=["2018", "2019", "2020"])

        counts_per_year_df = get_product_counts_per_year(
            dataframe, "year_month", "product_id")

        self.assertTrue(expected_dataframe.equals(counts_per_year_df))

    def test_get_product_counts_per_category_per_year(self):
        dataframe = pd.DataFrame({
            "year_month": ["201801", "201801", "201901", "201902", "201903", "202001", "202002", "202003", "202004", "202004"],
            "product_id": [1, 2, 1, 2, 2, 3, 5, 6, 7, 5],
            "coicop_division": ["01", "01", "01", "01", "02", "02", "02", "03", "03", "02"]
        })

        expected_dataframe = pd.DataFrame({
            "count": [2, 2, 1, 2, 2]
        },
            index=pd.MultiIndex.from_tuples(tuples=[("2018", "01"), ("2019", "01"), ("2019", "02"), (
                "2020", "02"), ("2020", "03")], names=["year", "coicop_division"])
        )

        counts_per_year_df = get_product_counts_per_category_and_time(
            dataframe, "year", "year_month", "product_id", "coicop_division")

        self.assertTrue(expected_dataframe.equals(counts_per_year_df))

from ssi.synthetic_data import generate_fake_data_with_coicop_levels
from ssi.data_exploration import filter_coicop_level
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


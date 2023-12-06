from ssi.files import *
from test_utils import get_test_path, remove_test_files
import unittest
import os


class FilesTest(unittest.TestCase):
    def setUp(self) -> None:
        remove_test_files()

    def test_get_revenue_files_in_folder(self):
        data_directory = os.path.join(os.getcwd(), "tests", "data")
        os.makedirs(data_directory, exist_ok=True)

        revenue_files = {
            "AH": 2,
            "Jumbo": 4,
            "Lidl": 1,
            "Plus": 3
        }

        for supermarket_name, number_of_files in revenue_files.items():
            for i in range(number_of_files):
                filename = f"Omzet_{supermarket_name}_{i}.parquet"
                with open(os.path.join(data_directory, filename), "w") as f:
                    f.write("")

        revenue_files_ah = get_revenue_files_in_folder(data_directory, "AH")
        self.assertTrue(len(revenue_files_ah) == 2)
        self.assertTrue(all([os.path.isfile(revenue_file)
                        for revenue_file in revenue_files_ah]))
        self.assertEqual(set([
            os.path.join(data_directory, "Omzet_AH_0.parquet"),
            os.path.join(data_directory, "Omzet_AH_1.parquet")
        ]), set(revenue_files_ah))

        revenue_files_jumbo = get_revenue_files_in_folder(
            data_directory, "Jumbo")
        self.assertTrue(len(revenue_files_jumbo) == 4)
        self.assertTrue(all([os.path.isfile(revenue_file)
                        for revenue_file in revenue_files_jumbo]))
        self.assertEqual(set([
            os.path.join(data_directory, "Omzet_Jumbo_0.parquet"),
            os.path.join(data_directory, "Omzet_Jumbo_1.parquet"),
            os.path.join(data_directory, "Omzet_Jumbo_2.parquet"),
            os.path.join(data_directory, "Omzet_Jumbo_3.parquet")
        ]), set(revenue_files_jumbo))

        revenue_files_lidl = get_revenue_files_in_folder(
            data_directory, "Lidl")
        self.assertTrue(len(revenue_files_lidl) == 1)
        self.assertTrue(all([os.path.isfile(revenue_file)
                        for revenue_file in revenue_files_lidl]))
        self.assertEqual(os.path.join(
            data_directory, "Omzet_Lidl_0.parquet"), revenue_files_lidl[0])

        revenue_files_plus = get_revenue_files_in_folder(
            data_directory, "Plus")
        self.assertTrue(len(revenue_files_plus) == 3)
        self.assertTrue(all([os.path.isfile(revenue_file)
                        for revenue_file in revenue_files_plus]))
        self.assertEqual(set([
            os.path.join(data_directory, "Omzet_Plus_0.parquet"),
            os.path.join(data_directory, "Omzet_Plus_1.parquet"),
            os.path.join(data_directory, "Omzet_Plus_2.parquet")
        ]), set(revenue_files_plus))

        for supermarket_name, number_of_files in revenue_files.items():
            for i in range(number_of_files):
                filename = f"Omzet_{supermarket_name}_{i}.parquet"
                os.remove(os.path.join(data_directory, filename))

    def test_get_feature_filename(self):
        self.assertEqual("ssi_supermarket1_count_vectorizer_features.parquet",
                         get_feature_filename("count_vectorizer", "supermarket1"))
        self.assertEqual("ssi_supermarket2_count_vectorizer_features.parquet",
                         get_feature_filename("count_vectorizer", "supermarket2"))
        self.assertEqual("ssi_supermarket2_tfidf_word_features.parquet",
                         get_feature_filename("tfidf_word", "supermarket2"))
        self.assertEqual("ssi_supermarket3_tfidf_char_features.parquet",
                         get_feature_filename("tfidf_char", "supermarket3"))
        self.assertEqual("ssi_supermarket1_tfidf_char34_features.parquet",
                         get_feature_filename("tfidf_char34", "supermarket1"))
        self.assertEqual("ssi_supermarket2_count_char_features.parquet",
                         get_feature_filename("count_char", "supermarket2"))
        self.assertEqual("ssi_supermarket3_spacy_nl_sm_features.parquet",
                         get_feature_filename("spacy_nl_sm", "supermarket3"))
        self.assertEqual("ssi_supermarket1_spacy_nl_md_features.parquet",
                         get_feature_filename("spacy_nl_md", "supermarket1"))

    def test_get_features_files_in_directory(self):
        expected_feature_filenames = [get_feature_filename(f"feature_extractor_{i}", supermarket_name)
                                      for i in range(10)
                                      for supermarket_name in ["AH", "Jumbo", "Lidl", "Plus"]
                                      ]
        for feature_filename in expected_feature_filenames:
            with open(get_test_path(feature_filename), "w") as file:
                file.write("test")

        feature_filenames = get_features_files_in_directory(get_test_path(""))
        self.assertEqual(len(expected_feature_filenames),
                         len(feature_filenames))
        self.assertEqual(set(expected_feature_filenames),
                         set(feature_filenames))

    def test_get_combined_revenue_filename(self):
        self.assertEqual("ssi_ah_revenue.parquet",
                         get_combined_revenue_filename("AH"))
        self.assertEqual("ssi_jumbo_revenue.parquet",
                         get_combined_revenue_filename("Jumbo"))
        self.assertEqual("ssi_lidl_revenue.parquet",
                         get_combined_revenue_filename("Lidl"))
        self.assertEqual("ssi_plus_revenue.parquet",
                         get_combined_revenue_filename("Plus"))

    def test_get_supermarket_name_from_standardized_filename(self):
        self.assertEqual("ah", get_supermarket_name("ssi_ah_revenue.parquet"))
        self.assertEqual("jumbo", get_supermarket_name(
            "path/to/ssi_jumbo_revenue.parquet"))
        self.assertEqual("lidl", get_supermarket_name(
            "other_path_to/ssi_lidl_revenue.parquet"))
        self.assertEqual("plus", get_supermarket_name(
            "ssi_plus_revenue.parquet"))
        self.assertEqual("ah", get_supermarket_name(
            "fake_data/ssi_ah_count_vectorizer_features.parquet"))
        self.assertEqual("jumbo", get_supermarket_name(
            "ssi_jumbo_count_vectorizer_features.parquet"))
        self.assertEqual("lidl", get_supermarket_name(
            "ssi_lidl_count_vectorizer_features.parquet"))
        self.assertEqual("plus", get_supermarket_name(
            "ssi_plus_count_vectorizer_features.parquet"))

    def test_get_combined_revenue_files_in_directory(self):
        data_directory = get_test_path("")
        os.makedirs(data_directory, exist_ok=True)

        for supermarket_name in ["AH", "Jumbo", "Lidl", "Plus"]:
            with open(os.path.join(data_directory, get_combined_revenue_filename(supermarket_name)), "w") as file:
                file.write("test")

        expected_filenames = [os.path.join(data_directory, get_combined_revenue_filename(supermarket_name))
                              for supermarket_name in ["AH", "Jumbo", "Lidl", "Plus"]]
        combined_revenue_files = get_combined_revenue_files_in_directory(
            data_directory)
        self.assertEqual(4, len(combined_revenue_files))
        self.assertEqual(set(expected_filenames), set(combined_revenue_files))

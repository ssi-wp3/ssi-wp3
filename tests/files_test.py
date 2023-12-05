from ssi.files import get_revenue_files_in_folder, get_feature_filename, get_features_files_in_directory, get_combined_revenue_filename, get_combined_revenue_files_in_directory
from test_utils import get_test_path
import unittest
import os

class FilesTest(unittest.TestCase):
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
            self.assertEqual("ssi_features_count_vectorizer.parquet", get_feature_filename("count_vectorizer"))
            self.assertEqual("ssi_features_tfidf_word.parquet", get_feature_filename("tfidf_word"))
            self.assertEqual("ssi_features_tfidf_char.parquet", get_feature_filename("tfidf_char"))
            self.assertEqual("ssi_features_tfidf_char34.parquet", get_feature_filename("tfidf_char34"))
            self.assertEqual("ssi_features_count_char.parquet", get_feature_filename("count_char"))
            self.assertEqual("ssi_features_spacy_nl_sm.parquet", get_feature_filename("spacy_nl_sm"))
            self.assertEqual("ssi_features_spacy_nl_md.parquet", get_feature_filename("spacy_nl_md"))

    def test_get_features_files_in_directory(self):
        expected_feature_filenames = [get_feature_filename(f"feature_extractor_{i}") for i in range(10)]
        for feature_filename in expected_feature_filenames:
            with open(get_test_path(feature_filename), "w") as file:
                file.write("test")
        
        feature_filenames = get_features_files_in_directory(get_test_path(""))
        self.assertEqual(len(expected_feature_filenames), len(feature_filenames))
        self.assertEqual(set(expected_feature_filenames), set(feature_filenames))

    def test_get_combined_revenue_filename(self):
        self.assertEqual("ssi_ah_revenue.parquet", get_combined_revenue_filename("AH"))
        self.assertEqual("ssi_jumbo_revenue.parquet", get_combined_revenue_filename("Jumbo"))
        self.assertEqual("ssi_lidl_revenue.parquet", get_combined_revenue_filename("Lidl"))
        self.assertEqual("ssi_plus_revenue.parquet", get_combined_revenue_filename("Plus"))

    def test_get_combined_revenue_files_in_directory(self):
        data_directory = get_test_path("")
        os.makedirs(data_directory, exist_ok=True)

        for supermarket_name in ["AH", "Jumbo", "Lidl", "Plus"]:
            with open(os.path.join(data_directory, get_combined_revenue_filename(supermarket_name)), "w") as file:
                file.write("test")

        expected_filenames = [os.path.join(data_directory, get_combined_revenue_filename(supermarket_name)) 
                              for supermarket_name in ["AH", "Jumbo", "Lidl", "Plus"]]
        combined_revenue_files = get_combined_revenue_files_in_directory(data_directory)
        self.assertEqual(4, len(combined_revenue_files))
        self.assertEqual(set(expected_filenames), set(combined_revenue_files))
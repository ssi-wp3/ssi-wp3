from ssi.preprocess_data import *
import unittest
import pandas as pd
import os


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
            "product_id": [1, 1, 2, 3, 4,
                           5, 6, 6, 8, 9,
                           10, 11, 11, 12, 13]})

        category_counts = get_category_counts(
            dataframe, coicop_column="coicop_number", product_id_column="product_id")
        self.assertTrue(category_counts.sort_values(by="coicop_number")["count"].equals(
            pd.Series([1, 3, 3, 4, 1])))

    def test_filter_columns(self):
        dataframe = pd.DataFrame({
            "coicop_number": ["011201" for _ in range(10)],
            "product_id": [i for i in range(10)],
            "product_name": [f"product_{i}" for i in range(10)],
            "isba_number": [i for i in range(10)],
            "isba_name": [f"isba_{i}" for i in range(10)],
            "ean_number": [i for i in range(10)],
            "ean_name": [f"ean_{i}" for i in range(10)]
        })
        filtered_dataframe1 = filter_columns(
            dataframe, columns=["coicop_number"])
        self.assertTrue(["coicop_number"] ==
                        filtered_dataframe1.columns.tolist())

        filtered_dataframe2 = filter_columns(
            dataframe, columns=["coicop_number", "product_id"])
        self.assertTrue(["coicop_number", "product_id"]
                        == filtered_dataframe2.columns.tolist())

        filtered_dataframe3 = filter_columns(
            dataframe, columns=["coicop_number", "product_id", "product_name"])
        self.assertTrue(["coicop_number", "product_id",
                        "product_name"] == filtered_dataframe3.columns.tolist())

    def test_preprocess_data(self):
        dataframe = pd.DataFrame({
            "coicop_number": ["11201", "11201", "22312", "22312", "022312",
                              "123423", "54534", "054534", "54534", "54534",
                              "65645", "065645", "65645", "65645", "065645"],
            "product_name": [f"product_{i}" for i in range(15)],
            "isba_number": [i for i in range(15)],
            "isba_name": [f"isba_{i}" for i in range(15)],
            "ean_number": [i for i in range(15)],
            "ean_name": [f"ean_{i}" for i in range(15)]
        })

        processed_dataframe = preprocess_data(
            dataframe, columns=["coicop_number", "ean_number", "ean_name"], )
        self.assertEqual(len(dataframe), len(processed_dataframe))
        self.assertEqual([6] * len(processed_dataframe),
                         processed_dataframe["coicop_number"].str.len().tolist())
        self.assertEqual(["coicop_number", "ean_number", "ean_name",
                         "product_id", "coicop_division", "coicop_group",
                          "coicop_class", "coicop_subclass"],
                         processed_dataframe.columns.tolist())
        self.assertEqual(["011201", "011201", "022312", "022312", "022312",
                          "123423", "054534", "054534", "054534", "054534",
                          "065645", "065645", "065645", "065645", "065645"],
                         processed_dataframe["coicop_number"].tolist())
        self.assertEqual([hash(f"ean_{i}") for i in range(
            15)], processed_dataframe["product_id"].tolist())
        self.assertEqual(["01", "01", "02", "02", "02",
                          "12", "05", "05", "05", "05",
                          "06", "06", "06", "06", "06"], processed_dataframe["coicop_division"].tolist())
        self.assertEqual(["011", "011", "022", "022", "022",
                          "123", "054", "054", "054", "054",
                          "065", "065", "065", "065", "065"], processed_dataframe["coicop_group"].tolist())
        self.assertEqual(["0112", "0112", "0223", "0223", "0223",
                          "1234", "0545", "0545", "0545", "0545",
                          "0656", "0656", "0656", "0656", "0656"], processed_dataframe["coicop_class"].tolist())
        self.assertEqual(["01120", "01120", "02231", "02231", "02231",
                          "12342", "05453", "05453", "05453", "05453",
                          "06564", "06564", "06564", "06564", "06564"], processed_dataframe["coicop_subclass"].tolist())

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

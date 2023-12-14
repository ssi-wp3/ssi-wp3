from ssi.preprocess_data import *
from ssi.synthetic_data import generate_fake_revenue_data
import unittest
import pandas as pd
import os
import datetime


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

    def test_split_month_year(self):
        self.assertEqual(("2018", "01"), split_month_year("201801"))
        self.assertEqual(("2018", "10"), split_month_year("201810"))
        self.assertEqual(("2019", "01"), split_month_year("201901"))
        self.assertEqual(("2019", "11"), split_month_year("201911"))
        self.assertEqual(("2020", "01"), split_month_year("202001"))
        self.assertEqual(("2020", "12"), split_month_year("202012"))

    def test_split_month_year_dataframe(self):
        dataframe = pd.DataFrame(
            {"month_year": ["201801", "201810", "201901", "201911", "202001", "202012"]})
        dataframe = split_month_year_column(
            dataframe, month_year_column="month_year")
        self.assertTrue(dataframe["year"].equals(
            pd.Series(["2018", "2018", "2019", "2019", "2020", "2020"])))
        self.assertTrue(dataframe["month"].equals(
            pd.Series(["01", "10", "01", "11", "01", "12"])))

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

    def test_rename_columns(self):
        dataframe = pd.DataFrame({
            "coicop_number": ["011201" for _ in range(10)],
            "product_id": [i for i in range(10)],
            "product_name": [f"product_{i}" for i in range(10)],
            "isba_number": [i for i in range(10)],
            "isba_name": [f"isba_{i}" for i in range(10)],
            "ean_number": [i for i in range(10)],
            "ean_name": [f"ean_{i}" for i in range(10)]
        })

        renamed_dataframe = rename_columns(dataframe, {
            "coicop_number": "coicop",
            "product_id": "id",
            "product_name": "name",
            "isba_number": "isba",
            "isba_name": "isba_name",
            "ean_number": "ean",
            "ean_name": "ean_name"
        })

        self.assertEqual(["coicop", "id", "name", "isba", "isba_name",
                          "ean", "ean_name"], renamed_dataframe.columns.tolist())

    def test_preprocess_data(self):
        dataframe = pd.DataFrame({
            "bg_number": [1 for _ in range(15)],
            "coicop_number": ["11201", "11201", "22312", "22312", "022312",
                              "123423", "54534", "054534", "54534", "54534",
                              "65645", "065645", "65645", "65645", "065645"],
            "month": [(datetime.date(2018, 1, 1) + datetime.timedelta(days=31*i)).strftime("%Y%m")
                      for i in range(15)],
            "product_name": [f"product_{i}" for i in range(15)],
            "isba_number": [i for i in range(15)],
            "isba_name": [f"isba_{i}" for i in range(15)],
            "ean_number": [i for i in range(15)],
            "ean_name": [f"ean_{i}" for i in range(15)]
        })

        processed_dataframe = preprocess_data(
            dataframe, columns=["bg_number", "month", "coicop_number", "ean_number", "ean_name"], coicop_column="coicop_number", product_id_column="product_id")
        self.assertEqual(len(dataframe), len(processed_dataframe))
        self.assertEqual([6] * len(processed_dataframe),
                         processed_dataframe["coicop_number"].str.len().tolist())
        self.assertEqual(["supermarket_id", "year_month", "coicop_number", "ean_number", "ean_name",
                          "year", "month",
                         "product_id", "coicop_division", "coicop_group",
                          "coicop_class", "coicop_subclass", "count"],
                         processed_dataframe.columns.tolist())
        self.assertEqual([1 for _ in range(15)],
                         processed_dataframe["supermarket_id"].tolist())

        self.assertEqual(["201801", "201802", "201803", "201804", "201805",
                          "201806", "201807", "201808", "201809", "201810",
                          "201811", "201812", "201901", "201902", "201903"],
                         processed_dataframe["year_month"].tolist())
        self.assertEqual(["2018", "2018", "2018", "2018", "2018", "2018",
                          "2018", "2018", "2018", "2018", "2018", "2018",
                          "2019", "2019", "2019"], processed_dataframe["year"].tolist())
        self.assertEqual(["01", "02", "03", "04", "05",
                          "06", "07", "08", "09", "10",
                          "11", "12", "01", "02", "03"], processed_dataframe["month"].tolist())

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
        self.assertEqual([2, 2, 3, 3, 3,
                          1, 4, 4, 4, 4,
                          5, 5, 5, 5, 5], processed_dataframe["count"].tolist())

    def test_combine_revenue_files(self):
        data_directory = os.path.join(os.getcwd(), "tests", "data")
        os.makedirs(data_directory, exist_ok=True)

        revenue_files = {
            "AH": 2,
            "Jumbo": 4,
            "Lidl": 1,
            "Plus": 3
        }

        # Generate some fake data
        filenames = []
        dataframes = []
        years = list(range(2018, 2022))
        for supermarket_name, number_of_files in revenue_files.items():
            for i in range(number_of_files):
                dataframe = generate_fake_revenue_data(
                    50, years[i], years[i])
                filenames.append(os.path.join(
                    data_directory, f"Omzet_{supermarket_name}_{i}.parquet"))
                dataframe.to_parquet(filenames[-1])
                dataframes.append(dataframe)

        combined_dataframe = combine_revenue_files(
            filenames, sort_columns=["bg_number", "month", "coicop_number"], sort_order=[True, True, True])
        self.assertEqual(500, len(combined_dataframe))

        expected_dataframe = pd.concat(dataframes).sort_values(
            by=["bg_number", "month", "coicop_number"], ascending=[True, True, True]).reset_index(drop=True)
        self.assertTrue(expected_dataframe.equals(combined_dataframe))

    def test_combine_revenue_files_in_folder(self):
        data_directory = os.path.join(os.getcwd(), "tests", "data")
        os.makedirs(data_directory, exist_ok=True)

        revenue_files = {
            "AH": 2,
            "Jumbo": 4,
            "Lidl": 1,
            "Plus": 3
        }

        # Generate some fake data
        filenames = []
        years = list(range(2018, 2022))
        for supermarket_name, number_of_files in revenue_files.items():
            for i in range(number_of_files):
                dataframe = generate_fake_revenue_data(
                    50, years[i], years[i])
                filenames.append(os.path.join(
                    data_directory, f"Omzet_{supermarket_name}_{i}.parquet"))
                dataframe.to_parquet(filenames[-1])

        number_of_files_read = 0
        for i, (supermarket_name, number_of_files) in enumerate(revenue_files.items()):
            supermarket_dataframe = combine_revenue_files_in_folder(
                data_directory, supermarket_name, sort_columns=["bg_number", "month", "coicop_number"], sort_order=[True, True, True])
            self.assertEqual(number_of_files * 50, len(supermarket_dataframe))
            expected_dataframe = pd.concat([pd.read_parquet(filename) for filename in filenames[number_of_files_read:number_of_files_read+number_of_files]]).sort_values(
                by=["bg_number", "month", "coicop_number"], ascending=[True, True, True]).reset_index(drop=True)

            self.assertTrue(expected_dataframe.equals(supermarket_dataframe))
            number_of_files_read += number_of_files

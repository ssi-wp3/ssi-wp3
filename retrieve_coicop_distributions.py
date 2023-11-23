import argparse
import pandas as pd


def get_coicop_distributions_for_filename(filename: str) -> pd.DataFrame:
    dataframe = pd.read_parquet(filename, engine="pyarrow")
    dataframe["product_id"] = dataframe["ean_name"].str.apply(hash)
    return dataframe.groupby(by=["bg_number", "month", "coicop_number"])[
        "product_id"].nunique().reset_index()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-directory",
                    help="The directory to read the parquet files from")
parser.add_argument("-o", "--output-filename", default="./coicop_distributions.csv",
                    help="The filename to write the coicop distributions to")
parser.add_argument("-d", "--delimiter", default=";",
                    help="The delimiter used in the csv files")
args = parser.parse_args()

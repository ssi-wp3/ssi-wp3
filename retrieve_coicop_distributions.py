from dotenv import load_dotenv
from pathlib import Path
import argparse
import pandas as pd
import os
import tqdm


def get_coicop_distributions_for_filename(filename: str) -> pd.DataFrame:
    dataframe = pd.read_parquet(filename, engine="pyarrow")
    dataframe["product_id"] = dataframe["ean_name"].str.apply(hash)
    return dataframe.groupby(by=["bg_number", "month", "coicop_number"])[
        "product_id"].nunique().reset_index()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-directory", default=None,
                    help="The directory to read the parquet files from")
parser.add_argument("-o", "--output-filename", default="./coicop_distributions.csv",
                    help="The filename to write the coicop distributions to")
parser.add_argument("-d", "--delimiter", default=";",
                    help="The delimiter used in the csv files")
args = parser.parse_args()

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

output_directory = os.getenv("OUTPUT_DIRECTORY")
preprocessed_files = [os.path.join(output_directory, filename)
                      for filename in os.listdir(output_directory)
                      if filename.startswith("ssi_omzet") and filename.endswith(".parquet")]


coicop_distributions = pd.concat([get_coicop_distributions_for_filename(preprocessed_file)
                                  for preprocessed_file in tqdm.tqdm(preprocessed_files)])
coicop_distributions.to_csv(
    args.output_filename, sep=args.delimiter, index=False, mode="a")

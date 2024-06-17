from pathlib import Path
from dotenv import load_dotenv
from ssi.data_exploration import ProductAnalysis
from ssi.files import get_supermarket_name, get_combined_revenue_files_in_directory
from ssi.constants import Constants
import pandas as pd
import argparse
import os
import tqdm

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument("-c", "--coicop-column", type=str, default=Constants.COICOP_LABEL_COLUMN,
                    help="Name of the column containing the coicop numbers")
args = parser.parse_args()

# load environment variables from .env file for project
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

output_directory = os.getenv("OUTPUT_DIRECTORY")

revenue_filenames = get_combined_revenue_files_in_directory(output_directory)
with tqdm.tqdm(total=len(revenue_filenames)) as progress_bar:
    for revenue_filename in revenue_filenames:
        progress_bar.set_description(
            f"Reading products from {revenue_filename}")
        features_dataframe = pd.read_parquet(
            revenue_filename, engine="pyarrow")

        supermarket_name = get_supermarket_name(revenue_filename)

        data_directory = os.path.join(output_directory, supermarket_name)
        supermarket_plot_directory = os.path.join(
            data_directory, "plots")
        os.makedirs(supermarket_plot_directory, exist_ok=True)

        product_analysis = ProductAnalysis(data_directory=data_directory,
                                           plot_directory=supermarket_plot_directory,
                                           supermarket_name=supermarket_name,
                                           coicop_level_columns=Constants.COICOP_LEVELS_COLUMNS)
        progress_bar.set_description(
            f"Analyzing products from {revenue_filename}, storing plots in {supermarket_plot_directory}, storing tables in {data_directory}")
        product_analysis.analyze_products(features_dataframe)
        progress_bar.update(1)

from pathlib import Path
from dotenv import load_dotenv
from ssi.data_exploration import ProductAnalysis
from ssi.files import get_features_files_in_directory, get_supermarket_name
from ssi.constants import Constants
import pandas as pd
import argparse
import os
import tqdm

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument("-c", "--coicop-column", type=str, default="coicop_number",
                    help="Name of the column containing the coicop numbers")
args = parser.parse_args()

# load environment variables from .env file for project
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

output_directory = os.getenv("OUTPUT_DIRECTORY")
features_directory = os.path.join(output_directory, "features")

feature_filenames = get_features_files_in_directory(features_directory)
with tqdm.tqdm(total=len(feature_filenames)) as progress_bar:
    for feature_filename in feature_filenames:
        progress_bar.set_description(
            f"Reading features from {feature_filename}")
        features_dataframe = pd.read_parquet(
            os.path.join(features_directory, feature_filename), engine="pyarrow")

        supermarket_name = get_supermarket_name(feature_filename)

        data_directory = os.path.join(output_directory, supermarket_name)
        supermarket_plot_directory = os.path.join(
            data_directory, supermarket_name)
        os.makedirs(supermarket_plot_directory, exist_ok=True)

        product_analysis = ProductAnalysis(data_directory=data_directory,
                                           plot_directory=supermarket_plot_directory,
                                           supermarket_name=supermarket_name,
                                           coicop_level_columns=Constants.COICOP_LEVELS_COLUMNS)
        progress_bar.set_description(
            f"Analyzing products from {feature_filename}, storing plots in {supermarket_plot_directory}, storing tables in {data_directory}")
        product_analysis.analyze_products(features_dataframe)
        progress_bar.update(1)

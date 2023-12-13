from ssi.feature_extraction import FeatureExtractorFactory, FeatureExtractorType
from ssi.files import get_combined_revenue_files_in_directory, get_supermarket_name
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument("-s", "--source-column", type=str, default="ean_name",
                    help="Name of the column containing the (receipt) text to extract features from")
parser.add_argument("-f", "--feature-extractors", type=str, nargs="+", default=[],
                    choices=[
                        feature_extractor_type.value for feature_extractor_type in FeatureExtractorType],
                    help="Feature extractors to use")
parser.add_argument("-b", "--batch-size", type=int,
                    default=10000, help="Batch size for feature extraction")
args = parser.parse_args()

# load environment variables from .env file for project
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

output_directory = os.getenv("OUTPUT_DIRECTORY")
features_directory = os.path.join(output_directory, "features")

print(
    f"Reading files in {output_directory} and extracting files to {features_directory}.")
if not os.path.exists(features_directory):
    print(f"Creating features directory {features_directory}")
    os.makedirs(features_directory)

feature_extractor_factory = FeatureExtractorFactory()

for combined_revenue_file in get_combined_revenue_files_in_directory(output_directory):
    print(f"Extracting features from {combined_revenue_file}")
    revenue_dataframe = pd.read_parquet(
        combined_revenue_file, engine="pyarrow")
    supermarket_name = get_supermarket_name(combined_revenue_file)
    feature_extractor_factory.extract_all_features_and_save(
        dataframe=revenue_dataframe,
        source_column=args.source_column,
        supermarket_name=supermarket_name,
        output_directory=features_directory,
        feature_extractors=[FeatureExtractorType(
            feature_extractor) for feature_extractor in args.feature_extractors],
        batch_size=args.batch_size)

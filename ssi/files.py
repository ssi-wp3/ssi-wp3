from typing import List
import os


def get_feature_filename(feature_extractor_type: str, supermarket_name: str) -> str:
    return f"ssi_{supermarket_name.lower()}_{feature_extractor_type}_features.parquet"


def get_features_files_in_directory(directory: str, extension: str = ".parquet") -> List[str]:
    return [filename
            for filename in os.listdir(directory)
            if filename.startswith("ssi_") and "_features" in filename and filename.endswith(extension)]


def get_combined_revenue_filename(supermarket_name: str) -> str:
    return f"ssi_{supermarket_name.lower()}_revenue.parquet"


def get_combined_revenue_files_in_directory(directory: str, extension: str = ".parquet") -> List[str]:
    return [os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.startswith("ssi_") and filename.endswith(extension) and "revenue" in filename]


def get_supermarket_name(filename: str) -> str:
    return os.path.basename(filename).split("_")[1]

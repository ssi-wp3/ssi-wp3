from typing import List
import os


def get_revenue_files_in_folder(data_directory: str, supermarket_name: str, filename_prefix: str = "Omzet") -> List[str]:
    return [os.path.join(data_directory, filename) for filename in os.listdir(
        data_directory) if filename.startswith(filename_prefix) and supermarket_name in filename]


def get_feature_filename(feature_extractor_type: str) -> str:
    return f"ssi_features_{feature_extractor_type}.parquet"

def get_features_files_in_directory(directory: str, extension: str = ".parquet") -> List[str]:
    return [filename 
            for filename in os.listdir(directory) 
            if filename.startswith("ssi_features_") and filename.endswith(extension)]

def get_combined_revenue_filename(supermarket_name: str) -> str:
    return f"ssi_{supermarket_name.lower()}_revenue.parquet"

def get_combined_revenue_files_in_directory(directory: str, extension: str = ".parquet") -> List[str]:
    return [os.path.join(directory, filename) 
            for filename in os.listdir(directory) 
            if filename.startswith("ssi_") and filename.endswith(extension) and "revenue" in filename]
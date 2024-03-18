from typing import List
import os


def get_feature_filename(feature_extractor_type: str,
                         supermarket_name: str,
                         project_prefix: str = "ssi",
                         feature_file_suffix: str = "features") -> str:
    return f"{project_prefix}_{supermarket_name.lower()}_{feature_extractor_type}_{feature_file_suffix}.parquet"


def get_features_files_in_directory(directory: str,
                                    project_prefix: str = "ssi",
                                    feature_file_suffix: str = "features",
                                    extension: str = ".parquet") -> List[str]:
    return [filename
            for filename in os.listdir(directory)
            if filename.startswith(f"{project_prefix}_") and f"_{feature_file_suffix}" in filename and filename.endswith(extension)]

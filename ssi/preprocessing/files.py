from typing import List, Optional
import os


def get_revenue_files_in_folder(data_directory: str, supermarket_name: str, filename_prefix: str = "Omzet", file_type: Optional[str] = None) -> List[str]:
    return [os.path.join(data_directory, filename)
            for filename in os.listdir(data_directory)
            if filename.startswith(filename_prefix) and supermarket_name in filename and (not file_type or filename.endswith(file_type))]

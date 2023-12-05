from ssi.preprocess_data import save_combined_revenue_files
from pathlib import Path
from dotenv import load_dotenv
import argparse
import os


parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument("-s", "--supermarket-name", type=str, required=True, help="Name of the supermarket")
parser.add_argument("-c", "--coicop-column", type=str, default="coicop_number", help="Name of the column containing the coicop numbers")
parser.add_argument("-p", "--product-id-column", type=str, default="product_id", help="Name of the column containing the product ids")
parser.add_argument("-pd", "--product-description-column", type=str, default="ean_name", help="Name of the column containing the product descriptions")
parser.add_argument("-clv", "--coicop-level-columns", nargs="+", type=str, default=["coicop_division", "coicop_group", "coicop_class", "coicop_subclass"], help="Names of the columns containing the coicop levels")
parser.add_argument("-sc", "--selected-columns", nargs="+", type=str, default=["bg_number", "month", "coicop_number", "ean_name", "amount"], help="Names of the columns to select")
parser.add_argument("-fp", "--filename-prefix", type=str, default="Omzet", help="Prefix of the revenue files")
args = parser.parse_args()

# load environment variables from .env file for project
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

output_directory = os.getenv("OUTPUT_DIRECTORY")
log_directory = os.path.join(output_directory, "logs")
if not os.path.exists(log_directory):
    print(f"Creating log directory {log_directory}")
    os.makedirs(log_directory)

save_combined_revenue_files(data_directory=output_directory, 
                            output_filename="ssi_", 
                            supermarket_name=args.supermarket_name,
                            log_directory=log_directory,
                            selected_columns=args.selected_columns,
                            coicop_column=args.coicop_column,
                            product_id_column=args.product_id_column,
                            product_description_column=args.product_description_column,
                            coicop_level_columns=args.coicop_level_columns,
                            filename_prefix=args.filename_prefix
                            )
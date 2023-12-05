from dotenv import load_dotenv
from pathlib import Path
from ssi.synthetic_data import generate_fake_revenue_data
import argparse
import os

parser = argparse.ArgumentParser(
    description='Generate synthetic data for the SSI project.')
parser.add_argument('-n', '--num_rows', type=int, default=1000,
                    help='Number of rows to generate')
parser.add_argument('-dr', '--date-range', type=str,
                    default='2020-2023', help='Date range to generate data for')
parser.add_argument('-o', '--output', type=str,
                    default='synthetic_data.parquet', help='Output file name')
parser.add_argument('-s', '--supermarket-id', type=str, default=None, help='Supermarket id to use when generating the data')
args = parser.parse_args()

# Define the number of rows in the dataset
num_rows = args.num_rows
start_date, end_date = (int(date) for date in args.date_range.split('-'))
# load environment variables from .env file for project
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)


output_directory = os.getenv("OUTPUT_DIRECTORY")
if not os.path.exists(output_directory):
    print(f"Creating output directory {output_directory}")
    os.makedirs(output_directory)

print(f"Writing synthetic data to {output_directory}/{args.output}")
dataframe = generate_fake_revenue_data(num_rows, start_date, end_date, args.supermarket_id)
print(dataframe.head())
dataframe.to_parquet(f"{output_directory}/{args.output}", engine='pyarrow')

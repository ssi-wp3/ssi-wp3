from dotenv import load_dotenv
from pathlib import Path
from ssi.synthetic_data import generate_fake_revenue_data
import argparse
import os

parser = argparse.ArgumentParser(
    description='Generate synthetic data for the SSI project.')
parser.add_argument('--num_rows', type=int, default=1000,
                    help='Number of rows to generate')
parser.add_argument('-dr', '--date-range', type=str,
                    default='2020-2023', help='Date range to generate data for')
parser.add_argument('-o', '--output', type=str,
                    default='synthetic_data.parquet', help='Output file name')
args = parser.parse_args()

# Define the number of rows in the dataset
num_rows = args.num_rows
start_date, end_date = (int(date) for date in args.date_range.split('-'))
# load environment variables from .env file for project
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)


output_directory = os.getenv("OUTPUT_DIRECTORY")
print(f"Writing synthetic data to {output_directory}/{args.output}")
dataframe = generate_fake_revenue_data(num_rows, start_date, end_date)
print(dataframe.head())
dataframe.to_parquet(f"{output_directory}/{args.output}", engine='pyarrow')

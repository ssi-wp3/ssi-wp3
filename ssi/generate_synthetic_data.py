import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from faker import Faker
import argparse
import os


parser = argparse.ArgumentParser(description='Generate synthetic data for the SSI project.')
parser.add_argument('--num_rows', type=int, default=1000, help='Number of rows to generate')
parser.add_argument('-dr', '--date-range', type=str, default='2020-2023', help='Date range to generate data for')
parser.add_argument('-o', '--output', type=str, default='synthetic_data.parquet', help='Output file name')
args = parser.parse_args()

# Define the number of rows in the dataset
num_rows = args.num_rows
start_date, end_date = (int(date) for date in args.date_range.split('-'))

# Initialize Faker
fake = Faker()

# Generate synthetic data
bg_number = np.random.randint(100000, 999999, num_rows)  # 6-digit supermarket identifier
month = np.random.choice([f"{year}{month:02d}" for year in range(start_date, end_date) for month in range(1, 13)], num_rows)  # Year and month
coicop_number = [f"{i:06d}" for i in np.random.randint(0, 999999, num_rows)]  # 6-digit COICOP label
coicop_name = [fake.catch_phrase() for _ in range(num_rows)]  # Product names
ean_number = [fake.ean() for _ in range(num_rows)]  # EAN product number
ean_name = [fake.bs() for _ in range(num_rows)]  # Product descriptions

# Create a DataFrame
df = pd.DataFrame({
    'bg_number': bg_number,
    'month': month,
    'coicop_number': coicop_number,
    'coicop_name': coicop_name,
    'ean_number': ean_number,
    'ean_name': ean_name
})

# load environment variables from .env file for project
dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)
output_directory = os.getenv("OUTPUT_DIRECTORY")

df.to_parquet(f"{output_directory}/{args.output}", engine='pyarrow')
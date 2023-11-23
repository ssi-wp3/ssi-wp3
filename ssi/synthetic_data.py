import pandas as pd
import numpy as np
from faker import Faker


def read_coicop_2018_data() -> pd.DataFrame:
    coicop_data = pd.read_csv('./coicop_2018', sep=",").rename(
        columns={"CODE_COICOP_2018": "coicop_number", "HEADING_COICOP_2018": "description"})

    coicop_data["coicop_number"] = coicop_data["coicop_number"].str.replace(
        ".", "")
    coicop_data["coicop_level"] = coicop_data["coicop_number"].str.len() - 1
    return coicop_data


def generate_fake_coicop_2018() -> str:
    """Generate fake COICOP data for the SSI project.

    Returns:
        str: A string with fake COICOP data
    """
    coicop_data = read_coicop_2018_data()

    return fake.numerify(text="######")


def generate_fake_revenue_data(num_rows: int, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate fake revenue data for the SSI project.

    Args:
        num_rows (int): The number of rows to generate
        start_date (str): The start date of the data
        end_date (str): The end date of the data

    Returns:
        pd.DataFrame: A DataFrame with fake revenue data
    """
    fake = Faker()

    # Generate synthetic data
    # 6-digit supermarket identifier
    bg_number = np.random.randint(100000, 999999, num_rows)
    month = np.random.choice([f"{year}{month:02d}" for year in range(
        start_date, end_date) for month in range(1, 13)], num_rows)  # Year and month
    coicop_number = [f"{i:06d}" for i in np.random.randint(
        0, 999999, num_rows)]  # 6-digit COICOP label
    coicop_name = [fake.catch_phrase()
                   for _ in range(num_rows)]  # Product names
    ean_number = [fake.ean() for _ in range(num_rows)]  # EAN product number
    ean_name = [fake.bs() for _ in range(num_rows)]  # Product descriptions

    # Create a DataFrame
    return pd.DataFrame({
        'bg_number': bg_number,
        'month': month,
        'coicop_number': coicop_number,
        'coicop_name': coicop_name,
        'ean_number': ean_number,
        'ean_name': ean_name
    })

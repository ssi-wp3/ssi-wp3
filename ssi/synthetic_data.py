from typing import List, Optional
from faker import Faker
from .preprocess_data import preprocess_data
import pandas as pd
import numpy as np
import random


def read_coicop_2018_data() -> pd.DataFrame:
    coicop_data = pd.read_csv('./ssi/coicop_2018.txt', sep=",").rename(
        columns={"CODE_COICOP_2018": "coicop_number", "HEADING_COICOP_2018": "description"})

    coicop_data["coicop_number"] = coicop_data["coicop_number"].str.replace(
        ".", "")
    coicop_data["coicop_level"] = coicop_data["coicop_number"].str.len() - 1
    return coicop_data


def generate_fake_coicop_2018(num_rows: int) -> str:
    """Generate fake COICOP data for the SSI project.

    Returns:
        str: A string with fake COICOP data
    """
    coicop_data = read_coicop_2018_data()
    coicop_data = coicop_data[coicop_data["coicop_level"] == 5]
    return coicop_data.sample(num_rows, replace=True)[["coicop_number", "description"]]


def generate_supermarked_ids(num_rows: int, supermarket_id: Optional[str] = None) -> List[str]:
    """Generate fake supermarked ids for the SSI project.

    Args:
        num_rows (int): The number of rows to generate
    """
    supermarked_ids = ["995001", "995002",
                       "995003"] if not supermarket_id else [supermarket_id]
    return random.choices(supermarked_ids, k=num_rows)


def generate_dates(num_rows: int, start_date: str, end_date: str) -> List[str]:
    """Generate fake dates for the SSI project.

    Args:
        num_rows (int): The number of rows to generate
        start_date (str): The start date of the data
        end_date (str): The end date of the data
    """
    dates = [f"{year}{month:02d}"
             for year in range(start_date, end_date+1)
             for month in range(1, 13)]
    return sorted(random.choices(dates, k=num_rows))


def generate_fake_revenue_data(num_rows: int, start_date: int, end_date: int, supermarket_id: Optional[str] = None) -> pd.DataFrame:
    """Generate fake revenue data for the SSI project.

    Args:
        num_rows (int): The number of rows to generate
        start_date (int): The start date of the data
        end_date (int): The end date of the data

    Returns:
        pd.DataFrame: A DataFrame with fake revenue data
    """
    fake = Faker()

    # Generate synthetic data
    # 6-digit supermarket identifier
    bg_number = generate_supermarked_ids(num_rows, supermarket_id)
    month = generate_dates(num_rows, start_date, end_date)  # Year and month
    fake_coicop_data = generate_fake_coicop_2018(
        num_rows)  # COICOP product code
    coicop_number = fake_coicop_data["coicop_number"]
    coicop_name = fake_coicop_data["description"]
    ean_number = [fake.ean() for _ in range(num_rows)]  # EAN product number
    ean_name = [fake.bs() for _ in range(num_rows)]  # Product descriptions

    # Create a DataFrame
    fake_data_df = pd.DataFrame({
        'bg_number': bg_number,
        'month': month,
        'coicop_number': coicop_number,
        'coicop_name': coicop_name,
        'ean_number': ean_number,
        'ean_name': ean_name,
        'revenue': np.random.randint(100, 1000, size=num_rows),
        'amount': np.random.randint(1, 10, size=num_rows)
    })
    return fake_data_df.sort_values(by=["bg_number", "month", "coicop_number"]).reset_index(drop=True)


def generate_fake_data_with_coicop_levels(num_rows: int, start_date: int, end_date: int) -> pd.DataFrame:
    dataframe = generate_fake_revenue_data(num_rows, start_date, end_date)
    dataframe = preprocess_data(dataframe, ["bg_number", "month", "coicop_number",
                                            "coicop_name", "ean_number", "ean_name"],
                                "coicop_number", "product_id", "ean_name", column_mapping={"bg_number": "supermarket_id", "month": "year_month"})
    return dataframe

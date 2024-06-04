from ssi.analysis.revenue import product_revenue_versus_lifetime, product_lifetime_in_periods, total_revenue_per_product
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-filename", type=str,
                    required=True, help="The input filename in parquet format")
parser.add_argument("-p", "--period-column", default="year_month", type=str,
                    help="The name of the period column")
parser.add_argument("-e", "--product-id-column", default="ean_number", type=str,
                    help="The name of the product id column")
parser.add_argument("-a", "--amount-column", default="amount", type=str,
                    help="The name of the amount column")
parser.add_argument("-r", "--revenue-column", default="revenue", type=str,
                    help="The name of the revenue column")
args = parser.parse_args()

args = parser.parse_args()

df = pd.read_parquet(args.input_filename, engine='pyarrow')

product_lifetime_df = product_lifetime_in_periods(
    df, args.period_column, args.product_id_column)
product_lifetime_df[args.product_id_column] = product_lifetime_df[args.product_id_column].astype(
    str)
print(product_lifetime_df.head())


revenue_per_product_df = total_revenue_per_product(
    df, args.product_id_column, args.amount_column, args.revenue_column)
revenue_per_product_df[args.product_id_column] = revenue_per_product_df[args.product_id_column].astype(
    str)
print(revenue_per_product_df.head())

print(pd.merge(product_lifetime_df, revenue_per_product_df, on=args.product_id_column))

revenue_lifetime_df = product_revenue_versus_lifetime(df,
                                                      period_column=args.period_column,
                                                      product_id_column=args.product_id_column,
                                                      amount_column=args.amount_column,
                                                      revenue_column=args.revenue_column)

print(revenue_lifetime_df)

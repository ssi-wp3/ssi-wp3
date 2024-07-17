import os
import pandas as pd 

import config

# columns
# 'store_id', 'year_month', 'ean_name', 'year', 'month',
# 'revenue', 'amount',  
# 'rep_id', 'product_id',
# 'isba_number', 'isba_name', 'esba_number', 'esba_name',
# 'coicop_name', 'coicop_number', 'coicop_level_1', 'coicop_level_2', 'coicop_level_3', 'coicop_level_4', 'store_name',
# 'receipt_text'

LOAD_COLUMNS = [
 'ean_name',
 'product_id',
 'receipt_text', 
 'store_id',
 'store_name',
 'coicop_name',
 'coicop_number',
 'coicop_level_1',
 'coicop_level_2',
 'coicop_level_3',
 'coicop_level_4',
 'year_month',
 'year',
 'month',
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  # drop records if no receipt texts
  df = df[df["receipt_text"].notna()]

  # drop 99999 (ununsed category)
  df = df[df["coicop_number"] != "999999"] 

  df = df.sort_values("year_month", ascending=False)

  # group by ean name, receipt text, and coicop
  groupby_cols = ["ean_name", "receipt_text", "coicop_number"]

  assert all(col in df.columns for col in groupby_cols)

  #
  # assign weights based on count
  #

  # count the number of duplicate records (by groupby_cols)
  df = df.set_index(groupby_cols)
  df["count"] = df.groupby(groupby_cols).size()
  df = df.reset_index()

  df = df.drop_duplicates(subset=groupby_cols, keep="first") # sorted by date, most recent first, so keep only the newest entry

  # normalize the count by the total count of the receipt text
  count_weight_col_name = "weight__count"

  df = df.set_index(groupby_cols)
  df[count_weight_col_name] = df["count"] / df.groupby("receipt_text")["count"].sum()
  df = df.reset_index()

  return df

def split_dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  # take 2023-05 and 2023-06 as test set
  query_test = (df["year_month"] == "202305") | (df["year_month"] == "202306")

  df_test = df[query_test]
  df_dev  = df[~query_test]

  return df_dev, df_test

def save_dataset(df: pd.DataFrame, out_fn: str) -> None:
  if not os.path.isdir(config.OUTPUT_DATA_DIR):
    os.mkdir(config.OUTPUT_DATA_DIR)

  output_path = os.path.join(config.OUTPUT_DATA_DIR, out_fn)
  df.to_parquet(output_path)

if __name__ == "__main__":
  df_stores = [] # all stores

  print("Loading datasets...")
  for store_name in config.STORES:
    dataset_fn = f"ssi_{store_name}_revenue.parquet"
    dataset_path = os.path.join(config.SOURCE_DATA_DIR, dataset_fn)

    df = pd.read_parquet(dataset_path, columns=LOAD_COLUMNS)
    df_stores.append(df)

  df_stores = pd.concat(df_stores)

  df_stores_dev, df_stores_test = split_dev_test(df_stores)

  print("Preprocessing datasets...")
  df_stores_dev = preprocess(df_stores_dev)

  print("Saving datasets...")
  out_dev_fn = f"dev_{'_'.join(config.STORES)}.parquet"
  save_dataset(df_stores_dev, out_fn=out_dev_fn)

  out_test_fn = f"test_{'_'.join(config.STORES)}.parquet"
  save_dataset(df_stores_test, out_fn=out_test_fn)


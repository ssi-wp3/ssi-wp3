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

def preprocess(df: pd.DataFrame, assign_weights=False) -> pd.DataFrame:
  df = df[df["receipt_text"].notna()]
  df = df[df["receipt_text"].str.len() > 2]

  # drop 99999 (ununsed category)
  df = df[df["coicop_number"] != "999999"] 

  df = df.sort_values("year_month", ascending=False)

  # 
  # remove duplicate records based on groupby_cols
  # 
  groupby_cols = ["ean_name", "receipt_text", "coicop_number"] # group by ean name, receipt text, and coicop

  assert all(col in df.columns for col in groupby_cols)

  # count the number of duplicate records (by groupby_cols)
  df = df.set_index(groupby_cols)
  df["count"] = df.groupby(groupby_cols).size()
  df = df.reset_index()

  df = df.drop_duplicates(subset=groupby_cols, keep="first") # sorted by date, most recent first, so keep only the newest entry

  if assign_weights:
    #
    # assign weights based on count
    #

    # normalize the count by the total count of the receipt text
    count_weight_col_name = "weight__count"

    df = df.set_index(groupby_cols)
    df[count_weight_col_name] = df["count"] / df.groupby("receipt_text")["count"].sum()
    df = df.reset_index()

  return df

def split_dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  # take 2023-05 and 2023-06 as test set
  query_test = (df["year_month"] == "202306") | (df["year_month"] == "202307") | (df["year_month"] == "202308")

  df_test = df[query_test]
  df_dev  = df[~query_test]

  return df_dev, df_test

def write_dataset(df: pd.DataFrame, out_fn: str, write_metadata=True) -> None:
  if not os.path.isdir(config.OUTPUT_DATA_DIR):
    os.mkdir(config.OUTPUT_DATA_DIR)

  output_path = os.path.join(config.OUTPUT_DATA_DIR, out_fn)
  df.to_parquet(output_path)

  if write_metadata:
    metadata_out_fn, _ = os.path.splitext(out_fn)
    metadata_out_fn = f"{metadata_out_fn}_metadata.txt"
    metadata_output_path = os.path.join(config.OUTPUT_DATA_DIR, metadata_out_fn)

    out = (
      "==================================\n"
      f"{out_fn}\n"
      "===================================\n\n"
    )
    
    out += f"Num. of Rows   : {df.shape[0]}\n"
    out += f"Num. of Columns: {df.shape[1]}\n"
    out += f"Min. Period    : {df['year_month'].min()}\n"
    out += f"Max. Period    : {df['year_month'].max()}\n"
    out += f"Stores         : {', '.join(df['store_name'].unique())}\n\n"

    # add column data
    out += "Columns:\n"

    for col_name in df.columns:
      out += f"\t{col_name}\n"

    out += (
    "\n----------------------------------\n"
    "Rows per COICOP Level 1 Categories\n"
    "----------------------------------\n"
    )

    # add coicop data
    coicop_counts = df["coicop_level_1"].value_counts()
    coicop_counts = coicop_counts.sort_index()

    for coicop_level, counts in coicop_counts.items():
      out += f"\t{coicop_level}: {counts:<10} {(counts / df.shape[0]):.4f}\n"

    with open(metadata_output_path, 'w') as fp:
      fp.write(out)
      
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
  df_stores_dev  = preprocess(df_stores_dev, assign_weights=True)
  df_stores_test = preprocess(df_stores_test)

  print("Saving datasets...")
  out_dev_fn = f"dev_{'_'.join(config.STORES)}.parquet"
  write_dataset(df_stores_dev, out_fn=out_dev_fn)

  out_test_fn = f"test_{'_'.join(config.STORES)}.parquet"
  write_dataset(df_stores_test, out_fn=out_test_fn)


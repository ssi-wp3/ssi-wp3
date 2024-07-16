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
  # drop 99999
  df = df[df["coicop_number"] != "999999"]

  df = df.sort_values("year_month", ascending=False)

  # group by ean name, receipt text, and coicop
  groupby_cols = ["ean_name", "receipt_text", "coicop_number"]

  assert all(col in df.columns for col in groupby_cols)

  df = df.set_index(groupby_cols)

  df["count"] = df.groupby(groupby_cols).size()
  df = df.reset_index()
  df = df.drop_duplicates(subset=groupby_cols, keep="first")

  df["count_weight_col_name"] = 1.0
  import pdb; pdb.set_trace()
  print(df.groupby("receipt_text")["count"] / df.groupby("receipt_text")["count"].sum())




  #
  # add weights
  #
  # weight_col_name
  # date_weight_col_name
  # count_weight_col_name

  # add 

  return df

#def 

if __name__ == "__main__":
  df_stores = [] # all stores

  for store_name in config.STORES:
    dataset_fn = f"ssi_{store_name}_revenue.parquet"
    dataset_path = os.path.join(config.SOURCE_DATA_DIR, dataset_fn)

    df = pd.read_parquet(dataset_path, columns=LOAD_COLUMNS)
    preprocess(df)
    import pdb; pdb.set_trace()
    df_stores.append(df)

  df_stores = pd.concat(df_stores)

  df_stores_preprocessed = preprocess(df_stores)

  # 

  # dfs[supermarket_name] = pd.read_parquet(data_file_path, columns=LOAD_COLUMNS)


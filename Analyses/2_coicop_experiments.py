import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

import config
from experimental_setups import experiments

def get_X_y(df: pd.DataFrame, predict_level: int) -> tuple[pd.Series, pd.Series]:
  X_col = "receipt_text" # text column
  y_col = f"coicop_level_{predict_level}"

  X = df[X_col]
  y = df[y_col]

  return X, y

def get_coicop_level_label(y: pd.Series, level: int) -> np.ndarray:
  if not isinstance(y, pd.Series):
    if isinstance(y, np.ndarray):
      y = pd.Series(y)
    elif isinstance(y, pd.DataFrame):
      print("Pandas DataFrame is given instead of Pandas Series.")
      exit(1)
    else:
      print("Unknown type given.")
      exit(1)
     
  if level > 5:
    print("Undefined COICOP level!")
    return

  label_pos_stop = level + 1
  ret = y.str.slice(start=0, stop=label_pos_stop) # @todo do this in numpy only
  ret = ret.to_numpy()
  return ret

if __name__ == "__main__":
  if config.SAMPLE_N is not None:
    df_dev_fn  = "sample_dev_lidl_ah_jumbo_plus.parquet"
    df_test_fn = "sample_test_lidl_ah_jumbo_plus.parquet"

    results_fn = "sample_results.csv"

  else:
    df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
    df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

    results_fn = "results.csv"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  while len(experiments) > 0:
    exp = experiments.pop(0)

    exp.eval_pipeline(df_dev, df_test, get_X_y, hierarchical_split_func=get_coicop_level_label)
    exp.write_results(out_fn=results_fn)

  exit(0)


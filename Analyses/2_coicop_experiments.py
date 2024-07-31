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

def get_coicop_level_label(y: pd.Series, level: int) -> pd.Series:
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
  return y.str.slice(start=0, stop=label_pos_stop)

if __name__ == "__main__":
  df_dev_fn  = "dev_lidl_ah_jumbo_plus.parquet"
  df_test_fn = "test_lidl_ah_jumbo_plus.parquet"

  df_dev_path  = os.path.join(config.OUTPUT_DATA_DIR, df_dev_fn)
  df_test_path = os.path.join(config.OUTPUT_DATA_DIR, df_test_fn)

  df_dev  = pd.read_parquet(df_dev_path)
  df_test = pd.read_parquet(df_test_path)

  for exp in experiments:
    exp.eval_pipeline(df_dev, df_test, get_X_y, hierarchical_split_func=get_coicop_level_label)
    exp.write_results(out_fn="results.csv")

  exit(0)


#  pipe = Pipeline([
#    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#    #("tfidf", TfidfTransformer()),
#    #("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#    ("clf", SGDClassifier(loss="log_loss", n_jobs=6, alpha=0.00000001)),
#  ])
#
#  hc = HierarchicalClassifier(pipe, depth=predict_coicop)
#  hc.fit(X_dev, y_dev, get_coicop_level_label)

  hc = Pipeline([
    #("hv", HashingVectorizer(input="content", binary=False, norm=None, analyzer="char", ngram_range=(3, 6), n_features=150_000)),
    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
    ("clf", SGDClassifier(loss="perceptron", n_jobs=8, random_state=config.SEED)),
    #("clf", SGDClassifier(loss="log_loss", n_jobs=6, alpha=0.00000001)),
  ])


  eval_pipeline(hc, df_dev, df_test, predict_level=predict_coicop, get_upper_level_func=get_coicop_level_label)

#  hc = Pipeline([
#      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#      ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#  ])
#  hc.fit(X_dev, y_dev[f"coicop_level_{predict_coicop}"])



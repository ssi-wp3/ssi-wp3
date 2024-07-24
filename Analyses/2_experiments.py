import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

import config
from hierarchical.models import HierarchicalClassifier 
from hierarchical.metrics import hierarchical_precision_score, hierarchical_recall_score, hierarchical_f1_score

def eval_pipeline(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, predict_coicop, sample_weight=None) -> None:
  #pipe.fit(X_dev, y_dev, clf__sample_weight=sample_weight)

  y_pred = pipe.predict(X_test)
  #y_proba = pipe.predict_proba(X_test)

#  print(accuracy_score(y_test, y_pred))
#  print(balanced_accuracy_score(y_test, y_pred))
#  print(classification_report(y_test, y_pred))

  print("hr precision:", hierarchical_precision_score(y_test, y_pred, get_coicop_level_label, predict_coicop))
  print("hr recall:", hierarchical_recall_score(y_test, y_pred, get_coicop_level_label, predict_coicop))
  print("hr f1:", hierarchical_f1_score(y_test, y_pred, get_coicop_level_label, predict_coicop))

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

  predict_coicop = 5

  X_dev, y_dev = get_X_y(df_dev, predict_coicop)
  X_test, y_test = get_X_y(df_test, predict_coicop)

  pipe = Pipeline([
    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
    ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
  ])

  hc = HierarchicalClassifier(pipe, depth=predict_coicop)
  hc.fit(X_dev, y_dev, get_coicop_level_label)

#  hc = Pipeline([
#    ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#    ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#  ])

  #hc.fit(X_dev, y_dev)

  eval_pipeline(hc, X_test, y_test, predict_coicop=predict_coicop)


#  hc = Pipeline([
#      ("hv", HashingVectorizer(input="content", binary=True, dtype=bool)),
#      ("clf", SGDClassifier(loss="perceptron", n_jobs=6)),
#  ])
#  hc.fit(X_dev, y_dev[f"coicop_level_{predict_coicop}"])


